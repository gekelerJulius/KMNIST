import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from CONFIG import DATA, ENSEMBLE, MODEL, PSEUDO_LABELS, SUBMISSION
from kmnist.data import LabeledImageFolderDataset
from kmnist.data.loaders import build_loader
from kmnist.data.transforms import build_test_transform
from kmnist.models import Autoencoder
from kmnist.submission.dataset import SubmissionImageDataset
from kmnist.submission.embeddings import compute_embeddings_and_logits, normalize_rows
from kmnist.submission.prediction import ensemble_predict_labels
from kmnist.submission.writer import write_diagnostics, write_json
from kmnist.utils.checkpoints import resolve_checkpoint
from kmnist.utils.device import get_device
from kmnist.utils.paths import labeled_dir, labels_csv_path, sample_submission_path, timestamped_pseudo_label_dir


@dataclass(frozen=True)
class PseudoLabelArtifacts:
    pseudo_labels_path: Path
    diagnostics_path: Path
    summary_path: Path
    selected_rows: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate high-confidence pseudo labels from unlabeled KMNIST images."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a Lightning checkpoint. Defaults to the best embedding checkpoint.",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=sample_submission_path(),
        help="Sample submission CSV whose ImagePath order and formatting should be preserved.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where pseudo-label artifacts will be written. Defaults to a timestamped output directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=SUBMISSION.batch_size,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=SUBMISSION.num_workers,
        help="DataLoader worker count. Use 0 if multiprocessing is problematic.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=PSEUDO_LABELS.max_distance,
        help="Maximum nearest-prototype distance for static threshold mode.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=PSEUDO_LABELS.min_margin,
        help="Minimum prototype margin for static threshold mode.",
    )
    parser.add_argument(
        "--class-cap",
        type=int,
        default=PSEUDO_LABELS.class_cap,
        help="Maximum selected pseudo labels per class.",
    )
    parser.add_argument(
        "--total-cap",
        type=int,
        default=PSEUDO_LABELS.total_cap,
        help="Maximum selected pseudo labels across all classes. Uses class-balanced selection when set.",
    )
    parser.add_argument(
        "--previous-pseudo-labels",
        type=Path,
        default=None,
        help="Existing pseudo-label CSV to keep append-only while adding new labels up to --total-cap.",
    )
    parser.add_argument(
        "--prototype-margin-gate",
        type=float,
        default=ENSEMBLE.prototype_margin_gate,
        help="Prototype margin required to trust prototype prediction on disagreement.",
    )
    parser.add_argument(
        "--classifier-confidence-gate",
        type=float,
        default=ENSEMBLE.classifier_confidence_gate,
        help="Classifier confidence required to trust classifier prediction on disagreement.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("labeled_relative", "static"),
        default=PSEUDO_LABELS.threshold_mode,
        help="Use class-specific labeled-relative limits or fixed static limits.",
    )
    parser.add_argument(
        "--labeled-distance-quantile",
        type=float,
        default=PSEUDO_LABELS.labeled_distance_quantile,
        help="Labeled own-prototype distance quantile for relative calibration.",
    )
    parser.add_argument(
        "--labeled-distance-scale",
        type=float,
        default=PSEUDO_LABELS.labeled_distance_scale,
        help="Multiplier applied to labeled distance quantiles for relative limits.",
    )
    parser.add_argument(
        "--labeled-margin-quantile",
        type=float,
        default=PSEUDO_LABELS.labeled_margin_quantile,
        help="Labeled true-class margin quantile for relative calibration.",
    )
    parser.add_argument(
        "--labeled-margin-scale",
        type=float,
        default=PSEUDO_LABELS.labeled_margin_scale,
        help="Multiplier applied to labeled margin quantiles for relative limits.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable deterministic classifier test-time augmentation for pseudo-label confidence.",
    )
    return parser.parse_args()


def selection_mask(
    result,
    max_distances: float | np.ndarray,
    min_margins: float | np.ndarray,
) -> np.ndarray:
    agreement_mask = result.prototype_labels == result.classifier_labels
    distance_mask = result.prototype_distances <= max_distances
    margin_mask = result.prototype_margins >= min_margins
    return agreement_mask & distance_mask & margin_mask


def compute_labeled_reference(
    model: Autoencoder,
    device,
    batch_size: int,
    num_workers: int,
    use_tta: bool = SUBMISSION.use_tta,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labeled_dataset = LabeledImageFolderDataset(
        labeled_dir(),
        labels_csv_path(),
        transform=build_test_transform(),
    )
    labeled_loader = build_loader(labeled_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    labeled_embeddings, labeled_logits, labels = compute_embeddings_and_logits(
        model,
        labeled_loader,
        device,
        "Embedding labeled reference",
        use_tta=use_tta,
    )
    normalized_embeddings = normalize_rows(labeled_embeddings)
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.array(sorted(np.unique(labels)), dtype=np.int64)
    prototypes = []
    for class_id in classes:
        prototype = normalized_embeddings[labels == class_id].mean(axis=0)
        prototype = prototype / np.clip(np.linalg.norm(prototype), a_min=1e-12, a_max=None)
        prototypes.append(prototype)
    return classes, np.stack(prototypes, axis=0), labeled_embeddings, labels


def labeled_relative_thresholds(
    labeled_embeddings: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    prototypes: np.ndarray,
    distance_quantile: float,
    distance_scale: float,
    margin_quantile: float,
    margin_scale: float,
) -> dict[int, dict[str, float]]:
    normalized_embeddings = normalize_rows(labeled_embeddings)
    distances = 1 - normalized_embeddings @ prototypes.T
    thresholds = {}
    class_to_index = {int(class_id): index for index, class_id in enumerate(classes)}
    for class_id in classes:
        class_id_int = int(class_id)
        class_index = class_to_index[class_id_int]
        class_mask = labels == class_id_int
        own_distances = distances[class_mask, class_index]
        diff_distances = np.delete(distances[class_mask], class_index, axis=1)
        true_class_margins = diff_distances.min(axis=1) - own_distances
        distance_limit = float(np.quantile(own_distances, distance_quantile) * distance_scale)
        margin_limit = float(max(0.0, np.quantile(true_class_margins, margin_quantile) * margin_scale))
        thresholds[class_id_int] = {
            "max_distance": distance_limit,
            "min_margin": margin_limit,
            "labeled_distance_quantile": float(np.quantile(own_distances, distance_quantile)),
            "labeled_margin_quantile": float(np.quantile(true_class_margins, margin_quantile)),
        }
    return thresholds


def thresholds_for_predictions(labels: np.ndarray, thresholds: dict[int, dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    max_distances = np.array([thresholds[int(label)]["max_distance"] for label in labels], dtype=np.float32)
    min_margins = np.array([thresholds[int(label)]["min_margin"] for label in labels], dtype=np.float32)
    return max_distances, min_margins


def cap_selected_indices(
    candidate_mask: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    margins: np.ndarray,
    classifier_confidences: np.ndarray,
    class_cap: int,
) -> np.ndarray:
    selected_indices = []
    for class_id in sorted(np.unique(labels[candidate_mask]).tolist()):
        class_indices = np.where(candidate_mask & (labels == class_id))[0]
        order = np.lexsort(
            (
                -classifier_confidences[class_indices],
                -margins[class_indices],
                distances[class_indices],
            )
        )
        selected_indices.extend(class_indices[order[:class_cap]].tolist())
    return np.array(sorted(selected_indices), dtype=np.int64)


def cap_selected_indices_balanced(
    candidate_mask: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    margins: np.ndarray,
    classifier_confidences: np.ndarray,
    total_cap: int | None,
    class_cap: int,
) -> np.ndarray:
    if total_cap is None:
        return cap_selected_indices(candidate_mask, labels, distances, margins, classifier_confidences, class_cap)

    candidate_indices = np.where(candidate_mask)[0]
    if total_cap <= 0 or len(candidate_indices) == 0:
        return np.array([], dtype=np.int64)

    ordered_by_class: dict[int, list[int]] = {}
    for class_id in sorted(np.unique(labels[candidate_mask]).tolist()):
        class_indices = np.where(candidate_mask & (labels == class_id))[0]
        order = np.lexsort(
            (
                -classifier_confidences[class_indices],
                -margins[class_indices],
                distances[class_indices],
            )
        )
        ordered_by_class[int(class_id)] = class_indices[order[:class_cap]].tolist()

    class_ids = sorted(ordered_by_class)
    selected = []
    selected_set = set()
    per_class_target = int(np.ceil(total_cap / max(1, len(class_ids))))
    for rank in range(per_class_target):
        for class_id in class_ids:
            if rank >= len(ordered_by_class[class_id]):
                continue
            if len(selected) >= total_cap:
                break
            index = ordered_by_class[class_id][rank]
            selected.append(index)
            selected_set.add(index)
        if len(selected) >= total_cap:
            break

    remaining = [index for indices in ordered_by_class.values() for index in indices if index not in selected_set]
    remaining = sorted(
        remaining,
        key=lambda index: (
            distances[index],
            -margins[index],
            -classifier_confidences[index],
        ),
    )
    for index in remaining:
        if len(selected) >= total_cap:
            break
        selected.append(index)

    return np.array(sorted(selected), dtype=np.int64)


def load_pseudo_label_rows(labels_csv: Path | None) -> list[tuple[str, int, float]]:
    if labels_csv is None:
        return []
    rows = []
    with labels_csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            weight = float(row.get(PSEUDO_LABELS.weight_column, 1.0) or 1.0)
            rows.append((row[DATA.image_path_column], int(row[DATA.label_column]), weight))
    return rows


def cap_selected_indices_cumulative(
    candidate_mask: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    margins: np.ndarray,
    classifier_confidences: np.ndarray,
    image_paths: list[str],
    total_cap: int | None,
    class_cap: int,
    previous_rows: list[tuple[str, int, float]],
) -> np.ndarray:
    if not previous_rows:
        return cap_selected_indices_balanced(
            candidate_mask,
            labels,
            distances,
            margins,
            classifier_confidences,
            total_cap,
            class_cap,
        )
    if total_cap is None:
        total_cap = len(previous_rows) + class_cap * MODEL.num_classes

    previous_paths = {path for path, _, _ in previous_rows}
    remaining_total = max(0, total_cap - len(previous_rows))
    if remaining_total == 0:
        return np.array([], dtype=np.int64)

    eligible_mask = candidate_mask & np.array([path not in previous_paths for path in image_paths], dtype=bool)
    if not eligible_mask.any():
        return np.array([], dtype=np.int64)

    previous_counts = {class_id: 0 for class_id in range(MODEL.num_classes)}
    for _, label, _ in previous_rows:
        previous_counts[int(label)] = previous_counts.get(int(label), 0) + 1

    target_per_class = int(np.ceil(total_cap / MODEL.num_classes))
    ordered_by_class: dict[int, list[int]] = {}
    for class_id in range(MODEL.num_classes):
        class_indices = np.where(eligible_mask & (labels == class_id))[0]
        order = np.lexsort(
            (
                -classifier_confidences[class_indices],
                -margins[class_indices],
                distances[class_indices],
            )
        )
        ordered_by_class[class_id] = class_indices[order].tolist()

    selected = []
    selected_set = set()
    per_class_added = {class_id: 0 for class_id in range(MODEL.num_classes)}
    for rank in range(target_per_class):
        for class_id in range(MODEL.num_classes):
            desired_room = min(class_cap, target_per_class) - previous_counts.get(class_id, 0)
            if per_class_added[class_id] >= max(0, desired_room):
                continue
            class_candidates = ordered_by_class[class_id]
            if rank >= len(class_candidates):
                continue
            if len(selected) >= remaining_total:
                break
            index = class_candidates[rank]
            selected.append(index)
            selected_set.add(index)
            per_class_added[class_id] += 1
        if len(selected) >= remaining_total:
            break

    remaining = [index for indices in ordered_by_class.values() for index in indices if index not in selected_set]
    remaining = sorted(
        remaining,
        key=lambda index: (
            distances[index],
            -margins[index],
            -classifier_confidences[index],
        ),
    )
    for index in remaining:
        if len(selected) >= remaining_total:
            break
        class_id = int(labels[index])
        if previous_counts.get(class_id, 0) + per_class_added[class_id] >= class_cap:
            continue
        selected.append(index)
        per_class_added[class_id] += 1

    return np.array(sorted(selected), dtype=np.int64)


def quality_stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "mean": None,
            "q05": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q95": None,
        }
    return {
        "mean": float(values.mean()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def pseudo_label_weights(
    prototype_margins: np.ndarray,
    classifier_confidences: np.ndarray,
    min_weight: float = PSEUDO_LABELS.min_sample_weight,
) -> np.ndarray:
    if prototype_margins.size == 0:
        return np.array([], dtype=np.float32)
    margin_q95 = float(np.quantile(prototype_margins, 0.95))
    margin_scale = max(margin_q95, 1e-12)
    margin_quality = np.clip(prototype_margins / margin_scale, 0.0, 1.0)
    quality = np.clip(classifier_confidences, 0.0, 1.0) * margin_quality
    return (min_weight + (1.0 - min_weight) * quality).astype(np.float32)


def write_pseudo_label_csv(output_path: Path, image_paths: list[str], labels: np.ndarray, weights: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[DATA.image_path_column, DATA.label_column, PSEUDO_LABELS.weight_column],
        )
        writer.writeheader()
        for image_path, label, weight in zip(image_paths, labels, weights):
            writer.writerow(
                {
                    DATA.image_path_column: image_path,
                    DATA.label_column: int(label),
                    PSEUDO_LABELS.weight_column: float(weight),
                }
            )


def diagnostic_rows(
    image_paths: list[str],
    result,
    selected_mask: np.ndarray,
    max_distances: float | np.ndarray,
    min_margins: float | np.ndarray,
) -> list[dict]:
    rows = []
    disagreement_mask = result.prototype_labels != result.classifier_labels
    low_confidence_mask = ~selected_mask
    row_max_distances = np.broadcast_to(max_distances, result.prototype_distances.shape)
    row_min_margins = np.broadcast_to(min_margins, result.prototype_margins.shape)
    for index in np.where(disagreement_mask | low_confidence_mask)[0]:
        if disagreement_mask[index]:
            reason = "prototype_classifier_disagreement"
        elif result.prototype_distances[index] > row_max_distances[index]:
            reason = "distance_too_high"
        elif result.prototype_margins[index] < row_min_margins[index]:
            reason = "margin_too_low"
        else:
            reason = "class_cap"
        rows.append(
            {
                "ImagePath": image_paths[index],
                "Label": int(result.labels[index]),
                "PrototypeLabel": int(result.prototype_labels[index]),
                "PrototypeDistance": float(result.prototype_distances[index]),
                "PrototypeMargin": float(result.prototype_margins[index]),
                "ClassifierLabel": int(result.classifier_labels[index]),
                "ClassifierConfidence": float(result.classifier_confidences[index]),
                "DecisionReason": reason,
                "MaxDistance": float(row_max_distances[index]),
                "MinMargin": float(row_min_margins[index]),
            }
        )
    return rows


def generate_pseudo_labels(
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
    sample_submission: Path | None = None,
    batch_size: int = SUBMISSION.batch_size,
    num_workers: int = SUBMISSION.num_workers,
    max_distance: float = PSEUDO_LABELS.max_distance,
    min_margin: float = PSEUDO_LABELS.min_margin,
    class_cap: int = PSEUDO_LABELS.class_cap,
    total_cap: int | None = PSEUDO_LABELS.total_cap,
    prototype_margin_gate: float = ENSEMBLE.prototype_margin_gate,
    classifier_confidence_gate: float = ENSEMBLE.classifier_confidence_gate,
    threshold_mode: str = PSEUDO_LABELS.threshold_mode,
    labeled_distance_quantile: float = PSEUDO_LABELS.labeled_distance_quantile,
    labeled_distance_scale: float = PSEUDO_LABELS.labeled_distance_scale,
    labeled_margin_quantile: float = PSEUDO_LABELS.labeled_margin_quantile,
    labeled_margin_scale: float = PSEUDO_LABELS.labeled_margin_scale,
    previous_pseudo_labels: Path | None = None,
    use_tta: bool = SUBMISSION.use_tta,
) -> PseudoLabelArtifacts:
    checkpoint_path = resolve_checkpoint(checkpoint, checkpoint_kind="embedding")
    output_dir = output_dir.resolve() if output_dir is not None else timestamped_pseudo_label_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_submission = sample_submission.resolve() if sample_submission is not None else sample_submission_path()

    device = get_device()
    model = Autoencoder.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)

    classes, prototypes, labeled_embeddings, labeled_labels = compute_labeled_reference(
        model,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tta=use_tta,
    )
    relative_thresholds = labeled_relative_thresholds(
        labeled_embeddings,
        labeled_labels,
        classes,
        prototypes,
        distance_quantile=labeled_distance_quantile,
        distance_scale=labeled_distance_scale,
        margin_quantile=labeled_margin_quantile,
        margin_scale=labeled_margin_scale,
    )

    submission_dataset = SubmissionImageDataset(sample_submission)
    submission_loader = build_loader(
        submission_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    embeddings, logits, image_paths = compute_embeddings_and_logits(
        model,
        submission_loader,
        device,
        "Embedding pseudo-label candidates",
        use_tta=use_tta,
    )
    prediction_result = ensemble_predict_labels(
        embeddings,
        logits,
        classes,
        prototypes,
        prototype_margin_gate=prototype_margin_gate,
        classifier_confidence_gate=classifier_confidence_gate,
    )

    if threshold_mode == "labeled_relative":
        max_distances, min_margins = thresholds_for_predictions(prediction_result.labels, relative_thresholds)
    else:
        max_distances = max_distance
        min_margins = min_margin

    candidate_mask = selection_mask(prediction_result, max_distances, min_margins)
    previous_rows = load_pseudo_label_rows(previous_pseudo_labels)
    selected_indices = cap_selected_indices_cumulative(
        candidate_mask,
        prediction_result.labels,
        prediction_result.prototype_distances,
        prediction_result.prototype_margins,
        prediction_result.classifier_confidences,
        image_paths,
        total_cap,
        class_cap,
        previous_rows,
    )
    selected_mask = np.zeros(len(prediction_result.labels), dtype=bool)
    selected_mask[selected_indices] = True

    new_selected_paths = [image_paths[index] for index in selected_indices]
    new_selected_labels = prediction_result.labels[selected_indices]
    new_selected_weights = pseudo_label_weights(
        prediction_result.prototype_margins[selected_indices],
        prediction_result.classifier_confidences[selected_indices],
    )
    selected_paths = [path for path, _, _ in previous_rows] + new_selected_paths
    selected_labels = np.array([label for _, label, _ in previous_rows] + new_selected_labels.tolist(), dtype=np.int64)
    selected_weights = np.array(
        [weight for _, _, weight in previous_rows] + new_selected_weights.tolist(),
        dtype=np.float32,
    )
    path_to_index = {path: index for index, path in enumerate(image_paths)}
    combined_indices = np.array([path_to_index[path] for path in selected_paths if path in path_to_index], dtype=np.int64)
    selected_mask[combined_indices] = True
    pseudo_labels_path = output_dir / "pseudo_labels.csv"
    diagnostics_path = output_dir / "diagnostics.csv"
    summary_path = output_dir / "summary.json"
    write_pseudo_label_csv(pseudo_labels_path, selected_paths, selected_labels, selected_weights)
    write_diagnostics(
        diagnostics_path,
        diagnostic_rows(
            image_paths,
            prediction_result,
            selected_mask,
            max_distances=max_distances,
            min_margins=min_margins,
        ),
    )

    selected_unique, selected_counts = np.unique(selected_labels, return_counts=True)
    candidate_unique, candidate_counts = np.unique(prediction_result.labels[candidate_mask], return_counts=True)
    reason_labels, reason_counts = np.unique(prediction_result.decision_reasons, return_counts=True)
    write_json(
        summary_path,
        {
            "checkpoint": str(checkpoint_path),
            "rows": int(len(prediction_result.labels)),
            "selected_rows": int(len(selected_labels)),
            "retained_previous_rows": int(len(previous_rows)),
            "newly_added_rows": int(len(selected_indices)),
            "total_selected_rows": int(len(selected_labels)),
            "candidate_rows_before_cap": int(candidate_mask.sum()),
            "selected_label_counts": {
                str(int(label)): int(count) for label, count in zip(selected_unique, selected_counts)
            },
            "candidate_label_counts_before_cap": {
                str(int(label)): int(count) for label, count in zip(candidate_unique, candidate_counts)
            },
            "decision_reason_counts": {
                str(reason): int(count) for reason, count in zip(reason_labels, reason_counts)
            },
            "max_distance": max_distance,
            "min_margin": min_margin,
            "threshold_mode": threshold_mode,
            "labeled_distance_quantile": labeled_distance_quantile,
            "labeled_distance_scale": labeled_distance_scale,
            "labeled_margin_quantile": labeled_margin_quantile,
            "labeled_margin_scale": labeled_margin_scale,
            "relative_thresholds_by_class": {
                str(class_id): values for class_id, values in relative_thresholds.items()
            },
            "class_cap": class_cap,
            "total_cap": total_cap,
            "budget_shortfall": int(max(0, (total_cap or len(selected_labels)) - len(selected_labels))),
            "selected_quality_stats": {
                "prototype_distance": quality_stats(prediction_result.prototype_distances[combined_indices]),
                "prototype_margin": quality_stats(prediction_result.prototype_margins[combined_indices]),
                "classifier_confidence": quality_stats(prediction_result.classifier_confidences[combined_indices]),
                "sample_weight": quality_stats(selected_weights),
            },
            "min_sample_weight": PSEUDO_LABELS.min_sample_weight,
            "prototype_margin_gate": prototype_margin_gate,
            "classifier_confidence_gate": classifier_confidence_gate,
            "use_tta": use_tta,
            "tta_views": len(SUBMISSION.tta_rotation_degrees) if use_tta else 1,
        },
    )

    return PseudoLabelArtifacts(
        pseudo_labels_path=pseudo_labels_path,
        diagnostics_path=diagnostics_path,
        summary_path=summary_path,
        selected_rows=len(selected_labels),
    )


def main() -> None:
    args = parse_args()
    artifacts = generate_pseudo_labels(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        sample_submission=args.sample_submission,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_distance=args.max_distance,
        min_margin=args.min_margin,
        class_cap=args.class_cap,
        total_cap=args.total_cap,
        prototype_margin_gate=args.prototype_margin_gate,
        classifier_confidence_gate=args.classifier_confidence_gate,
        threshold_mode=args.threshold_mode,
        labeled_distance_quantile=args.labeled_distance_quantile,
        labeled_distance_scale=args.labeled_distance_scale,
        labeled_margin_quantile=args.labeled_margin_quantile,
        labeled_margin_scale=args.labeled_margin_scale,
        previous_pseudo_labels=args.previous_pseudo_labels,
        use_tta=not args.no_tta,
    )

    checkpoint_path = resolve_checkpoint(args.checkpoint, checkpoint_kind="embedding")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Wrote pseudo labels: {artifacts.pseudo_labels_path}")
    print(f"Wrote diagnostics: {artifacts.diagnostics_path}")
    print(f"Wrote summary: {artifacts.summary_path}")
    print(f"Selected rows: {artifacts.selected_rows}")


if __name__ == "__main__":
    main()
