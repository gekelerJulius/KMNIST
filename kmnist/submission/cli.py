import argparse
from pathlib import Path

import numpy as np

from CONFIG import ENSEMBLE
from CONFIG import STAGED_TRAINING, SUBMISSION
from kmnist.analysis.method_comparison import (
    AUTO_METHODS,
    METHOD_CLASSIFIER,
    METHOD_ENSEMBLE,
    METHOD_KMEANS_TRAIN,
    METHOD_KMEANS_TRANSDUCTIVE,
    METHOD_PROTOTYPE,
    class_prototypes_from_embeddings,
    predict_with_kmeans_train,
    predict_with_kmeans_transductive,
    select_best_method,
    summarize_predictions,
    validation_predictions,
    validation_split_arrays,
)
from kmnist.data import LabeledImageFolderDataset
from kmnist.data.loaders import build_loader
from kmnist.data.transforms import build_test_transform
from kmnist.models import Autoencoder
from kmnist.submission.dataset import SubmissionImageDataset
from kmnist.submission.embeddings import compute_embeddings_and_logits
from kmnist.submission.ensemble import write_ensemble_submission
from kmnist.submission.prediction import classifier_predictions, prototype_predictions
from kmnist.submission.prediction import diagnostic_mask, ensemble_predict_labels
from kmnist.submission.writer import write_diagnostics, write_json, write_submission
from kmnist.utils.checkpoints import ensemble_checkpoints_from_staged_run, resolve_checkpoint
from kmnist.utils.device import get_device
from kmnist.utils.paths import labeled_dir, labels_csv_path, sample_submission_path, timestamped_submission_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a KMNIST submission by assigning unlabeled embeddings to nearest labeled class prototypes."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a Lightning checkpoint, checkpoint directory, or staged training run. Defaults to the best available checkpoint.",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=sample_submission_path(),
        help="Sample submission CSV whose ImagePath order and formatting should be preserved.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Submission CSV to write. Defaults to a timestamped outputs/submissions directory.",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=None,
        help="Diagnostics CSV to write. Defaults beside the submission CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Summary JSON to write. Defaults beside the submission CSV.",
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
        "--method",
        choices=("auto",) + AUTO_METHODS,
        default="auto",
        help="Prediction method. auto evaluates the validation split and uses the best method.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=STAGED_TRAINING.ensemble_size,
        help="Number of top staged-run checkpoints to ensemble when --checkpoint points to a postprocessed staged run.",
    )
    parser.add_argument(
        "--staged-submission-mode",
        choices=("best-single", "ensemble"),
        default=STAGED_TRAINING.staged_submission_mode,
        help="Submission mode when --checkpoint points to a postprocessed staged run.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable deterministic classifier test-time augmentation.",
    )
    return parser.parse_args()


def labeled_reference_arrays(
    model,
    device,
    batch_size: int,
    num_workers: int,
    use_tta: bool = SUBMISSION.use_tta,
):
    labeled_dataset = LabeledImageFolderDataset(
        labeled_dir(),
        labels_csv_path(),
        transform=build_test_transform(),
    )
    labeled_loader = build_loader(labeled_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    labeled_embeddings, labeled_logits, labeled_labels = compute_embeddings_and_logits(
        model,
        labeled_loader,
        device,
        "Embedding labeled reference",
        use_tta=use_tta,
    )
    return labeled_dataset, labeled_embeddings, labeled_logits, np.asarray(labeled_labels, dtype=np.int64)


def predict_submission_labels(
    method: str,
    labeled_embeddings: np.ndarray,
    labeled_labels: np.ndarray,
    unlabeled_embeddings: np.ndarray,
    logits: np.ndarray,
    prototype_margin_gate: float,
    classifier_confidence_gate: float,
):
    classes, prototypes = class_prototypes_from_embeddings(labeled_embeddings, labeled_labels)
    if method == METHOD_PROTOTYPE:
        return prototype_predictions(unlabeled_embeddings, classes, prototypes)[0], None
    if method == METHOD_CLASSIFIER:
        return classifier_predictions(logits, classes)[0], None
    if method == METHOD_KMEANS_TRAIN:
        return predict_with_kmeans_train(labeled_embeddings, labeled_labels, unlabeled_embeddings), None
    if method == METHOD_KMEANS_TRANSDUCTIVE:
        return predict_with_kmeans_transductive(labeled_embeddings, labeled_labels, unlabeled_embeddings), None
    if method == METHOD_ENSEMBLE:
        result = ensemble_predict_labels(
            unlabeled_embeddings,
            logits,
            classes,
            prototypes,
            prototype_margin_gate=prototype_margin_gate,
            classifier_confidence_gate=classifier_confidence_gate,
        )
        return result.labels, result
    raise ValueError(f"Unknown submission method: {method}")


def write_checkpoint_submission(
    checkpoint_path: Path,
    output_path: Path,
    diagnostics_path: Path,
    summary_path: Path,
    sample_submission: Path | None = None,
    batch_size: int = SUBMISSION.batch_size,
    num_workers: int = SUBMISSION.num_workers,
    method: str = "auto",
    prototype_margin_gate: float = ENSEMBLE.prototype_margin_gate,
    classifier_confidence_gate: float = ENSEMBLE.classifier_confidence_gate,
    checkpoint_metadata: dict | None = None,
    use_tta: bool = SUBMISSION.use_tta,
) -> dict:
    sample_path = sample_submission.resolve() if sample_submission is not None else sample_submission_path()
    device = get_device()
    model = Autoencoder.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)

    labeled_dataset, labeled_embeddings, labeled_logits, labeled_labels = labeled_reference_arrays(
        model,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        use_tta=use_tta,
    )

    submission_dataset = SubmissionImageDataset(sample_path)
    submission_loader = build_loader(
        submission_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    unlabeled_embeddings, logits, image_paths = compute_embeddings_and_logits(
        model,
        submission_loader,
        device,
        "Embedding submission images",
        use_tta=use_tta,
    )

    split = validation_split_arrays(labeled_dataset, labeled_embeddings, labeled_logits, labeled_labels)
    method_predictions = validation_predictions(split)
    method_summaries = summarize_predictions(method_predictions, split.validation_labels)
    selected_method = select_best_method(method_summaries) if method == "auto" else method

    predicted_labels, prediction_result = predict_submission_labels(
        selected_method,
        labeled_embeddings,
        labeled_labels,
        unlabeled_embeddings,
        logits,
        prototype_margin_gate=prototype_margin_gate,
        classifier_confidence_gate=classifier_confidence_gate,
    )
    write_submission(output_path, image_paths, predicted_labels)

    diagnostics = []
    if prediction_result is not None:
        for index in np.where(diagnostic_mask(prediction_result))[0]:
            diagnostics.append(
                {
                    "ImagePath": image_paths[index],
                    "Label": int(predicted_labels[index]),
                    "PrototypeLabel": int(prediction_result.prototype_labels[index]),
                    "PrototypeDistance": float(prediction_result.prototype_distances[index]),
                    "PrototypeMargin": float(prediction_result.prototype_margins[index]),
                    "ClassifierLabel": int(prediction_result.classifier_labels[index]),
                    "ClassifierConfidence": float(prediction_result.classifier_confidences[index]),
                    "DecisionReason": str(prediction_result.decision_reasons[index]),
                }
            )
    write_diagnostics(diagnostics_path, diagnostics)

    unique_labels, label_counts = np.unique(predicted_labels, return_counts=True)
    label_count_summary = {str(int(label)): int(count) for label, count in zip(unique_labels, label_counts)}
    if prediction_result is None:
        reason_summary = {}
    else:
        reason_labels, reason_counts = np.unique(prediction_result.decision_reasons, return_counts=True)
        reason_summary = {str(reason): int(count) for reason, count in zip(reason_labels, reason_counts)}
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata or {},
        "requested_method": method,
        "selected_method": selected_method,
        "validation_method_comparison": method_summaries,
        "rows": int(len(predicted_labels)),
        "label_counts": label_count_summary,
        "decision_reason_counts": reason_summary,
        "diagnostic_rows": len(diagnostics),
        "prototype_margin_gate": prototype_margin_gate,
        "classifier_confidence_gate": classifier_confidence_gate,
        "use_tta": use_tta,
        "tta_views": len(SUBMISSION.tta_rotation_degrees) if use_tta else 1,
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    sample_path = args.sample_submission.resolve()
    if args.output is None:
        output_dir = timestamped_submission_dir()
        output_path = output_dir / SUBMISSION.output_filename
    else:
        output_path = args.output.resolve()
        output_dir = output_path.parent
    diagnostics_path = (
        args.diagnostics_output.resolve()
        if args.diagnostics_output is not None
        else output_dir / "diagnostics.csv"
    )
    summary_path = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else output_dir / "summary.json"
    )

    if args.checkpoint is not None and args.checkpoint.is_dir() and args.staged_submission_mode == "ensemble":
        try:
            checkpoint_paths, metadata = ensemble_checkpoints_from_staged_run(
                args.checkpoint.resolve(),
                ensemble_size=args.ensemble_size,
            )
        except FileNotFoundError:
            checkpoint_paths = []
            metadata = []
        if checkpoint_paths:
            write_ensemble_submission(
                checkpoint_paths,
                output_path=output_path,
                summary_path=summary_path,
                sample_submission=sample_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                checkpoint_metadata=metadata,
                use_tta=not args.no_tta,
            )
            write_diagnostics(diagnostics_path, [])
            print(f"Checkpoints: {len(checkpoint_paths)}")
            print(f"Wrote submission: {output_path}")
            print(f"Wrote diagnostics: {diagnostics_path}")
            print(f"Wrote summary: {summary_path}")
            print("Selected method: classifier_probability_average")
            return

    checkpoint_path = resolve_checkpoint(args.checkpoint, checkpoint_kind="embedding")
    summary = write_checkpoint_submission(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        summary_path=summary_path,
        sample_submission=sample_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        method=args.method,
        prototype_margin_gate=args.prototype_margin_gate,
        classifier_confidence_gate=args.classifier_confidence_gate,
        use_tta=not args.no_tta,
    )
    count_summary = ", ".join(f"{label}:{count}" for label, count in summary["label_counts"].items())
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Wrote submission: {output_path}")
    print(f"Wrote diagnostics: {diagnostics_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Selected method: {summary['selected_method']}")
    print(f"Rows: {summary['rows']}")
    print(f"Predicted label counts: {count_summary}")


if __name__ == "__main__":
    main()
