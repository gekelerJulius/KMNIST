import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from CONFIG import ANALYSIS, ENSEMBLE, MODEL, TRAINING
from kmnist.data.loaders import stratified_labeled_split
from kmnist.submission.embeddings import normalize_rows
from kmnist.submission.prediction import classifier_predictions, ensemble_predict_labels, prototype_predictions


METHOD_PROTOTYPE = "prototype"
METHOD_CLASSIFIER = "classifier"
METHOD_KMEANS_TRAIN = "kmeans_train"
METHOD_KMEANS_TRANSDUCTIVE = "kmeans_transductive"
METHOD_ENSEMBLE = "ensemble"
AUTO_METHODS = (
    METHOD_PROTOTYPE,
    METHOD_CLASSIFIER,
    METHOD_KMEANS_TRAIN,
    METHOD_KMEANS_TRANSDUCTIVE,
    METHOD_ENSEMBLE,
)


@dataclass(frozen=True)
class ValidationSplitArrays:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_embeddings: np.ndarray
    train_logits: np.ndarray
    train_labels: np.ndarray
    validation_embeddings: np.ndarray
    validation_logits: np.ndarray
    validation_labels: np.ndarray


def validation_split_arrays(
    dataset,
    embeddings: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    seed: int = TRAINING.validation_seed,
) -> ValidationSplitArrays:
    train_split, validation_split = stratified_labeled_split(dataset, seed=seed)
    train_indices = np.asarray(train_split.indices, dtype=np.int64)
    validation_indices = np.asarray(validation_split.indices, dtype=np.int64)
    return ValidationSplitArrays(
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_embeddings=embeddings[train_indices],
        train_logits=logits[train_indices],
        train_labels=labels[train_indices].astype(np.int64),
        validation_embeddings=embeddings[validation_indices],
        validation_logits=logits[validation_indices],
        validation_labels=labels[validation_indices].astype(np.int64),
    )


def class_prototypes_from_embeddings(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized_embeddings = normalize_rows(embeddings)
    classes = np.array(sorted(np.unique(labels)), dtype=np.int64)
    prototypes = []
    for class_id in classes:
        prototype = normalized_embeddings[labels == class_id].mean(axis=0)
        prototype = prototype / np.clip(np.linalg.norm(prototype), a_min=1e-12, a_max=None)
        prototypes.append(prototype)
    return classes, np.stack(prototypes, axis=0)


def map_clusters_to_labels(
    cluster_ids: np.ndarray,
    labels: np.ndarray,
    cluster_centers: np.ndarray,
    classes: np.ndarray,
    prototypes: np.ndarray,
) -> dict[int, int]:
    mapping = {}
    for cluster_id in range(len(cluster_centers)):
        cluster_labels = labels[cluster_ids == cluster_id]
        if cluster_labels.size:
            counts = np.bincount(cluster_labels, minlength=MODEL.num_classes)
            mapping[cluster_id] = int(counts.argmax())
            continue

        normalized_center = cluster_centers[cluster_id] / np.clip(
            np.linalg.norm(cluster_centers[cluster_id]),
            a_min=1e-12,
            a_max=None,
        )
        mapping[cluster_id] = int(classes[(prototypes @ normalized_center).argmax()])
    return mapping


def predict_with_kmeans_train(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    target_embeddings: np.ndarray,
    random_state: int = ANALYSIS.random_state,
) -> np.ndarray:
    classes, prototypes = class_prototypes_from_embeddings(train_embeddings, train_labels)
    normalized_train_embeddings = normalize_rows(train_embeddings)
    normalized_target_embeddings = normalize_rows(target_embeddings)
    kmeans = KMeans(n_clusters=len(classes), random_state=random_state, n_init=20)
    train_clusters = kmeans.fit_predict(normalized_train_embeddings)
    cluster_mapping = map_clusters_to_labels(
        train_clusters,
        train_labels,
        kmeans.cluster_centers_,
        classes,
        prototypes,
    )
    target_clusters = kmeans.predict(normalized_target_embeddings)
    return np.array([cluster_mapping[int(cluster_id)] for cluster_id in target_clusters], dtype=np.int64)


def predict_with_kmeans_transductive(
    labeled_embeddings: np.ndarray,
    labeled_labels: np.ndarray,
    target_embeddings: np.ndarray,
    random_state: int = ANALYSIS.random_state,
) -> np.ndarray:
    classes, prototypes = class_prototypes_from_embeddings(labeled_embeddings, labeled_labels)
    normalized_labeled_embeddings = normalize_rows(labeled_embeddings)
    normalized_target_embeddings = normalize_rows(target_embeddings)
    all_embeddings = np.concatenate([normalized_labeled_embeddings, normalized_target_embeddings], axis=0)
    kmeans = KMeans(n_clusters=len(classes), random_state=random_state, n_init=20)
    cluster_ids = kmeans.fit_predict(all_embeddings)
    labeled_clusters = cluster_ids[: len(labeled_embeddings)]
    target_clusters = cluster_ids[len(labeled_embeddings) :]
    cluster_mapping = map_clusters_to_labels(
        labeled_clusters,
        labeled_labels,
        kmeans.cluster_centers_,
        classes,
        prototypes,
    )
    return np.array([cluster_mapping[int(cluster_id)] for cluster_id in target_clusters], dtype=np.int64)


def validation_predictions(split: ValidationSplitArrays) -> dict[str, np.ndarray]:
    classes, prototypes = class_prototypes_from_embeddings(split.train_embeddings, split.train_labels)
    predictions = {}
    predictions[METHOD_PROTOTYPE] = prototype_predictions(
        split.validation_embeddings,
        classes,
        prototypes,
    )[0]
    predictions[METHOD_CLASSIFIER] = classifier_predictions(split.validation_logits, classes)[0]
    predictions[METHOD_KMEANS_TRAIN] = predict_with_kmeans_train(
        split.train_embeddings,
        split.train_labels,
        split.validation_embeddings,
    )
    predictions[METHOD_KMEANS_TRANSDUCTIVE] = predict_with_kmeans_transductive(
        split.train_embeddings,
        split.train_labels,
        split.validation_embeddings,
    )
    predictions[METHOD_ENSEMBLE] = ensemble_predict_labels(
        split.validation_embeddings,
        split.validation_logits,
        classes,
        prototypes,
        prototype_margin_gate=ENSEMBLE.prototype_margin_gate,
        classifier_confidence_gate=ENSEMBLE.classifier_confidence_gate,
    ).labels
    return predictions


def summarize_predictions(predictions: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, dict]:
    classes = list(range(MODEL.num_classes))
    summaries = {}
    for method_name, predicted_labels in predictions.items():
        per_class_accuracy = {}
        for class_id in classes:
            class_mask = labels == class_id
            if not class_mask.any():
                continue
            per_class_accuracy[str(class_id)] = float((predicted_labels[class_mask] == class_id).mean())
        summaries[method_name] = {
            "accuracy": float(accuracy_score(labels, predicted_labels)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted_labels)),
            "per_class_accuracy": per_class_accuracy,
            "confusion_matrix": confusion_matrix(labels, predicted_labels, labels=classes).tolist(),
        }
    return summaries


def select_best_method(summaries: dict[str, dict], methods: tuple[str, ...] = AUTO_METHODS) -> str:
    return max(
        methods,
        key=lambda method_name: (
            summaries[method_name]["accuracy"],
            summaries[method_name]["balanced_accuracy"],
            method_name == METHOD_CLASSIFIER,
            method_name,
        ),
    )


def repeated_validation_seeds(split_count: int, base_seed: int = TRAINING.validation_seed) -> list[int]:
    return [base_seed + split_index for split_index in range(split_count)]


def repeated_validation_summary(
    dataset,
    embeddings: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    split_count: int,
    base_seed: int = TRAINING.validation_seed,
) -> dict:
    split_rows = []
    method_rows: dict[str, list[dict]] = {method: [] for method in AUTO_METHODS}
    for split_index, seed in enumerate(repeated_validation_seeds(split_count, base_seed)):
        split = validation_split_arrays(dataset, embeddings, logits, labels, seed=seed)
        method_predictions = validation_predictions(split)
        summaries = summarize_predictions(method_predictions, split.validation_labels)
        best_method = select_best_method(summaries)
        split_rows.append(
            {
                "split_index": split_index,
                "seed": seed,
                "best_method": best_method,
                "best_accuracy": summaries[best_method]["accuracy"],
                "best_balanced_accuracy": summaries[best_method]["balanced_accuracy"],
                "methods": summaries,
            }
        )
        for method_name, summary in summaries.items():
            method_rows[method_name].append(summary)

    aggregate_methods = {}
    for method_name, rows in method_rows.items():
        accuracies = np.array([row["accuracy"] for row in rows], dtype=np.float64)
        balanced = np.array([row["balanced_accuracy"] for row in rows], dtype=np.float64)
        aggregate_methods[method_name] = {
            "mean_accuracy": float(accuracies.mean()),
            "std_accuracy": float(accuracies.std(ddof=0)),
            "mean_balanced_accuracy": float(balanced.mean()),
            "std_balanced_accuracy": float(balanced.std(ddof=0)),
        }
    best_method = max(
        AUTO_METHODS,
        key=lambda method_name: (
            aggregate_methods[method_name]["mean_accuracy"],
            aggregate_methods[method_name]["mean_balanced_accuracy"],
            method_name == METHOD_CLASSIFIER,
            method_name,
        ),
    )
    best = aggregate_methods[best_method]
    return {
        "split_count": split_count,
        "base_seed": base_seed,
        "best_method": best_method,
        "mean_accuracy": best["mean_accuracy"],
        "std_accuracy": best["std_accuracy"],
        "mean_balanced_accuracy": best["mean_balanced_accuracy"],
        "std_balanced_accuracy": best["std_balanced_accuracy"],
        "methods": aggregate_methods,
        "splits": split_rows,
    }


def write_method_comparison(output_dir: Path, summaries: dict[str, dict], best_method: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "validation_method_comparison.json").open("w") as json_file:
        json.dump(
            {
                "best_method": best_method,
                "methods": summaries,
            },
            json_file,
            indent=2,
            sort_keys=True,
        )

    with (output_dir / "validation_method_comparison.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["method", "accuracy", "balanced_accuracy"],
        )
        writer.writeheader()
        for method_name, summary in sorted(
            summaries.items(),
            key=lambda item: item[1]["accuracy"],
            reverse=True,
        ):
            writer.writerow(
                {
                    "method": method_name,
                    "accuracy": summary["accuracy"],
                    "balanced_accuracy": summary["balanced_accuracy"],
                }
            )
