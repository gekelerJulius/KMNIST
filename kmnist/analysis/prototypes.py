import numpy as np


def class_prototypes(normalized_embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labeled_mask = labels >= 0
    classes = np.array(sorted(np.unique(labels[labeled_mask])), dtype=np.int64)
    prototypes = []
    for class_id in classes:
        prototype = normalized_embeddings[labels == class_id].mean(axis=0)
        prototype = prototype / np.clip(np.linalg.norm(prototype), a_min=1e-12, a_max=None)
        prototypes.append(prototype)
    return classes, np.stack(prototypes, axis=0)


def prototype_assignment_metrics(
    normalized_embeddings: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, float]]]:
    classes, prototypes = class_prototypes(normalized_embeddings, labels)
    distances = 1 - normalized_embeddings @ prototypes.T
    sorted_distances = np.sort(distances, axis=1)
    nearest_indices = distances.argmin(axis=1)
    nearest_labels = classes[nearest_indices]
    nearest_distances = sorted_distances[:, 0]
    nearest_gaps = sorted_distances[:, 1] - sorted_distances[:, 0]

    labeled_mask = labels >= 0
    summary = {}
    for class_id in classes:
        class_distances = distances[labeled_mask & (labels == class_id), np.where(classes == class_id)[0][0]]
        summary[int(class_id)] = {
            "mean": float(class_distances.mean()),
            "p90": float(np.quantile(class_distances, 0.9)),
            "max": float(class_distances.max()),
        }
    return nearest_labels, nearest_distances, nearest_gaps, summary
