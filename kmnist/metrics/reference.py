import torch
from torch.nn import functional as F

from kmnist.losses.embedding_losses import quantile_as_float


def reference_labeling_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    normalized_embeddings = F.normalize(embeddings, dim=1)
    similarities = normalized_embeddings @ normalized_embeddings.T
    sample_count = labels.size(0)
    self_mask = torch.eye(sample_count, dtype=torch.bool, device=labels.device)

    leave_one_out_similarities = similarities.masked_fill(self_mask, -torch.inf)
    knn_1_predictions = labels[leave_one_out_similarities.argmax(dim=1)]
    knn_1_acc = knn_1_predictions.eq(labels).float().mean()

    knn_k = min(3, sample_count - 1)
    knn_3_labels = labels[leave_one_out_similarities.topk(knn_k, dim=1).indices]
    knn_3_predictions = torch.mode(knn_3_labels, dim=1).values
    knn_3_acc = knn_3_predictions.eq(labels).float().mean()

    distances = 1 - similarities
    same_label_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    diff_label_mask = labels.unsqueeze(0).ne(labels.unsqueeze(1))
    same_distances = distances[same_label_mask]
    same_distance_mean = same_distances.mean() if same_distances.numel() > 0 else embeddings.new_zeros(())
    same_distance_p90 = quantile_as_float(same_distances, 0.9) if same_distances.numel() > 0 else embeddings.new_zeros(())
    same_distance_max = same_distances.max() if same_distances.numel() > 0 else embeddings.new_zeros(())
    diff_distance_mean = distances[diff_label_mask].mean() if diff_label_mask.any() else embeddings.new_zeros(())
    distance_margin_mean = diff_distance_mean - same_distance_mean
    nearest_diff_distance = distances.masked_fill(~diff_label_mask, torch.inf).min(dim=1).values
    finite_nearest_diff = nearest_diff_distance[torch.isfinite(nearest_diff_distance)]
    nearest_diff_distance_mean = (
        finite_nearest_diff.mean() if finite_nearest_diff.numel() > 0 else embeddings.new_zeros(())
    )

    classes = torch.unique(labels).sort().values
    prototypes = []
    for class_label in classes:
        class_embeddings = normalized_embeddings[labels.eq(class_label)]
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(F.normalize(prototype, dim=0))
    prototypes = torch.stack(prototypes, dim=0)
    prototype_distances = 1 - prototypes @ prototypes.T
    prototype_self_mask = torch.eye(classes.numel(), dtype=torch.bool, device=labels.device)
    off_diagonal_prototype_distances = prototype_distances[~prototype_self_mask]
    prototype_distance_min = off_diagonal_prototype_distances.min()
    prototype_distance_mean = off_diagonal_prototype_distances.mean()
    target_indices = torch.empty(labels.size(0), dtype=torch.long, device=labels.device)
    for class_index, class_label in enumerate(classes):
        target_indices[labels.eq(class_label)] = class_index
    own_prototype_distances = 1 - (normalized_embeddings * prototypes[target_indices]).sum(dim=1)
    center_distance_mean = own_prototype_distances.mean()
    center_distance_p90 = quantile_as_float(own_prototype_distances, 0.9)
    center_distance_max = own_prototype_distances.max()
    cluster_score = prototype_distance_min - same_distance_p90

    prototype_acc = leave_one_out_prototype_accuracy(normalized_embeddings, labels)
    return {
        "labeled_knn_1_acc": knn_1_acc,
        "labeled_knn_3_acc": knn_3_acc,
        "labeled_prototype_acc": prototype_acc,
        "labeled_same_distance_mean": same_distance_mean,
        "labeled_same_distance_p90": same_distance_p90,
        "labeled_same_distance_max": same_distance_max,
        "labeled_diff_distance_mean": diff_distance_mean,
        "labeled_distance_margin_mean": distance_margin_mean,
        "labeled_nearest_diff_distance_mean": nearest_diff_distance_mean,
        "labeled_prototype_distance_min": prototype_distance_min,
        "labeled_prototype_distance_mean": prototype_distance_mean,
        "labeled_center_distance_mean": center_distance_mean,
        "labeled_center_distance_p90": center_distance_p90,
        "labeled_center_distance_max": center_distance_max,
        "labeled_cluster_score": cluster_score,
    }


def leave_one_out_prototype_accuracy(
    normalized_embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    classes = torch.unique(labels).sort().values
    predictions = []

    for sample_index, label in enumerate(labels):
        class_prototypes = []
        for class_label in classes:
            class_mask = labels.eq(class_label)
            if class_label == label and class_mask.sum() > 1:
                class_mask = class_mask.clone()
                class_mask[sample_index] = False
            prototype = normalized_embeddings[class_mask].mean(dim=0)
            class_prototypes.append(F.normalize(prototype, dim=0))

        prototypes = torch.stack(class_prototypes, dim=0)
        prototype_similarities = prototypes @ normalized_embeddings[sample_index]
        predictions.append(classes[prototype_similarities.argmax()])

    predictions = torch.stack(predictions)
    return predictions.eq(labels).float().mean()
