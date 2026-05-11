import torch
from torch.nn import functional as F

from CONFIG import LOSSES


def quantile_as_float(tensor: torch.Tensor, q: float) -> torch.Tensor:
    return tensor.float().quantile(q)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = LOSSES.supervised_contrastive_temperature,
) -> torch.Tensor:
    normalized_embeddings = F.normalize(embeddings, dim=1)
    logits = normalized_embeddings @ normalized_embeddings.T
    logits = logits / temperature

    batch_size = labels.size(0)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    valid_anchor_mask = positive_mask.any(dim=1)
    if not valid_anchor_mask.any():
        return embeddings.new_zeros(())

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits = logits.masked_fill(self_mask, -torch.inf)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    positive_log_probabilities = log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1)
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    anchor_losses = -positive_log_probabilities / positive_counts
    return anchor_losses[valid_anchor_mask].mean()
