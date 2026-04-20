import torch
import numpy as np
import torch.nn.functional as F

def supcon_loss_node_equal(
    embeddings: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Supervised contrastive loss (SupCon),

    embeddings: (N, D)
    group_ids:  (N,) shower id per hit
    temperature: tau

    Returns scalar loss (0 if no anchor has positives).
    """
    device = embeddings.device
    N = embeddings.size(0)
    if N <= 1:
        return embeddings.new_tensor(0.0)

    # Normalize + cosine similarity
    z = F.normalize(embeddings.float(), p=2, dim=1)
    logits = (z @ z.t()) / float(temperature)

    # Masks
    not_self = ~torch.eye(N, dtype=torch.bool, device=device)
    logits = logits.masked_fill(~not_self, float("-inf"))

    labels = group_ids
    noise_mask = labels != 0  # True for real (non-noise) hits
    pos_mask = (labels[:, None] == labels[None, :]) & not_self & noise_mask[:, None] & noise_mask[None, :]
    valid_anchor = pos_mask.any(dim=1) & noise_mask

    if not bool(valid_anchor.any()):
        return embeddings.new_tensor(0.0)

    # Denominator
    log_denom = torch.logsumexp(logits, dim=1)

    # Log-probabilities
    log_prob = logits - log_denom[:, None]

    # Mean over positives per anchor
    pos_counts = pos_mask.sum(dim=1).clamp_min(1)
    mean_log_prob_pos = (
        log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_counts
    )

    loss_i = -mean_log_prob_pos

    # NODE-EQUAL reduction
    loss = loss_i[valid_anchor].mean()
    return loss