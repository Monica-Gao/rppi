from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def supervised_view_contrastive(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def compute_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    swap_logits: torch.Tensor,
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    cf_logits: torch.Tensor,
    weights: Dict[str, float],
    temperature: float = 0.1,
    rank_margin: float = 0.5,
    teacher_probs: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    same_species_logits: torch.Tensor | None = None,
    same_species_labels: torch.Tensor | None = None,
    degree_logits: torch.Tensor | None = None,
    degree_labels: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if label_smoothing > 0:
        targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
    else:
        targets = labels
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    sym = F.smooth_l1_loss(logits, swap_logits)
    con = supervised_view_contrastive(view_a, view_b, temperature=temperature)
    pos_mask = labels > 0.5
    if pos_mask.any():
        rank = F.relu(rank_margin - logits[pos_mask] + cf_logits[pos_mask]).mean()
    else:
        rank = logits.new_tensor(0.0)
    total = (
        bce
        + weights.get("symmetry", 0.0) * sym
        + weights.get("contrastive", 0.0) * con
        + weights.get("ranking", 0.0) * rank
    )
    if teacher_probs is not None and weights.get("distill", 0.0) > 0:
        valid_teacher = teacher_probs >= 0
        if valid_teacher.any():
            distill = F.binary_cross_entropy_with_logits(
                logits[valid_teacher],
                teacher_probs[valid_teacher].to(dtype=logits.dtype),
            )
        else:
            distill = logits.new_tensor(0.0)
    else:
        distill = logits.new_tensor(0.0)

    if same_species_logits is not None and same_species_labels is not None:
        same_species = F.binary_cross_entropy_with_logits(same_species_logits, same_species_labels)
    else:
        same_species = logits.new_tensor(0.0)
    if degree_logits is not None and degree_labels is not None:
        degree = F.cross_entropy(degree_logits, degree_labels)
    else:
        degree = logits.new_tensor(0.0)

    total = (
        total
        + weights.get("distill", 0.0) * distill
        + weights.get("same_species_adv", 0.0) * same_species
        + weights.get("degree_adv", 0.0) * degree
    )
    return {
        "total": total,
        "bce": bce,
        "symmetry": sym,
        "contrastive": con,
        "ranking": rank,
        "distill": distill,
        "same_species_adv": same_species,
        "degree_adv": degree,
    }
