from __future__ import annotations

import csv
import json
import os
from typing import Iterable, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
from training.image_fusion import project_image_reps

def compute_dynamic_alpha_and_agreement(
    *,
    text_reps: torch.Tensor,
    image_emb: torch.Tensor,
    image_mask: torch.Tensor,
    image_projection: torch.nn.Module,
    image_gate: torch.nn.Module,
    normalized: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-item alpha and text-image agreement for dynamic fusion.

    Agreement is cosine similarity between text embeddings and projected image embeddings.

    Returns:
        alpha: [N] float32 in [0,1]
        agreement: [N] float32 in [-1,1]
        mask: [N] bool
    """
    if text_reps.ndim != 2:
        raise ValueError(f"Expected text_reps shape [N, D], got {tuple(text_reps.shape)}")
    if image_emb.ndim != 2:
        raise ValueError(f"Expected image_emb shape [N, 512], got {tuple(image_emb.shape)}")
    if text_reps.size(0) != image_emb.size(0):
        raise ValueError(f"Batch mismatch: {text_reps.size(0)} vs {image_emb.size(0)}")

    mask = image_mask.to(device=text_reps.device, dtype=torch.bool).view(-1)

    image_reps = project_image_reps(
        text_reps=text_reps,
        image_emb=image_emb,
        image_projection=image_projection,
        normalized=normalized,
    )

    gate_inp = torch.cat([text_reps, image_reps], dim=-1)
    alpha = torch.sigmoid(image_gate(gate_inp)).squeeze(-1)

    # Cast to float32 for stable logging/plotting.
    alpha_f32 = alpha.detach().to(torch.float32)
    # Use normalized vectors so this is a true cosine similarity, even if callers
    # pass un-normalized embeddings.
    text_unit = F.normalize(text_reps, dim=-1)
    image_unit = F.normalize(image_reps, dim=-1)
    # check how much the image/text agree (cosine similarity)
    agreement_f32 = F.cosine_similarity(text_unit, image_unit, dim=-1).detach().to(torch.float32)

    return alpha_f32, agreement_f32, mask


def log_alpha_records(
    *,
    out_path: str,
    item_ids: Sequence[str],
    alpha: torch.Tensor,
    agreement: torch.Tensor,
    mask: torch.Tensor,
    extra_cols: Optional[dict] = None,
) -> int:
    """Log per-item alpha + agreement records to .jsonl or .csv.

    Writes only rows where mask==True.

    Returns:
        num_rows_written
    """
    if len(item_ids) != int(alpha.numel()):
        raise ValueError(f"item_ids/alpha mismatch: {len(item_ids)} vs {int(alpha.numel())}")

    extra_cols = extra_cols or {}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    indices = torch.nonzero(mask, as_tuple=False).view(-1).tolist()

    if out_path.lower().endswith(".jsonl"):
        with open(out_path, "w", encoding="utf-8") as f:
            for i in indices:
                rec = {
                    "item_id": item_ids[i],
                    "alpha": float(alpha[i].item()),
                    "agreement": float(agreement[i].item()),
                    **extra_cols,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(indices)

    if out_path.lower().endswith(".csv"):
        fieldnames = ["item_id", "alpha", "agreement", *list(extra_cols.keys())]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in indices:
                rec = {
                    "item_id": item_ids[i],
                    "alpha": float(alpha[i].item()),
                    "agreement": float(agreement[i].item()),
                    **extra_cols,
                }
                writer.writerow(rec)
        return len(indices)

    raise ValueError(
        f"Unsupported alpha log format for: {out_path}. Use a .jsonl or .csv extension."
    )
