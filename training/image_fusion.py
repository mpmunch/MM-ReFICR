import torch
import torch.nn.functional as F
from typing import Optional


def validate_image_fusion_mode(image_fusion_mode: str) -> str:
    mode = image_fusion_mode.lower()
<<<<<<< HEAD
    if mode not in {"linear", "concat", "dynamic"}:
        raise ValueError(
            f"Invalid image_fusion_mode: {mode}. Expected one of: linear, concat, dynamic"
=======
    if mode not in {"linear", "concat"}:
        raise ValueError(
            f"Invalid image_fusion_mode: {mode}. Expected one of: linear, concat"
>>>>>>> main
        )
    return mode


<<<<<<< HEAD
def project_image_reps(
    text_reps: torch.Tensor,
    image_emb: torch.Tensor,
    image_projection: torch.nn.Module,
    normalized: bool,
) -> torch.Tensor:
    """Project image embeddings into the same space as text reps.

    This helper keeps dtype/device handling and optional normalization consistent
    across training, inference, and analysis utilities.
    """
    image_reps = image_projection(
        image_emb.to(device=text_reps.device, dtype=text_reps.dtype)
    )
    if normalized:
        image_reps = F.normalize(image_reps, dim=-1).to(text_reps.dtype)
    return image_reps


def fuse_text_with_images_dynamic_lerp(
    text_reps: torch.Tensor,
    image_emb: torch.Tensor,
    image_mask: torch.Tensor,
    image_projection: torch.nn.Module,
    image_gate: torch.nn.Module,
    normalized: bool,
) -> torch.Tensor:
    """Dynamic LERP fusion with strict text fallback.

    Implements:
        alpha = sigmoid(W [E_text; E_image_hat] + b)
        E_fused = (1-alpha) E_text + alpha E_image_hat   if m=1
                  E_text                               if m=0

    Notes:
    - alpha is a per-sample scalar in [0, 1] with shape [N, 1]
    - E_image_hat is the projected (and optionally normalized) image representation
    """
    if text_reps.size(0) != image_emb.size(0):
        raise ValueError(
            f"Passage/image batch mismatch: {text_reps.size(0)} vs {image_emb.size(0)}"
        )

    image_reps = project_image_reps(
        text_reps=text_reps,
        image_emb=image_emb,
        image_projection=image_projection,
        normalized=normalized,
    )

    gate_inp = torch.cat([text_reps, image_reps], dim=-1)
    # Keep alpha dtype aligned with text reps (important for fp16/bf16 runs).
    dynamic_alpha = torch.sigmoid(image_gate(gate_inp)).to(dtype=text_reps.dtype)

    mask = image_mask.to(device=text_reps.device, dtype=text_reps.dtype).unsqueeze(-1)
    fused_reps = text_reps + mask * dynamic_alpha * (image_reps - text_reps)

    if normalized:
        fused_reps = F.normalize(fused_reps, dim=-1).to(text_reps.dtype)

    return fused_reps.contiguous()


=======
>>>>>>> main
def fuse_text_with_images_linear(
    text_reps: torch.Tensor,
    image_emb: torch.Tensor,
    image_mask: torch.Tensor,
    image_projection: torch.nn.Module,
    image_fusion_weight: float,
    normalized: bool,
) -> torch.Tensor:
    """
    Linearly interpolate text and image representations with strict text fallback.

    Args:
        text_reps: Passage/text reps after pooling, shape [N, D].
        image_emb: Raw image vectors, shape [N, 512].
        image_mask: Boolean mask for rows with matched image vectors, shape [N].
        image_projection: Learned layer that maps image vectors into text embedding space.
        image_fusion_weight: Alpha for interpolation.
        normalized: Whether to L2-normalize vectors before/after fusion.
    """
    if text_reps.size(0) != image_emb.size(0):
        raise ValueError(
            f"Passage/image batch mismatch: {text_reps.size(0)} vs {image_emb.size(0)}"
        )

    # Align image features to the text embedding space and match dtype/device.
<<<<<<< HEAD
    image_reps = project_image_reps(
        text_reps=text_reps,
        image_emb=image_emb,
        image_projection=image_projection,
        normalized=normalized,
    )
=======
    image_reps = image_projection(image_emb.to(device=text_reps.device, dtype=text_reps.dtype))
    if normalized:
        # Keep projected image vectors on the same scale as text vectors.
        image_reps = F.normalize(image_reps, dim=-1).to(text_reps.dtype)
>>>>>>> main

    # Broadcast mask over embedding dimension.
    # mask=1 -> apply interpolation, mask=0 -> keep text representation exactly.
    mask = image_mask.to(device=text_reps.device, dtype=text_reps.dtype).unsqueeze(-1)
    # Equivalent to: (1 - alpha * mask) * text + (alpha * mask) * image.
    fused_reps = text_reps + mask * image_fusion_weight * (image_reps - text_reps)

    if normalized:
        # Re-normalize after fusion to keep retrieval similarity behavior stable.
        fused_reps = F.normalize(fused_reps, dim=-1).to(text_reps.dtype)

    # Keep output memory layout consistent with surrounding training/inference code.
    return fused_reps.contiguous()


def fuse_text_with_images_concat(
    text_reps: torch.Tensor,
    image_emb: torch.Tensor,
    image_mask: torch.Tensor,
    image_projection: torch.nn.Module,
    normalized: bool,
    image_concat_projection: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    Concatenation strategy with strict text fallback.

    Flow:
    - project image vectors into embedding space
    - concatenate [text_reps, image_reps]
    - pass through image_concat_projection to return to embedding dimension D
    - keep strict text-only fallback for mask=0 rows
    """
    if text_reps.size(0) != image_emb.size(0):
        raise ValueError(
            f"Passage/image batch mismatch: {text_reps.size(0)} vs {image_emb.size(0)}"
        )
    if image_concat_projection is None:
        raise ValueError("image_concat_projection is required for image_fusion_mode='concat'")

<<<<<<< HEAD
    image_reps = project_image_reps(
        text_reps=text_reps,
        image_emb=image_emb,
        image_projection=image_projection,
        normalized=normalized,
    )
=======
    image_reps = image_projection(image_emb.to(device=text_reps.device, dtype=text_reps.dtype))
    if normalized:
        image_reps = F.normalize(image_reps, dim=-1).to(text_reps.dtype)
>>>>>>> main

    concat_reps = torch.cat([text_reps, image_reps], dim=-1)
    fused_candidate = image_concat_projection(concat_reps).to(text_reps.dtype)
    if normalized:
        fused_candidate = F.normalize(fused_candidate, dim=-1).to(text_reps.dtype)

    # Strict fallback:
    # mask=1 -> use concat-fused representation
    # mask=0 -> keep the original text representation exactly
    mask = image_mask.to(device=text_reps.device, dtype=text_reps.dtype).unsqueeze(-1)
    fused_reps = text_reps + mask * (fused_candidate - text_reps)

    return fused_reps.contiguous()


def apply_image_fusion(
    image_fusion_mode: str,
    text_reps: torch.Tensor,
    image_emb: torch.Tensor,
    image_mask: torch.Tensor,
    image_projection: torch.nn.Module,
    image_fusion_weight: float,
    normalized: bool,
    image_concat_projection: Optional[torch.nn.Module] = None,
<<<<<<< HEAD
    image_gate: Optional[torch.nn.Module] = None,
=======
>>>>>>> main
) -> torch.Tensor:
    """
    Dispatches to the configured image fusion strategy.
    """
    mode = validate_image_fusion_mode(image_fusion_mode)
    if mode == "linear":
        return fuse_text_with_images_linear(
            text_reps=text_reps,
            image_emb=image_emb,
            image_mask=image_mask,
            image_projection=image_projection,
            image_fusion_weight=image_fusion_weight,
            normalized=normalized,
        )

<<<<<<< HEAD
    if mode == "dynamic":
        if image_gate is None:
            raise ValueError("image_gate is required for image_fusion_mode='dynamic'")
        return fuse_text_with_images_dynamic_lerp(
            text_reps=text_reps,
            image_emb=image_emb,
            image_mask=image_mask,
            image_projection=image_projection,
            image_gate=image_gate,
            normalized=normalized,
        )

=======
>>>>>>> main
    return fuse_text_with_images_concat(
        text_reps=text_reps,
        image_emb=image_emb,
        image_mask=image_mask,
        image_projection=image_projection,
        normalized=normalized,
        image_concat_projection=image_concat_projection,
    )
