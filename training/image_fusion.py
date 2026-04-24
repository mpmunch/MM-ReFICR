import torch
import torch.nn.functional as F
from typing import Optional


def validate_image_fusion_mode(image_fusion_mode: str) -> str:
    mode = image_fusion_mode.lower()
    if mode not in {"linear", "concat"}:
        raise ValueError(
            f"Invalid image_fusion_mode: {mode}. Expected one of: linear, concat"
        )
    return mode


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
    image_reps = image_projection(image_emb.to(device=text_reps.device, dtype=text_reps.dtype))
    if normalized:
        # Keep projected image vectors on the same scale as text vectors.
        image_reps = F.normalize(image_reps, dim=-1).to(text_reps.dtype)

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
    Concatenation strategy placeholder.

    Intended future flow:
    - project image vectors into embedding space
    - concatenate [text_reps, image_reps]
    - pass through image_concat_projection to return to embedding dimension D
    - keep strict text-only fallback for mask=0 rows
    """
    raise NotImplementedError(
        "image_fusion_mode='concat' is not implemented yet. Use image_fusion_mode='linear'."
    )


def apply_image_fusion(
    image_fusion_mode: str,
    text_reps: torch.Tensor,
    image_emb: torch.Tensor,
    image_mask: torch.Tensor,
    image_projection: torch.nn.Module,
    image_fusion_weight: float,
    normalized: bool,
    image_concat_projection: Optional[torch.nn.Module] = None,
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

    return fuse_text_with_images_concat(
        text_reps=text_reps,
        image_emb=image_emb,
        image_mask=image_mask,
        image_projection=image_projection,
        normalized=normalized,
        image_concat_projection=image_concat_projection,
    )
