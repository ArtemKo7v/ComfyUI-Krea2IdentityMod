"""Full-grid near-matched fit, adapted from ComfyUI-Krea2Edit (Apache-2.0).

See NOTICE for attribution. Only the small supported pixel preprocessing branch is
implemented here; no upstream private functions or model internals are imported.
"""

from datetime import datetime, timezone

import torch
import torch.nn.functional as F

from .constants import (
    FIXED_METADATA,
    IDENTITY_MOD_FORMAT_VERSION,
    KREA2_NEAR_MATCH_CROP_TOLERANCE,
    KREA2_VAE_DOWNSCALE_FACTOR,
)
from .identity_mod import (
    ArtemKo7vKrea2IdentityModData,
    ArtemKo7vKrea2IdentityModError,
    _validate_latent_tensor,
    _validate_target_latent,
)


def _validate_image(image: torch.Tensor) -> None:
    """Require exactly one dense, finite RGB/RGBA-style ComfyUI image."""
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ArtemKo7vKrea2IdentityModError("IMAGE must be a B x H x W x C torch.Tensor.")
    if image.shape[0] != 1:
        raise ArtemKo7vKrea2IdentityModError(
            "Krea2 IdentityMod PoC supports exactly one reference image. "
            f"Received image batch size: {image.shape[0]}."
        )
    if min(image.shape[1:3]) <= 0 or image.shape[3] < 3:
        raise ArtemKo7vKrea2IdentityModError("IMAGE requires positive H/W and at least 3 channels.")
    if not image.is_floating_point() or image.layout != torch.strided:
        raise ArtemKo7vKrea2IdentityModError("IMAGE must be a dense floating-point tensor.")
    if image.device.type == "meta" or not torch.isfinite(image[..., :3]).all().item():
        raise ArtemKo7vKrea2IdentityModError("IMAGE RGB channels must contain finite pixel data.")


def _prepare_reference_image(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Center-crop supported aspect ratios, then reproduce upstream float32 bicubic fit."""
    _validate_image(image)
    source_height, source_width = image.shape[1:3]
    scale_fit = min(height / source_height, width / source_width)
    tolerance = 1.0 - KREA2_NEAR_MATCH_CROP_TOLERANCE
    if not (source_height * scale_fit >= height * tolerance
            and source_width * scale_fit >= width * tolerance):
        raise ArtemKo7vKrea2IdentityModError(
            "Krea2 IdentityMod PoC does not support this source/target aspect-ratio "
            f"combination.\n\nSource: {source_width}x{source_height}\n"
            f"Target: {width}x{height}\n\n"
            "PoC v0.1 only supports near-matched aspect ratios that can be converted "
            "to the full target grid before VAE encoding. Arbitrary aspect-ratio fit "
            "geometry is outside PoC v0.1. Prepare the source to the target aspect "
            "ratio or create a separate IdentityMod for another target geometry."
        )
    scale_fill = max(height / source_height, width / source_width)
    crop_height = min(source_height, int(round(height / scale_fill)))
    crop_width = min(source_width, int(round(width / scale_fill)))
    y0 = (source_height - crop_height) // 2
    x0 = (source_width - crop_width) // 2
    cropped = image.movedim(-1, 1)[..., y0:y0 + crop_height, x0:x0 + crop_width]
    # Match upstream's float32 interpolation, including alpha removal AFTER resizing.
    resized = F.interpolate(cropped.float(), size=(height, width), mode="bicubic", antialias=True)
    return resized.movedim(1, -1)[..., :3].clamp(0.0, 1.0)


@torch.no_grad()
def _create_identity_mod(
    image: torch.Tensor, vae, target_latent: dict, identity_name: str, description: str = ""
) -> ArtemKo7vKrea2IdentityModData:
    """Encode once and store the raw VAE result, independent of any Krea2 MODEL object."""
    if not isinstance(identity_name, str) or not isinstance(description, str):
        raise ArtemKo7vKrea2IdentityModError("Identity name and description must be strings.")
    target = _validate_target_latent(target_latent)
    if not callable(getattr(vae, "encode", None)):
        raise ArtemKo7vKrea2IdentityModError(
            "Connect a Krea2-compatible VAE with an encode method."
        )
    height, width = target.shape[-2:]
    scale = KREA2_VAE_DOWNSCALE_FACTOR
    prepared = _prepare_reference_image(image, height * scale, width * scale)
    reference = vae.encode(prepared)
    _validate_latent_tensor(reference, allow_temporal=True)
    # Some image VAEs return B,C,1,H,W. Remove only the verified singleton frame axis.
    if reference.ndim == 5:
        reference = reference.squeeze(2)
    if (reference.shape[0] != 1 or reference.shape[1] != target.shape[1]
            or reference.shape[-2:] != target.shape[-2:]):
        raise ArtemKo7vKrea2IdentityModError(
            "Encoded reference latent does not match the target Krea2 latent geometry.\n\n"
            f"Reference latent: {tuple(reference.shape)}\nTarget latent: {tuple(target.shape)}\n\n"
            "Make sure the Krea2-compatible VAE and target latent are being used."
        )
    metadata = {
        **FIXED_METADATA,
        "format_version": IDENTITY_MOD_FORMAT_VERSION,
        "identity_name": identity_name,
        "description": description,
        "target_width": str(width * scale), "target_height": str(height * scale),
        "target_latent_width": str(width), "target_latent_height": str(height),
        "latent_channels": str(reference.shape[1]), "latent_dtype": str(reference.dtype),
        "source_width": str(image.shape[2]), "source_height": str(image.shape[1]),
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return ArtemKo7vKrea2IdentityModData(reference, metadata)
