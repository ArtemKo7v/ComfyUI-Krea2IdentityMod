"""Validated, CPU-resident appearance references independent of a loaded model."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import MappingProxyType

import torch

from .constants import (
    DIMENSION_METADATA_KEYS,
    FIXED_METADATA,
    KREA2_LATENT_PATCH_SIZE,
    KREA2_VAE_DOWNSCALE_FACTOR,
    REQUIRED_METADATA_KEYS,
)


class ArtemKo7vKrea2IdentityModError(ValueError):
    """An invalid IdentityMod, input tensor, or storage path was supplied."""


def _validate_format_version(version: str) -> None:
    """Accept stable 0.1.x files and reject incompatible or malformed versions."""
    if not re.fullmatch(r"0\.1\.(0|[1-9][0-9]*)", version):
        raise ArtemKo7vKrea2IdentityModError(
            f"Unsupported IdentityMod format version: {version!r}. Expected 0.1.x."
        )


def _validate_latent_tensor(samples: torch.Tensor, *, allow_temporal: bool = False) -> None:
    """Validate image latents without changing their geometry or numeric representation."""
    if not isinstance(samples, torch.Tensor):
        raise ArtemKo7vKrea2IdentityModError("Latent samples must be a torch.Tensor.")
    ranks = (4, 5) if allow_temporal else (4,)
    if samples.ndim not in ranks:
        raise ArtemKo7vKrea2IdentityModError(
            f"Invalid latent rank {samples.ndim}; expected "
            + ("B x C x H x W or B x C x 1 x H x W." if allow_temporal
               else "B x C x H x W (rank 4).")
        )
    if samples.ndim == 5 and samples.shape[2] != 1:
        raise ArtemKo7vKrea2IdentityModError("Image latents require temporal dimension T=1.")
    if any(dimension <= 0 for dimension in samples.shape):
        raise ArtemKo7vKrea2IdentityModError("All latent dimensions must be positive.")
    if not samples.is_floating_point() or samples.layout != torch.strided:
        raise ArtemKo7vKrea2IdentityModError("Latent must be a dense floating-point tensor.")
    if samples.device.type == "meta":
        raise ArtemKo7vKrea2IdentityModError("Latent must contain data, not a meta tensor.")


def _validate_target_latent(target_latent: dict) -> torch.Tensor:
    """Return validated target samples; the target batch can exceed one."""
    if not isinstance(target_latent, dict) or "samples" not in target_latent:
        raise ArtemKo7vKrea2IdentityModError(
            "target_latent must be a ComfyUI LATENT dictionary containing 'samples'."
        )
    samples = target_latent["samples"]
    _validate_latent_tensor(samples, allow_temporal=True)
    _validate_patch_grid(samples)
    return samples


def _validate_patch_grid(samples: torch.Tensor) -> None:
    """Prevent upstream crop mode from interpolating a reference after target padding."""
    if any(size % KREA2_LATENT_PATCH_SIZE for size in samples.shape[-2:]):
        raise ArtemKo7vKrea2IdentityModError(
            "Krea2 IdentityMod requires even latent H/W (pixel dimensions divisible by 16). "
            "An unaligned grid would trigger latent resizing in Krea2EditModelPatch."
        )


def _validate_metadata(metadata: Mapping[str, str], samples: torch.Tensor) -> None:
    """Check format invariants and cross-check every stored shape and dtype field."""
    if not isinstance(metadata, Mapping):
        raise ArtemKo7vKrea2IdentityModError("IdentityMod metadata must be a string mapping.")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ArtemKo7vKrea2IdentityModError("All metadata keys and values must be strings.")
    missing = REQUIRED_METADATA_KEYS - metadata.keys()
    if missing:
        raise ArtemKo7vKrea2IdentityModError(
            f"Missing required IdentityMod metadata: {', '.join(sorted(missing))}."
        )
    _validate_format_version(metadata["format_version"])
    for key, expected in FIXED_METADATA.items():
        if metadata[key] != expected:
            raise ArtemKo7vKrea2IdentityModError(
                f"Invalid IdentityMod metadata {key}: expected {expected!r}, "
                f"received {metadata[key]!r}."
            )
    for key in DIMENSION_METADATA_KEYS:
        if not re.fullmatch(r"[1-9][0-9]*", metadata[key]):
            raise ArtemKo7vKrea2IdentityModError(f"Metadata {key} must be a positive integer.")
    expected_shape = (
        1, int(metadata["latent_channels"]), int(metadata["target_latent_height"]),
        int(metadata["target_latent_width"]),
    )
    if tuple(samples.shape) != expected_shape:
        raise ArtemKo7vKrea2IdentityModError(
            f"Latent shape {tuple(samples.shape)} does not match metadata {expected_shape}."
        )
    for axis in ("height", "width"):
        if int(metadata[f"target_{axis}"]) != (
            int(metadata[f"target_latent_{axis}"]) * KREA2_VAE_DOWNSCALE_FACTOR
        ):
            raise ArtemKo7vKrea2IdentityModError(
                f"Metadata target_{axis} does not match the latent grid and VAE scale."
            )
    if metadata["latent_dtype"] != str(samples.dtype):
        raise ArtemKo7vKrea2IdentityModError("Latent dtype does not match metadata latent_dtype.")
    try:
        timestamp = datetime.fromisoformat(metadata["created_at_utc"].replace("Z", "+00:00"))
        if timestamp.utcoffset() != timedelta(0):
            raise ValueError("not UTC")
    except ValueError as error:
        raise ArtemKo7vKrea2IdentityModError(
            "Metadata created_at_utc must be an ISO 8601 UTC timestamp."
        ) from error


@dataclass(frozen=True, eq=False)
class ArtemKo7vKrea2IdentityModData:
    """Own a detached raw VAE latent and read-only metadata; treat the tensor as read-only.

    Tensor storage is copied on construction and before handing it to downstream nodes,
    so normal node execution cannot mutate a cached IdentityMod through a shared input.
    """

    reference_latent: torch.Tensor
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validate the portable representation and take ownership of its storage."""
        self.validate()
        object.__setattr__(
            self, "reference_latent", self.reference_latent.detach().cpu().contiguous().clone()
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def validate(self) -> None:
        """Revalidate before serialization or runtime use, including direct tensor edits."""
        _validate_latent_tensor(self.reference_latent)
        _validate_patch_grid(self.reference_latent)
        if self.reference_latent.shape[0] != 1:
            raise ArtemKo7vKrea2IdentityModError("Stored IdentityMod latent requires batch size 1.")
        if not torch.isfinite(self.reference_latent).all().item():
            raise ArtemKo7vKrea2IdentityModError("IdentityMod latent contains non-finite values.")
        _validate_metadata(self.metadata, self.reference_latent)

    @property
    def target_width(self) -> int:
        """Target pixel width."""
        return int(self.metadata["target_width"])

    @property
    def target_height(self) -> int:
        """Target pixel height."""
        return int(self.metadata["target_height"])

    @property
    def identity_name(self) -> str:
        """Human-readable name with no effect on inference."""
        return self.metadata["identity_name"]


def _require_identity_mod(identity_mod: ArtemKo7vKrea2IdentityModData) -> None:
    """Give a clear error for incorrectly connected or externally modified data."""
    if not isinstance(identity_mod, ArtemKo7vKrea2IdentityModData):
        raise ArtemKo7vKrea2IdentityModError("Expected a Krea2 IdentityMod data object.")
    identity_mod.validate()


def _validate_target_match(identity_mod: ArtemKo7vKrea2IdentityModData, target: dict) -> None:
    """Reject mismatches before exposing the raw latent to the existing Krea2 patch."""
    _require_identity_mod(identity_mod)
    samples = _validate_target_latent(target)
    reference = identity_mod.reference_latent
    if samples.shape[-2:] != reference.shape[-2:]:
        height, width = samples.shape[-2:]
        scale = KREA2_VAE_DOWNSCALE_FACTOR
        raise ArtemKo7vKrea2IdentityModError(
            "IdentityMod target geometry mismatch.\n\n"
            f"IdentityMod was created for: "
            f"{identity_mod.target_width}x{identity_mod.target_height}\n"
            f"Current target is: {width * scale}x{height * scale}\n\n"
            "PoC v0.1 IdentityMods are target-resolution-specific. "
            "Create or load an IdentityMod for the current output geometry."
        )
    if samples.shape[1] != reference.shape[1]:
        raise ArtemKo7vKrea2IdentityModError(
            f"Latent channel mismatch: reference C={reference.shape[1]}, "
            f"target C={samples.shape[1]}. Use the matching Krea2 VAE/model combination."
        )
