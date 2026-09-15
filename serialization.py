"""Lossless safetensors serialization, with no Python object deserialization."""

from pathlib import Path

from safetensors import SafetensorError, safe_open
from safetensors.torch import save

from .constants import IDENTITY_MOD_TENSOR_KEY
from .identity_mod import (
    ArtemKo7vKrea2IdentityModData,
    ArtemKo7vKrea2IdentityModError,
    _require_identity_mod,
)


def _serialize_identity_mod(identity_mod: ArtemKo7vKrea2IdentityModData) -> bytes:
    """Encode exactly one CPU tensor while preserving dtype and all metadata."""
    _require_identity_mod(identity_mod)
    tensor = identity_mod.reference_latent.detach().cpu().contiguous()
    return save({IDENTITY_MOD_TENSOR_KEY: tensor}, metadata=dict(identity_mod.metadata))


def _load_identity_mod(path: Path) -> ArtemKo7vKrea2IdentityModData:
    """Load on CPU, tolerate unknown keys, and validate the complete v0.1 payload."""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if IDENTITY_MOD_TENSOR_KEY not in handle.keys():
                raise ArtemKo7vKrea2IdentityModError(
                    f"IdentityMod is missing required tensor '{IDENTITY_MOD_TENSOR_KEY}'."
                )
            tensor = handle.get_tensor(IDENTITY_MOD_TENSOR_KEY)
        return ArtemKo7vKrea2IdentityModData(tensor, metadata)
    except (OSError, SafetensorError) as error:
        raise ArtemKo7vKrea2IdentityModError(
            f"Unable to load IdentityMod file '{path.name}': {error}"
        ) from error
