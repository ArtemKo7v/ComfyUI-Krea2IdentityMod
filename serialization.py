"""Lossless safetensors serialization, with no Python object deserialization."""

from pathlib import Path

from safetensors import SafetensorError, safe_open
from safetensors.torch import save

from .constants import IDENTITY_MOD_TENSOR_KEY, QWEN_CACHE_TENSOR_KEYS
from .identity_mod import (
    ArtemKo7vKrea2IdentityModData,
    ArtemKo7vKrea2IdentityModError,
    ArtemKo7vKrea2QwenVisionCache,
    _require_identity_mod,
)


def _serialize_identity_mod(identity_mod: ArtemKo7vKrea2IdentityModData) -> bytes:
    """Encode appearance and optional vision tensors without changing their dtype."""
    _require_identity_mod(identity_mod)
    tensor = identity_mod.reference_latent.detach().cpu().contiguous()
    tensors = {IDENTITY_MOD_TENSOR_KEY: tensor}
    cache = identity_mod.qwen_vision_cache
    if cache is not None:
        tensors.update(zip(QWEN_CACHE_TENSOR_KEYS, (cache.merged, cache.grid, *cache.deepstack)))
    return save(tensors, metadata=dict(identity_mod.metadata))


def _load_identity_mod(path: Path) -> ArtemKo7vKrea2IdentityModData:
    """Load on CPU and reject incomplete or contradictory cache payloads."""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if IDENTITY_MOD_TENSOR_KEY not in handle.keys():
                raise ArtemKo7vKrea2IdentityModError(
                    f"IdentityMod is missing required tensor '{IDENTITY_MOD_TENSOR_KEY}'."
                )
            tensor = handle.get_tensor(IDENTITY_MOD_TENSOR_KEY)
            cache = None
            cache_keys = {key for key in handle.keys() if key.startswith("qwen_vision_")}
            expects_cache = metadata.get("qwen_cache_present") == "true"
            if cache_keys or expects_cache:
                if not expects_cache or cache_keys != set(QWEN_CACHE_TENSOR_KEYS):
                    raise ArtemKo7vKrea2IdentityModError(
                        "Missing, extra, or contradictory Qwen vision cache tensors; "
                        "rebuild with Krea2 IdentityMod Add Qwen Vision Cache."
                    )
                merged, grid, *deepstack = [handle.get_tensor(key) for key in QWEN_CACHE_TENSOR_KEYS]
                cache = ArtemKo7vKrea2QwenVisionCache(merged, grid, tuple(deepstack))
        return ArtemKo7vKrea2IdentityModData(tensor, metadata, cache)
    except (OSError, SafetensorError) as error:
        raise ArtemKo7vKrea2IdentityModError(
            f"Unable to load IdentityMod file '{path.name}': {error}"
        ) from error
