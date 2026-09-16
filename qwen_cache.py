"""Extract prompt-independent vision features through the loaded Krea2 CLIP API."""

from contextlib import contextmanager
from threading import RLock

import torch
import torch.nn.functional as F

from .constants import IDENTITY_MOD_QWEN_FORMAT_VERSION, KREA2_QWEN_TAP_LAYERS
from .identity_mod import (
    ArtemKo7vKrea2IdentityModData, ArtemKo7vKrea2IdentityModError,
    ArtemKo7vKrea2QwenVisionCache, _require_identity_mod,
)
from .preprocessing import _validate_image

# CLIP clones share the transformer. Extraction and injection use the same lock.
_QWEN_PREPROCESS_PATCH_LOCK = RLock()


def _get_krea2_qwen3vl_transformer(clip):
    """Validate the Krea2 12-tap, Qwen3-VL 4B integration before using its API."""
    stage = getattr(clip, "cond_stage_model", None)
    encoder = getattr(stage, "qwen3vl_4b", None)
    transformer = getattr(encoder, "transformer", None)
    layers = getattr(encoder, "layer", None)
    if (transformer is None or getattr(transformer, "model_type", None) != "qwen3vl_4b"
            or not isinstance(layers, (list, tuple))
            or tuple(layers) != KREA2_QWEN_TAP_LAYERS
            or not callable(getattr(transformer, "preprocess_embed", None))
            or not callable(getattr(transformer, "visual", None))
            or not callable(getattr(clip, "load_model", None))
            or not callable(getattr(clip, "tokenize", None))
            or not callable(getattr(clip, "encode_from_tokens_scheduled", None))
            or getattr(getattr(clip, "patcher", None), "load_device", None) is None):
        raise ArtemKo7vKrea2IdentityModError(
            "Unsupported CLIP. Use the Krea2 Qwen3-VL 4B text encoder with its 12-layer taps "
            "and a ComfyUI version exposing Qwen3-VL visual preprocessing."
        )
    return transformer


def _prepare_grounding_image(image: torch.Tensor, grounding_px: int) -> torch.Tensor:
    """Match upstream longest-side area downscaling; do not alter color or precision."""
    _validate_image(image)
    if type(grounding_px) is not int or not 0 <= grounding_px <= 4096:
        raise ArtemKo7vKrea2IdentityModError("grounding_px must be an integer from 0 to 4096.")
    height, width = image.shape[1:3]
    samples = image.movedim(-1, 1)
    if grounding_px and max(height, width) > grounding_px:
        scale = grounding_px / max(height, width)
        size = (round(height * scale), round(width * scale))
        if min(size) < 1:
            raise ArtemKo7vKrea2IdentityModError(
                "Grounding resolution makes an image dimension zero; increase grounding_px."
            )
        samples = F.interpolate(samples, size=size, mode="area")
    return samples.movedim(1, -1)[..., :3]


@contextmanager
def _loaded_qwen_device(clip):
    """Use ComfyUI's loading/offloading policy and execution-device context."""
    from comfy import model_management

    clip.load_model()
    device = clip.patcher.load_device
    with model_management.cuda_device_context(device):
        yield device


@torch.no_grad()
def _add_qwen_vision_cache(
    identity_mod: ArtemKo7vKrea2IdentityModData, clip, image: torch.Tensor,
    grounding_px: int = 768,
) -> ArtemKo7vKrea2IdentityModData:
    """Run the vision boundary once and return a new, CPU-resident full IdentityMod."""
    _require_identity_mod(identity_mod)
    transformer = _get_krea2_qwen3vl_transformer(clip)
    prepared = _prepare_grounding_image(image, grounding_px)
    with _QWEN_PREPROCESS_PATCH_LOCK, _loaded_qwen_device(clip) as device:
        merged, extra = transformer.preprocess_embed({"type": "image", "data": prepared}, device=device)
        if not isinstance(extra, dict) or "grid" not in extra or "deepstack" not in extra:
            raise ArtemKo7vKrea2IdentityModError(
                "Qwen preprocessing did not return grid and DeepStack; use a compatible ComfyUI."
            )
        cache = ArtemKo7vKrea2QwenVisionCache(merged, extra["grid"], extra["deepstack"])
    metadata = dict(identity_mod.metadata)
    metadata.update(cache.tensor_metadata())
    metadata.update({
        "format_version": IDENTITY_MOD_QWEN_FORMAT_VERSION,
        "qwen_grounding_px": str(grounding_px),
        "qwen_input_width": str(prepared.shape[2]), "qwen_input_height": str(prepared.shape[1]),
    })
    return ArtemKo7vKrea2IdentityModData(identity_mod.reference_latent, metadata, cache)
