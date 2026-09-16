"""Scoped image-feature injection; tokenization and language inference stay in ComfyUI."""

from contextlib import contextmanager
from numbers import Real

import torch

from .identity_mod import (
    ArtemKo7vKrea2IdentityModData, ArtemKo7vKrea2IdentityModError,
    ArtemKo7vKrea2QwenVisionCache, _require_identity_mod,
)
from .qwen_cache import _get_krea2_qwen3vl_transformer, _QWEN_PREPROCESS_PATCH_LOCK

_CACHE_MARKER = "artemko7v_qwen3vl_cached"
_DEFAULT_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)


def _grounding_template(system_prompt: str = "") -> str:
    """Keep upstream single-image ChatML layout and whitespace exactly."""
    system = system_prompt.strip() or _DEFAULT_SYSTEM_PROMPT
    # The tokenizer formats this string with the current prompt, just as upstream does.
    return (
        "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
    )


def _cached_tokens(clip, prompt: str, system_prompt: str) -> tuple[dict, object]:
    """Replace exactly one image descriptor without passing any source pixels."""
    if not isinstance(prompt, str) or not isinstance(system_prompt, str):
        raise ArtemKo7vKrea2IdentityModError("Prompt and system_prompt must be strings.")
    marker = object()
    tokens = clip.tokenize(prompt, images=[marker], llama_template=_grounding_template(system_prompt))
    if not isinstance(tokens, dict) or "qwen3vl_4b" not in tokens:
        raise ArtemKo7vKrea2IdentityModError(
            "Unsupported token layout. Use the Krea2 Qwen3-VL 4B tokenizer."
        )
    descriptors = []
    unfilled = 0
    for rows in tokens.values():
        for row in rows:
            for entry in row:
                token = entry[0]
                if isinstance(token, dict) and token.get("type") == "image":
                    descriptors.append(token)
                elif isinstance(token, Real) and token == 151655:
                    unfilled += 1
    if len(descriptors) != 1 or unfilled or descriptors[0].get("data") is not marker:
        raise ArtemKo7vKrea2IdentityModError(
            "Cached grounded encoding requires exactly one image placeholder. "
            "Remove extra image/ChatML tokens from the prompt and use the Krea2 tokenizer."
        )
    descriptors[0].pop("data")
    descriptors[0][_CACHE_MARKER] = marker
    return tokens, marker


@contextmanager
def _cached_preprocess(transformer, cache: ArtemKo7vKrea2QwenVisionCache, marker: object):
    """Intercept only this invocation's descriptor, restoring even nested/error paths."""
    with _QWEN_PREPROCESS_PATCH_LOCK:
        original = transformer.preprocess_embed
        had_instance_attribute = "preprocess_embed" in vars(transformer)

        def preprocess(embed, device):
            if (isinstance(embed, dict) and embed.get("type") == "image"
                    and embed.get(_CACHE_MARKER) is marker):
                # Copies also protect CPU caches against downstream in-place operations.
                return cache.merged.to(device=device, copy=True), {
                    # Upstream builds grid on the image device (normally CPU).
                    "grid": cache.grid.clone(),
                    "deepstack": [t.to(device=device, copy=True) for t in cache.deepstack],
                }
            return original(embed, device=device)

        try:
            transformer.preprocess_embed = preprocess
            yield
        finally:
            if had_instance_attribute:
                transformer.preprocess_embed = original
            else:
                delattr(transformer, "preprocess_embed")


@torch.no_grad()
def _encode_cached_grounded(
    clip, identity_mod: ArtemKo7vKrea2IdentityModData, prompt: str, system_prompt: str = "",
) -> list:
    """Recompute prompt-dependent conditioning using the stock scheduled CLIP path."""
    _require_identity_mod(identity_mod)
    cache = identity_mod.qwen_vision_cache
    if cache is None:
        raise ArtemKo7vKrea2IdentityModError(
            "This IdentityMod does not contain a Qwen3-VL vision cache.\n\n"
            "It still works with Krea2 IdentityMod To Latent. For image-free grounded encoding, "
            "use Krea2 IdentityMod Add Qwen Vision Cache and save the updated IdentityMod."
        )
    transformer = _get_krea2_qwen3vl_transformer(clip)
    with _QWEN_PREPROCESS_PATCH_LOCK:
        tokens, marker = _cached_tokens(clip, prompt, system_prompt)
        with _cached_preprocess(transformer, cache, marker):
            return clip.encode_from_tokens_scheduled(tokens)
