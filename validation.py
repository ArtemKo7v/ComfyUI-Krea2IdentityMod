"""Opt-in, in-process real-model A/B helper; never run or register it on import.

The caller supplies an already loaded CLIP, IMAGE, appearance IdentityMod, and the
actual upstream Krea2EditGroundedEncode.encode callable. No model discovery or
downloads are performed. Run only in an otherwise idle ComfyUI process.
"""

import logging
from pathlib import Path
import tempfile
from time import perf_counter
from unittest.mock import patch

import torch

from .identity_mod import ArtemKo7vKrea2IdentityModData
from .qwen_cache import _add_qwen_vision_cache, _get_krea2_qwen3vl_transformer, _QWEN_PREPROCESS_PATCH_LOCK
from .qwen_injection import _encode_cached_grounded
from .serialization import _load_identity_mod, _serialize_identity_mod

logger = logging.getLogger(__name__)
_PROMPTS = (
    "Change her outfit to a black leather jacket.",
    "Put her in a snowy mountain environment.",
    "Make this a cinematic night portrait. Keep her facial features and hairstyle, "
    "use soft blue rim lighting and warm street lights reflected in the wet pavement, "
    "with a shallow depth of field and an out-of-focus city behind her.",
    "",
)


def _compare_tensors(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> dict:
    """Compute precision-independent error metrics after checking shape and dtype."""
    same_shape = a.shape == b.shape
    same_dtype = a.dtype == b.dtype
    left, right = a.detach().cpu(), b.detach().cpu()
    delta = (left.double() - right.double()).abs() if same_shape else None
    return {
        "shape_a": list(a.shape), "shape_b": list(b.shape),
        "dtype_a": str(a.dtype), "dtype_b": str(b.dtype),
        "max_absolute_error": delta.max().item() if delta is not None and delta.numel() else None,
        "mean_absolute_error": delta.mean().item() if delta is not None and delta.numel() else None,
        "equal": same_shape and same_dtype and torch.equal(left, right),
        "allclose": same_shape and same_dtype and torch.allclose(left, right, atol=atol, rtol=rtol),
    }


def _compare_conditioning(a: list, b: list, atol: float, rtol: float) -> list[dict]:
    """Compare all scheduled entries, including masks and pooled tensor metadata."""
    if len(a) != len(b):
        raise AssertionError("Direct and cached conditioning have different schedule lengths.")
    reports = []
    for index, ((left, left_meta), (right, right_meta)) in enumerate(zip(a, b)):
        reports.append({"entry": index, "field": "cond", **_compare_tensors(left, right, atol, rtol)})
        if left_meta.keys() != right_meta.keys():
            raise AssertionError("Conditioning metadata keys differ.")
        for key in left_meta:
            lv, rv = left_meta[key], right_meta[key]
            if isinstance(lv, torch.Tensor) and isinstance(rv, torch.Tensor):
                reports.append({"entry": index, "field": key, **_compare_tensors(lv, rv, atol, rtol)})
            elif lv is not rv and lv != rv:
                raise AssertionError(f"Conditioning metadata differs: {key}.")
    return reports


def run_qwen_ab(
    clip, appearance: ArtemKo7vKrea2IdentityModData, image: torch.Tensor, direct_encode,
    grounding_values: tuple[int, ...] = (512, 768, 1024),
    prompts: tuple[str, ...] = _PROMPTS,
    system_prompts: tuple[str, ...] = ("", "Describe the reference precisely and follow the editing instruction."),
    atol: float = 1e-6, rtol: float = 1e-5,
) -> list[dict]:
    """Return JSON-ready real-model measurements, reusing one saved cache per size.

    The reported tolerances are diagnostic, not automatic acceptance criteria.
    Investigate every non-exact result. Times include encode overhead and device
    synchronization, not cache construction. This does not test the image sampler.
    """
    transformer = _get_krea2_qwen3vl_transformer(clip)
    device = torch.device(clip.patcher.load_device)

    def timed(call):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        result = call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return result, perf_counter() - start

    records = []
    with _QWEN_PREPROCESS_PATCH_LOCK, torch.no_grad(), tempfile.TemporaryDirectory() as directory:
        for grounding_px in grounding_values:
            created = _add_qwen_vision_cache(appearance, clip, image, grounding_px)
            path = Path(directory) / f"grounding_{grounding_px}.safetensors"
            path.write_bytes(_serialize_identity_mod(created))
            loaded = _load_identity_mod(path)
            del created
            for system_prompt in system_prompts:
                for prompt in prompts:
                    direct_call = lambda: direct_encode(
                        clip=clip, prompt=prompt, image=image, grounding_px=grounding_px,
                        system_prompt=system_prompt,
                    )[0]
                    cached_call = lambda: _encode_cached_grounded(clip, loaded, prompt, system_prompt)
                    direct, direct_seconds = timed(direct_call)
                    cached, cached_seconds = timed(cached_call)
                    metrics = _compare_conditioning(direct, cached, atol, rtol)
                    # Instrument the actual tower, not merely the adapter boundary.
                    tower = transformer.visual
                    owner, name = (tower, "forward") if hasattr(tower, "forward") else (transformer, "visual")
                    with patch.object(owner, name, side_effect=RuntimeError("Vision tower should not run")):
                        guarded = cached_call()
                        guarded_metrics = _compare_conditioning(cached, guarded, atol, rtol)
                        direct_failed = False
                        try:
                            direct_call()
                        except RuntimeError as error:
                            if "Vision tower should not run" not in str(error):
                                raise
                            direct_failed = True
                        if not direct_failed:
                            raise AssertionError("Direct encoding did not reach the instrumented vision tower.")
                    record = {
                        "grounding_px": grounding_px, "prompt": prompt, "system_prompt": system_prompt,
                        "atol": atol, "rtol": rtol, "direct_seconds": direct_seconds,
                        "cached_seconds": cached_seconds, "tensors": metrics,
                        "guarded_cached_tensors": guarded_metrics,
                        "cached_succeeds_with_vision_disabled": True, "direct_fails_with_vision_disabled": True,
                    }
                    records.append(record)
                    logger.info("Qwen A/B: grounding=%d exact=%s direct=%.3fs cached=%.3fs",
                                grounding_px, all(item["equal"] for item in metrics),
                                direct_seconds, cached_seconds)
                    del direct, cached, guarded
    return records
