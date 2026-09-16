"""Cache extraction, preprocessing, validation, and lossless file coverage."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from tests.support import make_mod
from tests.qwen_support import (
    make_cache, make_full_mod, ArtemKo7vFakeClip, install_model_management_stub,
)
from artemko7v_identitymod_test import qwen_cache
from artemko7v_identitymod_test.constants import QWEN_CACHE_TENSOR_KEYS
from artemko7v_identitymod_test.identity_mod import (
    ArtemKo7vKrea2IdentityModData, ArtemKo7vKrea2IdentityModError, ArtemKo7vKrea2QwenVisionCache,
)
from artemko7v_identitymod_test.serialization import _load_identity_mod, _serialize_identity_mod


class ArtemKo7vQwenCacheTests(unittest.TestCase):
    """Exercise the portable schema without importing the ComfyUI installation."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "full.safetensors"

    def test_roundtrip_exact_all_tensors_and_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                original = make_full_mod(dtype)
                self.path.write_bytes(_serialize_identity_mod(original))
                loaded = _load_identity_mod(self.path)
                self.assertEqual(original.metadata, loaded.metadata)
                self.assertTrue(torch.equal(original.reference_latent, loaded.reference_latent))
                for a, b in zip(self._tensors(original).values(), self._tensors(loaded).values()):
                    self.assertTrue(torch.equal(a, b))
                    self.assertEqual(a.dtype, b.dtype)
                    self.assertEqual(a.shape, b.shape)
                    self.assertEqual(b.device.type, "cpu")
                    self.assertTrue(b.is_contiguous())
                    self.assertFalse(b.requires_grad)

    @staticmethod
    def _tensors(mod):
        cache = mod.qwen_vision_cache
        return {"appearance_latent": mod.reference_latent,
                **dict(zip(QWEN_CACHE_TENSOR_KEYS, (cache.merged, cache.grid, *cache.deepstack)))}

    def test_every_vision_tensor_required_and_extra_rejected(self):
        mod = make_full_mod()
        for key in QWEN_CACHE_TENSOR_KEYS:
            tensors = self._tensors(mod)
            del tensors[key]
            save_file(tensors, self.path, metadata=dict(mod.metadata))
            with self.subTest(key=key), self.assertRaises(ArtemKo7vKrea2IdentityModError):
                _load_identity_mod(self.path)
        tensors = self._tensors(mod)
        tensors["qwen_vision_deepstack_3"] = torch.zeros(1)
        save_file(tensors, self.path, metadata=dict(mod.metadata))
        with self.assertRaises(ArtemKo7vKrea2IdentityModError):
            _load_identity_mod(self.path)

    def test_every_qwen_metadata_field_required_and_cross_checked(self):
        mod = make_full_mod()
        for key in mod.metadata:
            if not key.startswith("qwen_"):
                continue
            for value in (None, "invalid"):
                metadata = dict(mod.metadata)
                if value is None:
                    del metadata[key]
                else:
                    metadata[key] = value
                save_file(self._tensors(mod), self.path, metadata=metadata)
                with self.subTest(key=key, value=value), self.assertRaises(ArtemKo7vKrea2IdentityModError):
                    _load_identity_mod(self.path)

    def test_reject_invalid_tensor_types_shapes_and_values(self):
        cache = make_cache()
        for name, value in (
            ("merged", cache.merged.long()), ("merged", cache.merged[:, :32]),
            ("merged", torch.full_like(cache.merged, float("nan"))),
            ("grid", cache.grid.float()), ("grid", torch.tensor([[1, 2, 2]])),
            ("grid", torch.tensor([[2, 4, 4]])), ("grid", torch.tensor([[1, 3, 4]])),
            ("grid", cache.grid.flatten()), ("deepstack", cache.deepstack[:2]),
            ("deepstack", (cache.merged.long(),) * 3),
            ("deepstack", (cache.merged[:2],) * 3),
        ):
            kwargs = dict(merged=cache.merged, grid=cache.grid, deepstack=cache.deepstack)
            kwargs[name] = value
            with self.subTest(name=name), self.assertRaises(ArtemKo7vKrea2IdentityModError):
                ArtemKo7vKrea2QwenVisionCache(**kwargs)

    def test_version_payload_consistency_and_compatible_patch(self):
        full = make_full_mod()
        for version, flag in (("0.1.0", "true"), ("0.1.0", "false"), ("0.2.0", "false")):
            save_file(self._tensors(full), self.path,
                      metadata=dict(full.metadata, format_version=version, qwen_cache_present=flag))
            with self.assertRaises(ArtemKo7vKrea2IdentityModError):
                _load_identity_mod(self.path)
        save_file(self._tensors(full), self.path, metadata=dict(full.metadata, format_version="0.2.12"))
        self.assertEqual(_load_identity_mod(self.path).metadata["format_version"], "0.2.12")

    def test_cache_owns_detached_contiguous_storage(self):
        for device in ("cpu",):
            raw = torch.arange(10240.0, device=device).reshape(2560, 4).T.requires_grad_()
            grid = torch.tensor([[1, 4, 4]], device=device)
            cache = ArtemKo7vKrea2QwenVisionCache(raw, grid, (raw, raw, raw))
            expected = raw.detach().cpu().clone()
            raw.detach().zero_()
            grid.zero_()
            for tensor in (cache.merged, *cache.deepstack):
                self.assertEqual(tensor.device.type, "cpu")
                self.assertTrue(tensor.is_contiguous())
                self.assertFalse(tensor.requires_grad)
                self.assertTrue(torch.equal(tensor, expected))
            self.assertEqual(cache.grid.tolist(), [[1, 4, 4]])

    def test_extract_once_and_replace_without_changing_input(self):
        install_model_management_stub(self)
        clip = ArtemKo7vFakeClip()
        mod = make_mod()
        image = torch.rand(1, 64, 128, 4)
        with patch.object(qwen_cache, "_prepare_grounding_image",
                          wraps=qwen_cache._prepare_grounding_image) as prepare:
            full = qwen_cache._add_qwen_vision_cache(mod, clip, image, 64)
        prepare.assert_called_once()
        self.assertEqual(clip.loads, 1)
        self.assertEqual(clip.transformer.preprocess_calls, 1)
        self.assertEqual(clip.encodes, 0)
        self.assertEqual(clip.transformer.image.shape, (1, 32, 64, 3))
        self.assertIsNone(mod.qwen_vision_cache)
        self.assertTrue(torch.equal(mod.reference_latent, full.reference_latent))
        expected, extra = clip.transformer.last_output
        self.assertTrue(torch.equal(full.qwen_vision_cache.merged, expected))
        for a, b in zip(full.qwen_vision_cache.deepstack, extra["deepstack"]):
            self.assertTrue(torch.equal(a, b))
        replaced = qwen_cache._add_qwen_vision_cache(full, clip, image + 1, 0)
        self.assertIsNot(full, replaced)
        self.assertEqual(full.metadata["qwen_grounding_px"], "64")
        self.assertEqual(replaced.metadata["qwen_grounding_px"], "0")
        self.assertFalse(torch.equal(full.qwen_vision_cache.merged, replaced.qwen_vision_cache.merged))

    def test_grounding_preprocessing(self):
        image = torch.randn(1, 90, 120, 4)
        for limit in (0, 120, 768):
            self.assertTrue(torch.equal(qwen_cache._prepare_grounding_image(image, limit), image[..., :3]))
        expected = torch.nn.functional.interpolate(image.movedim(-1, 1), (48, 64), mode="area")
        self.assertTrue(torch.equal(qwen_cache._prepare_grounding_image(image, 64),
                                    expected.movedim(1, -1)[..., :3]))
        for invalid in (-1, 4097, True, 1.5):
            with self.assertRaises(ArtemKo7vKrea2IdentityModError):
                qwen_cache._prepare_grounding_image(image, invalid)
        with self.assertRaises(ArtemKo7vKrea2IdentityModError):
            qwen_cache._prepare_grounding_image(image.repeat(2, 1, 1, 1), 64)

    def test_unsupported_clip(self):
        for clip in (None, object(), ArtemKo7vFakeClip()):
            if isinstance(clip, ArtemKo7vFakeClip):
                clip.transformer.model_type = "qwen3vl_8b"
            with self.assertRaisesRegex(ArtemKo7vKrea2IdentityModError, "Unsupported CLIP"):
                qwen_cache._get_krea2_qwen3vl_transformer(clip)
