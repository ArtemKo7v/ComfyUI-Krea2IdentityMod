"""Geometry, RGB handling, and exactly-once VAE encoding tests."""

import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F

from tests.support import ArtemKo7vFakeVAE
from artemko7v_identitymod_test.identity_mod import ArtemKo7vKrea2IdentityModError
from artemko7v_identitymod_test.preprocessing import _create_identity_mod, _prepare_reference_image


class ArtemKo7vPreprocessingTests(unittest.TestCase):
    """Validate supported preprocessing before exposing it through ComfyUI nodes."""

    def test_near_match_center_crop_and_rgb(self):
        image = torch.linspace(-0.2, 1.2, 100 * 108 * 4).reshape(1, 100, 108, 4)
        prepared = _prepare_reference_image(image, 64, 64)
        expected = F.interpolate(image[:, :, 4:104].movedim(-1, 1), size=(64, 64),
                                 mode="bicubic", antialias=True).movedim(1, -1)[..., :3].clamp(0, 1)
        self.assertTrue(torch.equal(prepared, expected))
        self.assertEqual(prepared.shape, (1, 64, 64, 3))
        self.assertGreaterEqual(prepared.min().item(), 0)
        self.assertLessEqual(prepared.max().item(), 1)

    def test_aspect_ratio_boundary(self):
        _prepare_reference_image(torch.zeros(1, 92, 100, 3), 64, 64)
        with self.assertRaisesRegex(ArtemKo7vKrea2IdentityModError, "aspect-ratio"):
            _prepare_reference_image(torch.zeros(1, 91, 100, 3), 64, 64)

    def test_unsupported_aspect_ratio_fails_before_encode(self):
        vae = ArtemKo7vFakeVAE()
        with self.assertRaisesRegex(ArtemKo7vKrea2IdentityModError, "1344x768"):
            _create_identity_mod(torch.zeros(1, 1024, 1024, 3), vae,
                                 {"samples": torch.zeros(1, 2, 96, 168)}, "Alice")
        self.assertEqual(vae.calls, 0)

    def test_supported_geometries_and_single_encode(self):
        for height, width in ((64, 64), (96, 64), (64, 96)):
            with self.subTest(height=height, width=width):
                vae = ArtemKo7vFakeVAE()
                image = torch.rand(1, height * 2, width * 2, 3)
                mod = _create_identity_mod(
                    image, vae, {"samples": torch.zeros(1, 2, height // 8, width // 8)}, "A"
                )
                self.assertEqual(vae.calls, 1)
                self.assertEqual(vae.image.shape, (1, height, width, 3))
                self.assertEqual((mod.target_height, mod.target_width), (height, width))
                self.assertTrue(torch.equal(
                    mod.reference_latent, vae.image[:, ::8, ::8, :2].movedim(-1, 1)
                ))
                self.assertFalse(mod.reference_latent.requires_grad)

    def test_invalid_images(self):
        for image in (
            None, torch.zeros(64, 64, 3), torch.zeros(2, 64, 64, 3),
            torch.zeros(0, 64, 64, 3), torch.zeros(1, 0, 64, 3),
            torch.zeros(1, 64, 64, 2), torch.zeros(1, 64, 64, 3, dtype=torch.uint8),
            torch.full((1, 64, 64, 3), float("nan")),
        ):
            with self.subTest(image_type=type(image)):
                with self.assertRaises(ArtemKo7vKrea2IdentityModError):
                    _prepare_reference_image(image, 64, 64)

    def test_interpolation_uses_float32(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            prepared = _prepare_reference_image(torch.ones(1, 100, 100, 4, dtype=dtype), 64, 64)
            self.assertEqual(prepared.dtype, torch.float32)

    def test_wrong_vae_shape_and_channels(self):
        for shape in ((1, 2, 6, 8), (1, 4, 8, 8), (2, 2, 8, 8), (1, 2, 2, 8, 8)):
            vae = Mock()
            vae.encode.return_value = torch.zeros(shape)
            with self.subTest(shape=shape):
                with self.assertRaises(ArtemKo7vKrea2IdentityModError):
                    _create_identity_mod(torch.zeros(1, 64, 64, 3), vae,
                                         {"samples": torch.zeros(1, 2, 8, 8)}, "A")
                vae.encode.assert_called_once()

    def test_singleton_temporal_axis_and_raw_dtype(self):
        vae = Mock()
        raw = torch.randn(1, 2, 1, 8, 8, dtype=torch.float16)
        vae.encode.return_value = raw
        mod = _create_identity_mod(torch.zeros(1, 64, 64, 3), vae,
                                   {"samples": torch.zeros(3, 2, 1, 8, 8)}, "A")
        self.assertTrue(torch.equal(mod.reference_latent, raw.squeeze(2)))
        self.assertEqual(mod.reference_latent.dtype, torch.float16)

    def test_invalid_target_never_calls_vae(self):
        for target in (
            None, {}, {"samples": None}, {"samples": torch.zeros(2, 8, 8)},
            {"samples": torch.zeros(1, 2, 2, 8, 8)}, {"samples": torch.zeros(1, 2, 7, 8)},
            {"samples": torch.zeros(1, 2, 8, 0)},
        ):
            with self.subTest(target_type=type(target)):
                vae = ArtemKo7vFakeVAE()
                with self.assertRaises(ArtemKo7vKrea2IdentityModError):
                    _create_identity_mod(torch.zeros(1, 64, 64, 3), vae, target, "A")
                self.assertEqual(vae.calls, 0)
