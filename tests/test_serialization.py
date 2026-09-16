"""Format roundtrip and malformed-file coverage using real safetensors."""

import tempfile
import base64
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tests.support import make_mod
from artemko7v_identitymod_test.identity_mod import ArtemKo7vKrea2IdentityModError
from artemko7v_identitymod_test.serialization import _load_identity_mod, _serialize_identity_mod


class ArtemKo7vSerializationTests(unittest.TestCase):
    """Exercise the actual file format independently of node registration."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "identity.safetensors"

    def test_exact_roundtrip_all_common_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                original = make_mod(dtype)
                self.path.write_bytes(_serialize_identity_mod(original))
                loaded = _load_identity_mod(self.path)
                self.assertTrue(torch.equal(original.reference_latent, loaded.reference_latent))
                self.assertEqual(loaded.reference_latent.dtype, dtype)
                self.assertEqual(dict(loaded.metadata), dict(original.metadata))
                self.assertEqual(loaded.reference_latent.device.type, "cpu")
                self.assertTrue(loaded.reference_latent.is_contiguous())
                with safe_open(str(self.path), framework="pt") as handle:
                    self.assertEqual(handle.keys(), ["appearance_latent"])

    def test_frozen_v01_file_loads_without_vision_cache(self):
        fixture = Path(__file__).parent / "fixtures" / "appearance_v01.safetensors.b64"
        self.path.write_bytes(base64.b64decode(fixture.read_text(encoding="ascii")))
        loaded = _load_identity_mod(self.path)
        self.assertIsNone(loaded.qwen_vision_cache)
        self.assertEqual(loaded.metadata["format_version"], "0.1.0")
        self.assertEqual(loaded.reference_latent.shape, (1, 2, 8, 8))
        self.assertEqual(loaded.identity_name, "Alice")
        self.path.write_bytes(_serialize_identity_mod(loaded))
        again = _load_identity_mod(self.path)
        self.assertTrue(torch.equal(loaded.reference_latent, again.reference_latent))

    def test_invalid_metadata(self):
        for key, value in (
            ("format", "other"), ("format_version", "1.0.0"),
            ("format_version", "0.2.0"), ("format_version", "0.1.0-beta"),
            ("format_version", "0.1"), ("target_width", "65"),
            ("latent_dtype", "torch.float16"), ("latent_channels", "16"),
            ("source_width", "0"), ("created_at_utc", "yesterday"),
            ("created_at_utc", "2026-09-15T00:00:00"), ("qwen_cache_present", "true"),
        ):
            with self.subTest(key=key, value=value):
                original = make_mod()
                metadata = dict(original.metadata, **{key: value})
                save_file({"appearance_latent": original.reference_latent}, self.path,
                          metadata=metadata)
                with self.assertRaises(ArtemKo7vKrea2IdentityModError):
                    _load_identity_mod(self.path)

    def test_every_required_field_is_required(self):
        original = make_mod()
        for key in original.metadata:
            with self.subTest(key=key):
                metadata = dict(original.metadata)
                del metadata[key]
                save_file({"appearance_latent": original.reference_latent}, self.path,
                          metadata=metadata)
                with self.assertRaisesRegex(ArtemKo7vKrea2IdentityModError, "Missing required"):
                    _load_identity_mod(self.path)

    def test_missing_tensor(self):
        save_file({"other": torch.zeros(1)}, self.path, metadata=dict(make_mod().metadata))
        with self.assertRaisesRegex(ArtemKo7vKrea2IdentityModError, "appearance_latent"):
            _load_identity_mod(self.path)

    def test_invalid_tensors(self):
        for tensor in (
            torch.zeros(2, 8, 8), torch.zeros(1, 2, 1, 8, 8), torch.zeros(2, 2, 8, 8),
            torch.zeros(1, 2, 0, 8), torch.zeros(1, 2, 8, 8, dtype=torch.int64),
            torch.full((1, 2, 8, 8), float("nan")), torch.zeros(1, 2, 9, 8),
        ):
            with self.subTest(shape=tensor.shape, dtype=tensor.dtype):
                save_file({"appearance_latent": tensor}, self.path,
                          metadata=dict(make_mod().metadata))
                with self.assertRaises(ArtemKo7vKrea2IdentityModError):
                    _load_identity_mod(self.path)

    def test_compatible_patch_version_and_unknown_fields(self):
        original = make_mod()
        metadata = dict(original.metadata, format_version="0.1.12", future_note="ignored")
        save_file({"appearance_latent": original.reference_latent, "future": torch.zeros(1)},
                  self.path, metadata=metadata)
        self.assertEqual(_load_identity_mod(self.path).metadata, metadata)

    def test_corrupt_and_missing_files(self):
        self.path.write_bytes(b"invalid file")
        with self.assertRaisesRegex(ArtemKo7vKrea2IdentityModError, "Unable to load"):
            _load_identity_mod(self.path)
        with self.assertRaises(ArtemKo7vKrea2IdentityModError):
            _load_identity_mod(self.path.with_name("missing.safetensors"))
