"""Node registration, lifecycle, cache invalidation, and raw LATENT handoff tests."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tests.support import ROOT, ArtemKo7vFakeVAE, install_folder_paths_stub


class ArtemKo7vNodeTests(unittest.TestCase):
    """Run the full node lifecycle using real serialization and a lightweight VAE."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.stub = install_folder_paths_stub(self, Path(self.directory.name))
        from artemko7v_identitymod_test import nodes
        self.nodes = nodes

    def test_lifecycle_and_no_vae_on_loaded_path(self):
        vae = ArtemKo7vFakeVAE()
        target = {"samples": torch.zeros(1, 2, 8, 8)}
        created, = self.nodes.ArtemKo7vKrea2IdentityModCreate().create(
            torch.rand(1, 100, 108, 4), vae, target, "Alice", "Test identity"
        )
        filename, = self.nodes.ArtemKo7vKrea2IdentityModSave().save(created, "alice/test")
        loaded, = self.nodes.ArtemKo7vKrea2IdentityModLoad().load(filename)
        with patch.object(vae, "encode", side_effect=AssertionError("Unexpected VAE encode")):
            with patch("torch.nn.functional.interpolate", side_effect=AssertionError("Resize")):
                latent, = self.nodes.ArtemKo7vKrea2IdentityModToLatent().to_latent(loaded, target)
                info, = self.nodes.ArtemKo7vKrea2IdentityModInfo().get_info(loaded)
        self.assertEqual(latent["samples"].shape, (1, 2, 1, 8, 8))
        self.assertTrue(torch.equal(latent["samples"].squeeze(2), created.reference_latent))
        self.assertEqual(latent["samples"].dtype, created.reference_latent.dtype)
        self.assertEqual(set(latent), {"samples"})
        self.assertIn("Name: Alice", info)
        self.assertIn("Qwen cache: no", info)
        self.assertEqual(vae.calls, 1)
        latent["samples"].zero_()
        self.assertTrue(torch.equal(loaded.reference_latent, created.reference_latent))

    def test_load_cache_invalidation(self):
        from tests.support import make_mod
        from artemko7v_identitymod_test.paths import _resolve_identity_mod_file
        filename, = self.nodes.ArtemKo7vKrea2IdentityModSave().save(make_mod(), "cached")
        before = self.nodes.ArtemKo7vKrea2IdentityModLoad.IS_CHANGED(filename)
        path = _resolve_identity_mod_file(filename)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
        after = self.nodes.ArtemKo7vKrea2IdentityModLoad.IS_CHANGED(filename)
        self.assertNotEqual(before, after)

    def test_handoff_preserves_channelwise_5d_model_normalization(self):
        """A 4D source broadcasts channels into frames in Krea2's Wan21 normalization."""
        from tests.support import make_mod
        from artemko7v_identitymod_test.identity_mod import ArtemKo7vKrea2IdentityModData
        original = make_mod()
        raw = torch.randn(1, 16, 8, 8, generator=torch.Generator().manual_seed(31))
        mod = ArtemKo7vKrea2IdentityModData(
            raw, dict(original.metadata, latent_channels="16")
        )
        # Distinct per-channel constants expose accidental broadcasting without model weights.
        mean = torch.linspace(-1, 1, 16).reshape(1, 16, 1, 1, 1)
        std = torch.linspace(1, 3, 16).reshape(1, 16, 1, 1, 1)
        expected = (raw - mean.squeeze(2)) / std.squeeze(2)
        malformed = (raw - mean) / std
        self.assertEqual(malformed.shape, (1, 16, 16, 8, 8))
        # The upstream 5D-to-4D reshape followed by batch selection retains wrong scaling.
        malformed_first = malformed.reshape(16, 16, 8, 8)[:1]
        self.assertFalse(torch.equal(malformed_first, expected))

        for target_shape in ((1, 16, 8, 8), (3, 16, 1, 8, 8)):
            with self.subTest(target_shape=target_shape):
                latent, = self.nodes.ArtemKo7vKrea2IdentityModToLatent().to_latent(
                    mod, {"samples": torch.zeros(target_shape)}
                )
                normalized = (latent["samples"] - mean) / std
                self.assertEqual(normalized.shape, (1, 16, 1, 8, 8))
                self.assertTrue(torch.equal(normalized.squeeze(2), expected))
                self.assertTrue(torch.equal(latent["samples"].squeeze(2), raw))

    def test_package_registration_and_node_contracts(self):
        name = "artemko7v_identitymod_entry_test"
        spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.addCleanup(lambda: [sys.modules.pop(key, None) for key in list(sys.modules)
                                 if key == name or key.startswith(name + ".")])
        spec.loader.exec_module(module)
        self.assertEqual(len(module.NODE_CLASS_MAPPINGS), 5)
        self.assertEqual(set(module.NODE_CLASS_MAPPINGS), set(module.NODE_DISPLAY_NAME_MAPPINGS))
        self.assertEqual(module.__all__, ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"])
        for key, node in module.NODE_CLASS_MAPPINGS.items():
            self.assertTrue(key.startswith("ArtemKo7v"))
            self.assertEqual(node.CATEGORY, "ArtemKo7v/Krea2 IdentityMod")
            self.assertTrue(node.DESCRIPTION)
            self.assertTrue(callable(getattr(node(), node.FUNCTION)))
            self.assertIn("required", node.INPUT_TYPES())
            self.assertEqual(len(node.RETURN_TYPES), len(node.RETURN_NAMES))
        self.assertTrue(self.nodes.ArtemKo7vKrea2IdentityModSave.OUTPUT_NODE)
