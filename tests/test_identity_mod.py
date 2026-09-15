"""Data ownership and strict runtime geometry checks."""

import unittest
from dataclasses import FrozenInstanceError

import torch

from tests.support import make_mod
from artemko7v_identitymod_test.identity_mod import (
    ArtemKo7vKrea2IdentityModData, ArtemKo7vKrea2IdentityModError, _validate_target_match,
)


class ArtemKo7vIdentityModTests(unittest.TestCase):
    """Ensure cached data retains its original values and target geometry."""

    def test_data_owns_tensor_and_metadata(self):
        original = make_mod()
        metadata = dict(original.metadata)
        tensor = original.reference_latent.clone().requires_grad_(True)
        mod = ArtemKo7vKrea2IdentityModData(tensor, metadata)
        tensor.detach().zero_()
        metadata["identity_name"] = "Changed"
        self.assertTrue(torch.equal(mod.reference_latent, original.reference_latent))
        self.assertEqual(mod.identity_name, "Alice")
        self.assertFalse(mod.reference_latent.requires_grad)
        with self.assertRaises(TypeError):
            mod.metadata["identity_name"] = "Changed"
        with self.assertRaises(FrozenInstanceError):
            mod.reference_latent = tensor

    def test_noncontiguous_input(self):
        original = make_mod()
        tensor = original.reference_latent.transpose(2, 3)
        mod = ArtemKo7vKrea2IdentityModData(tensor, original.metadata)
        self.assertTrue(mod.reference_latent.is_contiguous())
        self.assertTrue(torch.equal(mod.reference_latent, tensor))

    def test_exact_target_required(self):
        mod = make_mod()
        _validate_target_match(mod, {"samples": torch.zeros(4, 2, 1, 8, 8)})
        for target in (torch.zeros(1, 2, 8, 6), torch.zeros(1, 4, 8, 8)):
            with self.assertRaises(ArtemKo7vKrea2IdentityModError):
                _validate_target_match(mod, {"samples": target})

    def test_1024_target_cannot_be_used_for_768(self):
        original = make_mod()
        metadata = dict(original.metadata, target_width="1024", target_height="1024",
                        target_latent_width="128", target_latent_height="128")
        mod = ArtemKo7vKrea2IdentityModData(torch.zeros(1, 2, 128, 128), metadata)
        _validate_target_match(mod, {"samples": torch.zeros(1, 2, 128, 128)})
        with self.assertRaisesRegex(ArtemKo7vKrea2IdentityModError, "768x1024"):
            _validate_target_match(mod, {"samples": torch.zeros(1, 2, 128, 96)})
