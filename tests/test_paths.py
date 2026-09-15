"""Filesystem containment, collision, and failed-write tests."""

import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from tests.support import install_folder_paths_stub, make_mod


class ArtemKo7vPathTests(unittest.TestCase):
    """Exercise storage with real files inside temporary directories."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.stub = install_folder_paths_stub(self, self.root / "models")
        from artemko7v_identitymod_test import paths
        self.paths = paths
        self.mod = make_mod()

    def test_registration_is_idempotent_and_has_no_disk_side_effects(self):
        self.paths._register_model_directory()
        self.assertEqual(len(self.stub.get_folder_paths("krea2_identitymods")), 1)
        self.assertFalse((self.root / "models").exists())

    def test_extension_increment_and_overwrite(self):
        for filename, expected in (
            ("alice/square", "alice/square.safetensors"),
            ("alice/square.safetensors", "alice/square_001.safetensors"),
            ("alice/square", "alice/square_002.safetensors"),
            ("upper.SAFETENSORS", "upper.SAFETENSORS"),
        ):
            self.assertEqual(self.paths._save_identity_mod(self.mod, filename), expected)
        from artemko7v_identitymod_test.identity_mod import ArtemKo7vKrea2IdentityModData
        from artemko7v_identitymod_test.serialization import _load_identity_mod
        replacement = ArtemKo7vKrea2IdentityModData(
            self.mod.reference_latent, dict(self.mod.metadata, identity_name="Replacement")
        )
        self.assertEqual(self.paths._save_identity_mod(replacement, "alice/square", True),
                         "alice/square.safetensors")
        loaded = _load_identity_mod(
            self.paths._resolve_identity_mod_file("alice/square.safetensors")
        )
        self.assertEqual(loaded.identity_name, "Replacement")
        self.assertEqual(len(self.paths._list_identity_mod_files()), 4)

    def test_traversal_and_nonportable_names(self):
        from artemko7v_identitymod_test.identity_mod import ArtemKo7vKrea2IdentityModError
        for name in (
            "../../evil", "..\\evil", "/absolute/file", "C:\\outside\\foo", "C:foo",
            "\\\\server\\share\\foo", "a/../evil", "a//b", "a/./b", "", " ",
            "NUL", "con.txt", "a/file:stream", "foo.", "foo ", "a\x00b", "a/",
        ):
            with self.subTest(name=name):
                with self.assertRaises(ArtemKo7vKrea2IdentityModError):
                    self.paths._save_identity_mod(self.mod, name)
                with self.assertRaises(ArtemKo7vKrea2IdentityModError):
                    self.paths._resolve_identity_mod_file(name)
        self.assertFalse((self.root / "models").exists())

    def test_load_filters_extensions_and_uses_extra_roots(self):
        self.paths._save_identity_mod(self.mod, "nested/reference")
        default = self.paths._default_directory()
        (default / "ignored.pt").write_bytes(b"not an IdentityMod")
        extra = self.root / "extra"
        extra.mkdir()
        (extra / "external.safetensors").write_bytes(b"fixture")
        self.stub.add_model_folder_path("krea2_identitymods", str(extra))
        files = [name.replace("\\", "/") for name in self.paths._list_identity_mod_files()]
        self.assertEqual(files, ["external.safetensors", "nested/reference.safetensors"])
        self.assertEqual(self.paths._resolve_identity_mod_file("external.safetensors"),
                         extra / "external.safetensors")

    def test_symlink_escape_rejected(self):
        from artemko7v_identitymod_test.identity_mod import ArtemKo7vKrea2IdentityModError
        root = self.paths._default_directory()
        root.mkdir(parents=True)
        outside = self.root / "outside"
        outside.mkdir()
        link = root / "escape"
        try:
            if os.name == "nt":
                import _winapi
                _winapi.CreateJunction(str(outside), str(link))
            else:
                link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Directory links unavailable: {error}")
        with self.assertRaises(ArtemKo7vKrea2IdentityModError):
            self.paths._save_identity_mod(self.mod, "escape/evil")
        with self.assertRaises(ArtemKo7vKrea2IdentityModError):
            self.paths._resolve_identity_mod_file("escape/evil.safetensors")
        self.assertEqual(list(outside.iterdir()), [])

    def test_atomic_non_overwrite_under_concurrent_saves(self):
        from artemko7v_identitymod_test.serialization import _load_identity_mod
        with ThreadPoolExecutor(max_workers=4) as executor:
            filenames = list(executor.map(
                lambda _: self.paths._save_identity_mod(self.mod, "same"), range(8)
            ))
        self.assertEqual(len(set(filenames)), 8)
        for name in filenames:
            _load_identity_mod(self.paths._resolve_identity_mod_file(name))
        self.assertFalse(list(self.paths._default_directory().glob("*.tmp")))

    def test_failed_replace_keeps_previous_file_and_cleans_temporary(self):
        self.paths._save_identity_mod(self.mod, "previous")
        previous = self.paths._resolve_identity_mod_file("previous.safetensors").read_bytes()
        from artemko7v_identitymod_test.identity_mod import ArtemKo7vKrea2IdentityModError
        with patch.object(self.paths.os, "replace", side_effect=OSError("simulated failure")):
            with self.assertRaises(ArtemKo7vKrea2IdentityModError):
                self.paths._save_identity_mod(self.mod, "previous", True)
        self.assertEqual(self.paths._resolve_identity_mod_file("previous.safetensors").read_bytes(),
                         previous)
        self.assertFalse(list(self.paths._default_directory().glob("*.tmp")))
