"""Small deterministic fixtures; core tests do not import ComfyUI."""

import sys
import types
from pathlib import Path
from unittest.mock import patch

import torch

# Give the repository a legal package name without executing its ComfyUI entry point.
PACKAGE_NAME = "artemko7v_identitymod_test"
ROOT = Path(__file__).resolve().parents[1]
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = package

from artemko7v_identitymod_test.constants import FIXED_METADATA
from artemko7v_identitymod_test.identity_mod import ArtemKo7vKrea2IdentityModData


def make_mod(dtype=torch.float32):
    samples = torch.randn(
        1, 2, 8, 8, generator=torch.Generator().manual_seed(17), dtype=torch.float64
    ).to(dtype)
    metadata = {
        **FIXED_METADATA,
        "format_version": "0.1.0",
        "identity_name": "Alice",
        "description": "Unit-test reference",
        "target_width": "64", "target_height": "64",
        "target_latent_width": "8", "target_latent_height": "8",
        "latent_channels": "2", "latent_dtype": str(dtype),
        "source_width": "128", "source_height": "128",
        "created_at_utc": "2026-09-15T00:00:00Z",
    }
    return ArtemKo7vKrea2IdentityModData(samples, metadata)


class ArtemKo7vFakeVAE:
    """Record input and emit deterministic two-channel spatially downsampled data."""

    def __init__(self):
        self.calls = 0
        self.image = None

    def encode(self, image):
        """Record the exact image reaching the appearance VAE."""
        self.calls += 1
        self.image = image.clone()
        return image[:, ::8, ::8, :2].movedim(-1, 1).contiguous()


def install_folder_paths_stub(test_case, models_directory):
    """Install an isolated ComfyUI folder registry for one test's temporary directory."""
    stub = types.ModuleType("folder_paths")
    stub.models_dir = str(models_directory)
    stub.folder_names_and_paths = {}

    def register(name, path, is_default=False):
        paths, extensions = stub.folder_names_and_paths.setdefault(name, ([], set()))
        if path in paths:
            paths.remove(path)
        paths.insert(0, path) if is_default else paths.append(path)

    def list_files(name):
        return sorted({
            str(path.relative_to(root))
            for root in stub.get_folder_paths(name)
            for path in Path(root).rglob("*") if path.is_file()
        })

    stub.add_model_folder_path = register
    stub.get_folder_paths = lambda name: stub.folder_names_and_paths[name][0][:]
    stub.get_filename_list = list_files
    modules = patch.dict(sys.modules, {"folder_paths": stub})
    modules.start()
    test_case.addCleanup(modules.stop)
    from artemko7v_identitymod_test import paths
    binding = patch.object(paths, "folder_paths", stub)
    binding.start()
    test_case.addCleanup(binding.stop)
    paths._register_model_directory()
    return stub
