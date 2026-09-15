"""ComfyUI lifecycle and diagnostic nodes for Krea2 IdentityMod PoC v0.1."""

import logging

import torch

from .constants import IDENTITY_MOD_COMFY_TYPE, LOG_PREFIX, NODE_CATEGORY
from .identity_mod import (
    ArtemKo7vKrea2IdentityModData,
    _require_identity_mod,
    _validate_target_match,
)
from .paths import _list_identity_mod_files, _resolve_identity_mod_file, _save_identity_mod
from .preprocessing import _create_identity_mod
from .serialization import _load_identity_mod

logger = logging.getLogger(__name__)


class ArtemKo7vKrea2IdentityModCreate:
    """Create one target-specific appearance reference with one VAE encode."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = (IDENTITY_MOD_COMFY_TYPE,)
    RETURN_NAMES = ("identity_mod",)
    FUNCTION = "create"
    DESCRIPTION = (
        "Creates a target-resolution-specific Krea2 IdentityMod by preprocessing a reference "
        "image and caching its raw VAE appearance latent."
    )

    @classmethod
    def INPUT_TYPES(cls):
        """Declare creation inputs and explain the PoC geometry boundary."""
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "One RGB reference with near-matched target aspect ratio."
                }),
                "vae": ("VAE", {
                    "tooltip": "Krea2-compatible VAE. Encodes the reference exactly once."
                }),
                "target_latent": ("LATENT", {
                    "tooltip": "Generation target geometry. Pixel H/W must be multiples of 16."
                }),
                "identity_name": ("STRING", {
                    "default": "Identity", "tooltip": "Descriptive name; does not affect inference."
                }),
            },
            "optional": {
                "description": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Optional metadata only; does not affect inference."
                }),
            },
        }

    def create(
        self, image: torch.Tensor, vae, target_latent: dict, identity_name: str,
        description: str = "",
    ) -> tuple[ArtemKo7vKrea2IdentityModData]:
        """Validate and encode a reference, leaving model scaling to Krea2EditModelPatch."""
        identity_mod = _create_identity_mod(image, vae, target_latent, identity_name, description)
        logger.info(
            "%s Created IdentityMod %r for %dx%d, latent shape=%s.", LOG_PREFIX,
            identity_mod.identity_name, identity_mod.target_width, identity_mod.target_height,
            tuple(identity_mod.reference_latent.shape),
        )
        return (identity_mod,)


class ArtemKo7vKrea2IdentityModSave:
    """Save portable references within the default models/krea2_identitymods directory."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Saves a Krea2 IdentityMod as a portable safetensors file under models/krea2_identitymods."
    )

    @classmethod
    def INPUT_TYPES(cls):
        """Declare a relative filename and explicit overwrite control."""
        return {
            "required": {
                "identity_mod": (IDENTITY_MOD_COMFY_TYPE, {
                    "tooltip": "IdentityMod to save losslessly."
                }),
                "filename": ("STRING", {
                    "default": "identity_mod",
                    "tooltip": "Relative name, e.g. alice/square_1024. Extension is added for you."
                }),
            },
            "optional": {
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Replace the file. When disabled, use an incremented free filename."
                }),
            },
        }

    def save(
        self, identity_mod: ArtemKo7vKrea2IdentityModData, filename: str, overwrite: bool = False,
    ) -> tuple[str]:
        """Return the relative filename actually published to disk."""
        saved = _save_identity_mod(identity_mod, filename, overwrite)
        logger.info("%s Saved IdentityMod to %r.", LOG_PREFIX, saved)
        return (saved,)


class ArtemKo7vKrea2IdentityModLoad:
    """Load validated CPU data and invalidate ComfyUI's cache when its file changes."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = (IDENTITY_MOD_COMFY_TYPE,)
    RETURN_NAMES = ("identity_mod",)
    FUNCTION = "load"
    DESCRIPTION = "Loads and validates a Krea2 IdentityMod safetensors file."

    @classmethod
    def INPUT_TYPES(cls):
        """List safetensors recursively from ComfyUI's registered IdentityMod folders."""
        return {"required": {"identity_mod_file": (_list_identity_mod_files(), {
            "tooltip": "Select a target-specific IdentityMod from models/krea2_identitymods."
        })}}

    @classmethod
    def IS_CHANGED(cls, identity_mod_file: str) -> tuple[str, int, int, int]:
        """Track replacement, modification time, and size without loading a tensor."""
        path = _resolve_identity_mod_file(identity_mod_file)
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)

    def load(self, identity_mod_file: str) -> tuple[ArtemKo7vKrea2IdentityModData]:
        """Return a CPU-resident, schema-validated IdentityMod."""
        identity_mod = _load_identity_mod(_resolve_identity_mod_file(identity_mod_file))
        logger.info("%s Loaded IdentityMod from %r.", LOG_PREFIX, identity_mod_file)
        return (identity_mod,)


class ArtemKo7vKrea2IdentityModToLatent:
    """Expose a raw reference only after checking the actual generation target."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("source_latent",)
    FUNCTION = "to_latent"
    DESCRIPTION = (
        "Validates an IdentityMod against the current target resolution and exposes its cached "
        "appearance latent as a Krea2-compatible ComfyUI LATENT with a singleton frame axis."
    )

    @classmethod
    def INPUT_TYPES(cls):
        """Require the same target latent that feeds the sampler."""
        return {"required": {
            "identity_mod": (IDENTITY_MOD_COMFY_TYPE, {
                "tooltip": "Loaded or newly created IdentityMod."
            }),
            "target_latent": ("LATENT", {
                "tooltip": "Use KSampler's target latent. Geometry must match exactly."
            }),
        }}

    def to_latent(
        self, identity_mod: ArtemKo7vKrea2IdentityModData, target_latent: dict,
    ) -> tuple[dict[str, torch.Tensor]]:
        """Restore T=1 for Krea2 normalization and protect the cached tensor's storage."""
        _validate_target_match(identity_mod, target_latent)
        # Krea2 uses Wan21's B,C,T,H,W channel normalization. A 4D source broadcasts
        # to B,C,C,H,W there instead of preserving a single frame. Disk storage stays 4D.
        return ({"samples": identity_mod.reference_latent.unsqueeze(2).clone()},)


class ArtemKo7vKrea2IdentityModInfo:
    """Return diagnostic metadata without changing inference behavior."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    DESCRIPTION = "Displays identity metadata, target geometry, and raw appearance latent details."

    @classmethod
    def INPUT_TYPES(cls):
        """Declare the IdentityMod to inspect."""
        return {"required": {"identity_mod": (IDENTITY_MOD_COMFY_TYPE, {
            "tooltip": "IdentityMod whose metadata should be summarized."
        })}}

    def get_info(self, identity_mod: ArtemKo7vKrea2IdentityModData) -> tuple[str]:
        """Return an English summary suitable for any ComfyUI string display node."""
        _require_identity_mod(identity_mod)
        metadata = identity_mod.metadata
        return ("\n".join((
            "Krea2 IdentityMod", "",
            f"Name: {metadata['identity_name']}",
            f"Description: {metadata['description']}",
            f"Format: {metadata['format_version']}",
            f"Purpose: {metadata['purpose']}",
            f"Source: {metadata['source_width']}x{metadata['source_height']}",
            f"Target: {metadata['target_width']}x{metadata['target_height']}",
            f"Latent: {metadata['latent_channels']}x{metadata['target_latent_height']}"
            f"x{metadata['target_latent_width']}",
            f"Dtype: {metadata['latent_dtype']}",
            f"Preprocess: {metadata['preprocess_mode']}",
            "Qwen cache: no",
            f"Created: {metadata['created_at_utc']}",
        )),)


NODE_CLASS_MAPPINGS = {
    "ArtemKo7vKrea2IdentityModCreate": ArtemKo7vKrea2IdentityModCreate,
    "ArtemKo7vKrea2IdentityModSave": ArtemKo7vKrea2IdentityModSave,
    "ArtemKo7vKrea2IdentityModLoad": ArtemKo7vKrea2IdentityModLoad,
    "ArtemKo7vKrea2IdentityModToLatent": ArtemKo7vKrea2IdentityModToLatent,
    "ArtemKo7vKrea2IdentityModInfo": ArtemKo7vKrea2IdentityModInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArtemKo7vKrea2IdentityModCreate": "Krea2 IdentityMod Create",
    "ArtemKo7vKrea2IdentityModSave": "Krea2 IdentityMod Save",
    "ArtemKo7vKrea2IdentityModLoad": "Krea2 IdentityMod Load",
    "ArtemKo7vKrea2IdentityModToLatent": "Krea2 IdentityMod To Latent",
    "ArtemKo7vKrea2IdentityModInfo": "Krea2 IdentityMod Info",
}
