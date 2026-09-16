"""ComfyUI lifecycle, appearance, and cached grounding nodes for IdentityMod."""

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
from .qwen_cache import _add_qwen_vision_cache
from .qwen_injection import _encode_cached_grounded

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
        lines = [
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
            "Appearance cache: yes",
            "Qwen cache: " + ("yes" if identity_mod.qwen_vision_cache is not None else "no"),
            f"Created: {metadata['created_at_utc']}",
        ]
        if identity_mod.qwen_vision_cache is not None:
            lines.extend([
                "", f"Qwen model: {metadata['qwen_model_type']}",
                f"Grounding resolution: {metadata['qwen_grounding_px']}",
                f"Qwen input: {metadata['qwen_input_width']}x{metadata['qwen_input_height']}",
                f"Visual tokens: {metadata['qwen_merged_tokens']}",
                f"Visual width: {metadata['qwen_merged_width']}",
                f"Visual dtype: {metadata['qwen_merged_dtype']}",
                f"Grid: {metadata['qwen_grid_shape']} ({metadata['qwen_grid_dtype']})",
                f"DeepStack tensors: {metadata['qwen_deepstack_count']}",
                f"DeepStack dtype: {metadata['qwen_deepstack_dtype']}",
                f"Qwen cache schema: {metadata['qwen_cache_schema']}",
            ])
        return ("\n".join(lines),)


class ArtemKo7vKrea2IdentityModAddQwenVisionCache:
    """Add prompt-independent Qwen3-VL visual features to an IdentityMod."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = (IDENTITY_MOD_COMFY_TYPE,)
    RETURN_NAMES = ("identity_mod",)
    FUNCTION = "add_qwen_vision_cache"
    DESCRIPTION = (
        "Adds or replaces the Qwen3-VL vision cache for image-free grounded encoding. "
        "Use the same source image as for the appearance cache."
    )

    @classmethod
    def INPUT_TYPES(cls):
        """Grounding resolution is a creation-time parameter only."""
        return {"required": {
            "identity_mod": (IDENTITY_MOD_COMFY_TYPE,),
            "clip": ("CLIP", {"tooltip": "Krea2 Qwen3-VL 4B text encoder."}),
            "image": ("IMAGE", {"tooltip": "The same single reference used for appearance."}),
            "grounding_px": ("INT", {
                "default": 768, "min": 0, "max": 4096, "step": 64,
                "tooltip": "Maximum input side; 0 keeps native size. Does not upscale.",
            }),
        }}

    def add_qwen_vision_cache(
        self, identity_mod: ArtemKo7vKrea2IdentityModData, clip, image: torch.Tensor,
        grounding_px: int = 768,
    ) -> tuple[ArtemKo7vKrea2IdentityModData]:
        """Return a new full-cache object, leaving the input unchanged."""
        result = _add_qwen_vision_cache(identity_mod, clip, image, grounding_px)
        logger.info("%s Added Qwen vision cache: %s tokens, grounding_px=%d.", LOG_PREFIX,
                    result.metadata["qwen_merged_tokens"], grounding_px)
        return (result,)


class ArtemKo7vKrea2IdentityModGroundedEncode:
    """Encode a current Krea2 edit prompt using cached Qwen3-VL visual features."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    DESCRIPTION = (
        "Uses the IdentityMod vision cache instead of a source image; runs the current "
        "prompt through the stock Qwen language model. Empty negative prompts are supported."
    )

    @classmethod
    def INPUT_TYPES(cls):
        """Declare image-free positive or negative encoding inputs."""
        return {
            "required": {
                "clip": ("CLIP",), "identity_mod": (IDENTITY_MOD_COMFY_TYPE,),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {"system_prompt": ("STRING", {"default": "", "multiline": True})},
        }

    def encode(
        self, clip, identity_mod: ArtemKo7vKrea2IdentityModData, prompt: str,
        system_prompt: str = "",
    ) -> tuple[list]:
        """Return stock scheduled conditioning without invoking the vision tower."""
        return (_encode_cached_grounded(clip, identity_mod, prompt, system_prompt),)


NODE_CLASS_MAPPINGS = {
    "ArtemKo7vKrea2IdentityModCreate": ArtemKo7vKrea2IdentityModCreate,
    "ArtemKo7vKrea2IdentityModSave": ArtemKo7vKrea2IdentityModSave,
    "ArtemKo7vKrea2IdentityModLoad": ArtemKo7vKrea2IdentityModLoad,
    "ArtemKo7vKrea2IdentityModToLatent": ArtemKo7vKrea2IdentityModToLatent,
    "ArtemKo7vKrea2IdentityModInfo": ArtemKo7vKrea2IdentityModInfo,
    "ArtemKo7vKrea2IdentityModAddQwenVisionCache": ArtemKo7vKrea2IdentityModAddQwenVisionCache,
    "ArtemKo7vKrea2IdentityModGroundedEncode": ArtemKo7vKrea2IdentityModGroundedEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArtemKo7vKrea2IdentityModCreate": "Krea2 IdentityMod Create",
    "ArtemKo7vKrea2IdentityModSave": "Krea2 IdentityMod Save",
    "ArtemKo7vKrea2IdentityModLoad": "Krea2 IdentityMod Load",
    "ArtemKo7vKrea2IdentityModToLatent": "Krea2 IdentityMod To Latent",
    "ArtemKo7vKrea2IdentityModInfo": "Krea2 IdentityMod Info",
    "ArtemKo7vKrea2IdentityModAddQwenVisionCache": "Krea2 IdentityMod Add Qwen Vision Cache",
    "ArtemKo7vKrea2IdentityModGroundedEncode": "Krea2 IdentityMod Grounded Encode",
}
