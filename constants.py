"""Shared constants for the target-specific IdentityMod PoC format."""

IDENTITY_MOD_FORMAT = "krea2_identitymod"
IDENTITY_MOD_FORMAT_VERSION = "0.1.0"
IDENTITY_MOD_CREATOR = "ComfyUI-Krea2IdentityMod"
IDENTITY_MOD_TENSOR_KEY = "appearance_latent"
IDENTITY_MOD_FOLDER_NAME = "krea2_identitymods"
IDENTITY_MOD_COMFY_TYPE = "ARTEMKO7V_KREA2_IDENTITY_MOD"
NODE_CATEGORY = "ArtemKo7v/Krea2 IdentityMod"
LOG_PREFIX = "[ArtemKo7v Krea2IdentityMod]"
KREA2_VAE_DOWNSCALE_FACTOR = 8
KREA2_LATENT_PATCH_SIZE = 2
KREA2_NEAR_MATCH_CROP_TOLERANCE = 0.08

FIXED_METADATA = {
    "format": IDENTITY_MOD_FORMAT,
    "creator": IDENTITY_MOD_CREATOR,
    "model_family": "krea2",
    "purpose": "appearance_reference",
    "preprocess_mode": "full_target_grid_near_matched_fit",
    "vae_downscale_factor": str(KREA2_VAE_DOWNSCALE_FACTOR),
    "qwen_cache_present": "false",
}
DIMENSION_METADATA_KEYS = (
    "target_width", "target_height", "target_latent_width", "target_latent_height",
    "latent_channels", "source_width", "source_height",
)
REQUIRED_METADATA_KEYS = frozenset(FIXED_METADATA) | frozenset(DIMENSION_METADATA_KEYS) | {
    "format_version", "identity_name", "description", "latent_dtype", "created_at_utc",
}
