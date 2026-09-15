"""Register the Krea2 IdentityMod node pack with ComfyUI."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .paths import _register_model_directory

_register_model_directory()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
