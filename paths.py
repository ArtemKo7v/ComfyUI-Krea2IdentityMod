"""ComfyUI directory registration, discovery, and contained atomic file writes."""

import os
import re
import tempfile
from pathlib import Path, PureWindowsPath

import folder_paths

from .constants import IDENTITY_MOD_FOLDER_NAME
from .identity_mod import ArtemKo7vKrea2IdentityModData, ArtemKo7vKrea2IdentityModError
from .serialization import _serialize_identity_mod


def _default_directory() -> Path:
    """Save only in the default IdentityMod directory, even with extra search paths."""
    return Path(folder_paths.models_dir) / IDENTITY_MOD_FOLDER_NAME


def _register_model_directory() -> None:
    """Register the default directory without writing anything during package import."""
    folder_paths.add_model_folder_path(
        IDENTITY_MOD_FOLDER_NAME, str(_default_directory()), is_default=True
    )


def _relative_filename(filename: str, *, append_extension: bool = False) -> Path:
    """Validate portable relative names, including Windows path and device-name hazards."""
    if not isinstance(filename, str) or not filename.strip():
        raise ArtemKo7vKrea2IdentityModError("IdentityMod filename must not be empty.")
    normalized = filename.replace("\\", "/")
    if PureWindowsPath(filename).drive or normalized.startswith("/"):
        raise ArtemKo7vKrea2IdentityModError("IdentityMod filename must be a relative path.")
    parts = normalized.split("/")
    for part in parts:
        if (part in ("", ".", "..") or part.endswith((" ", "."))
                or re.search(r'[<>:"|?*\x00-\x1f]', part)
                or re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)):
            raise ArtemKo7vKrea2IdentityModError(
                "Unsafe IdentityMod filename. Use relative subfolders without traversal, "
                "reserved characters, or device names."
            )
    if not normalized.lower().endswith(".safetensors"):
        if not append_extension:
            raise ArtemKo7vKrea2IdentityModError("IdentityMod files must use .safetensors.")
        normalized += ".safetensors"
    return Path(normalized)


def _contained_path(directory: Path, relative: Path) -> Path:
    """Resolve symlinks and junctions before checking containment with commonpath."""
    try:
        root = directory.resolve()
        target = (root / relative).resolve()
        root_key = _comparison_path(root)
        target_key = _comparison_path(target)
        if os.path.commonpath((root_key, target_key)) != root_key or target_key == root_key:
            raise ValueError("outside directory")
    except (ValueError, OSError, RuntimeError) as error:
        raise ArtemKo7vKrea2IdentityModError(
            "IdentityMod path escapes the registered storage directory or cannot be resolved: "
            f"root={str(directory)!r}, relative={str(relative)!r}."
        ) from error
    return target


def _comparison_path(path: Path) -> str:
    """Compare Windows extended and ordinary resolved paths using the same spelling.

    Python can return the extended prefix when a parent directory appears during
    concurrent resolution. Keep the original resolved path for actual file access.
    """
    value = str(path)
    if os.name == "nt":
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
    return os.path.normcase(value)


def _list_identity_mod_files() -> list[str]:
    """Use ComfyUI's recursive model discovery and explicitly filter extensions."""
    return sorted(
        name for name in folder_paths.get_filename_list(IDENTITY_MOD_FOLDER_NAME)
        if name.lower().endswith(".safetensors")
    )


def _resolve_identity_mod_file(filename: str) -> Path:
    """Resolve relative files across registered roots, rejecting escaping links."""
    relative = _relative_filename(filename)
    for directory in folder_paths.get_folder_paths(IDENTITY_MOD_FOLDER_NAME):
        candidate = _contained_path(Path(directory), relative)
        if candidate.is_file():
            return candidate
    raise ArtemKo7vKrea2IdentityModError(f"IdentityMod file not found: {filename!r}.")


def _save_identity_mod(
    identity_mod: ArtemKo7vKrea2IdentityModData, filename: str, overwrite: bool = False
) -> str:
    """Publish complete files atomically; concurrent non-overwrite saves get unique names."""
    relative = _relative_filename(filename, append_extension=True)
    root = _default_directory()
    destination = _contained_path(root, relative)
    payload = _serialize_identity_mod(identity_mod)
    temporary = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = _contained_path(root, relative)
        # The temporary file lives on the destination filesystem for atomic publication.
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".identitymod-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, _contained_path(root, relative))
        else:
            index = 0
            while True:
                candidate_relative = relative if index == 0 else relative.with_name(
                    f"{relative.stem}_{index:03d}{relative.suffix}"
                )
                candidate = _contained_path(root, candidate_relative)
                try:
                    # Hard-link creation is atomic and refuses to replace an existing file.
                    os.link(temporary, candidate)
                    relative = candidate_relative
                    break
                except FileExistsError:
                    index += 1
        return relative.as_posix()
    except OSError as error:
        raise ArtemKo7vKrea2IdentityModError(
            f"Unable to save IdentityMod {filename!r}: {error}. "
            "Check directory permissions and filesystem support for atomic hard links."
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
