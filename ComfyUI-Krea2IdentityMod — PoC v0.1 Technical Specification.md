# ComfyUI-Krea2IdentityMod — PoC v0.1

## 1. Project overview

Create a new ComfyUI custom-node repository:

`ComfyUI-Krea2IdentityMod`

The purpose of the repository is to implement a Proof of Concept for a portable **Krea2 IdentityMod** file.

The PoC must test whether the appearance-conditioning part of Krea 2 Identity Edit can be precomputed once as a VAE latent, saved to a `.safetensors` file, loaded later, and reused through the existing Krea2 Identity Edit inference path without re-encoding the source image through the VAE.

The PoC is specifically focused on the **appearance / VAE reference path**.

It must **not** attempt to solve Qwen3-VL visual-feature caching yet.

The expected conceptual flow is:

```text
Creation time:

Reference IMAGE
    ↓
Target-resolution-compatible image preprocessing
    ↓
Krea2 / Qwen Image VAE
    ↓
Raw source VAE latent
    ↓
Krea2 IdentityMod
    ↓
.safetensors
```

Runtime:

```text
identity.safetensors
    ↓
Load Krea2 IdentityMod
    ↓
Validate against current target latent
    ↓
Standard ComfyUI LATENT
    ↓
Krea2EditModelPatch.source_latent
    ↓
Krea2 Identity Edit inference
```

The original reference image is still used separately by `Krea2EditGroundedEncode` in this PoC:

```text
Reference IMAGE
    ↓
Krea2EditGroundedEncode
    ↓
Qwen3-VL semantic conditioning
```

Therefore PoC v0.1 is **not yet a completely image-free IdentityMod runtime**.

The PoC only removes the repeated VAE appearance encoding.

---

# 2. Main hypothesis to validate

The PoC must answer the following technical question:

> Can the exact source VAE latent used by Krea2 Identity Edit be generated once, serialized losslessly, loaded later, and supplied as `source_latent` while preserving the resulting Krea2 Identity Edit output?

Success means:

1. The source latent can be created once.
2. Saving/loading through `.safetensors` does not alter it.
3. The loaded latent can be converted back to a normal ComfyUI `LATENT`.
4. Existing `Krea2EditModelPatch` can consume it.
5. Runtime VAE encoding of the reference image is no longer necessary for the appearance path.
6. For supported target geometry, output should be identical or numerically/perceptually indistinguishable from the baseline pixel-path workflow.
7. No changes to Krea2 weights or Identity Edit LoRA weights are required.

---

# 3. Important architectural decision for PoC v0.1

Do **not** implement a replacement for `Krea2EditModelPatch`.

Do **not** duplicate the Krea2 transformer forward pass.

Do **not** patch the Krea2 DiT directly in this repository.

The PoC must reuse the existing:

```text
Krea2EditModelPatch
Krea2EditGroundedEncode
```

from `ComfyUI-Krea2Edit`.

The IdentityMod repository is responsible only for:

```text
IMAGE
  ↓
preprocess
  ↓
VAE
  ↓
portable latent representation
  ↓
save/load
  ↓
validated LATENT
```

This keeps the experiment isolated.

If the experiment succeeds, a future version may implement its own native IdentityMod-aware Krea2 patch to support arbitrary reference geometries.

---

# 4. Why target resolution must be fixed in PoC v0.1

Krea2 Identity Edit v1.2 has special reference geometry handling for aspect-ratio mismatches.

For a genuine mismatched aspect ratio, the reference VAE latent may occupy a smaller spatial grid than the target and is positioned in the target coordinate system using specific RoPE positions.

A generic cached latent therefore cannot automatically be reused at every resolution without reproducing that geometry logic.

PoC v0.1 intentionally avoids that complexity.

Each IdentityMod is therefore **target-resolution-specific**.

Example:

```text
alice_1024x1024.safetensors
```

is valid for:

```text
1024 × 1024
```

but must not silently be used for:

```text
1344 × 768
```

The runtime validation node must reject such a mismatch.

The PoC should prove the caching concept first.

Arbitrary aspect ratio support is a future phase.

---

# 5. Supported geometry in PoC v0.1

IdentityMod creation must only support source images that fall into the same "near-matched aspect ratio" branch as the current Krea2 Identity Edit `fit` implementation.

Use the same conceptual threshold:

```python
CROP_TOL = 0.08
```

For source dimensions:

```text
source_height = ih
source_width = iw
```

and target dimensions:

```text
target_height = th
target_width = tw
```

calculate:

```python
scale_fit = min(th / ih, tw / iw)
```

The source is considered compatible only when:

```python
ih * scale_fit >= th * (1.0 - CROP_TOL)
and
iw * scale_fit >= tw * (1.0 - CROP_TOL)
```

If this condition is false, creation must fail with an explicit error.

Do not fall back to stretching.

Do not silently create a different geometry.

Do not resize latents.

The error must clearly explain that arbitrary aspect-ratio `fit` geometry is outside PoC v0.1.

Example error:

```text
Krea2 IdentityMod PoC does not support this source/target aspect-ratio combination.

Source: 1024x1024
Target: 1344x768

PoC v0.1 only supports near-matched aspect ratios that can be converted to the full target grid before VAE encoding.

Prepare the source to the target aspect ratio or create a separate IdentityMod for another target geometry.
```

All error text must be English.

---

# 6. Reference preprocessing algorithm

For a supported source image, preprocess it into the exact target pixel grid before VAE encoding.

Use the target spatial size derived from `target_latent`.

Current Krea2 image latents use an 8× pixel-to-latent scale for this path.

Define:

```python
KREA2_VAE_DOWNSCALE_FACTOR = 8
```

Given:

```python
target_samples = target_latent["samples"]
target_latent_height = target_samples.shape[-2]
target_latent_width = target_samples.shape[-1]
```

calculate:

```python
target_height = target_latent_height * 8
target_width = target_latent_width * 8
```

For the supported near-matched geometry, reproduce the crop-to-target behavior:

```python
scale_fill = max(
    target_height / source_height,
    target_width / source_width,
)

crop_height = min(
    source_height,
    int(round(target_height / scale_fill)),
)

crop_width = min(
    source_width,
    int(round(target_width / scale_fill)),
)
```

Center-crop:

```python
y0 = (source_height - crop_height) // 2
x0 = (source_width - crop_width) // 2
```

Then resize the cropped IMAGE to exactly:

```text
target_width × target_height
```

using:

```python
torch.nn.functional.interpolate(
    ...,
    mode="bicubic",
    antialias=True,
)
```

The VAE must receive a normal ComfyUI-style image tensor:

```text
B × H × W × C
```

with only RGB channels:

```python
image[..., :3]
```

and values clamped to:

```text
0.0 .. 1.0
```

Then:

```python
reference_latent = vae.encode(prepared_image)
```

Important:

The IdentityMod must store the **raw output of `vae.encode()`**.

Do **not** call:

```python
model.model.process_latent_in(...)
```

during IdentityMod creation.

Latent model scaling must remain the responsibility of the existing `Krea2EditModelPatch` at generation time.

This keeps the IdentityMod representation independent from a particular loaded Krea2 `MODEL` object.

---

# 7. Input validation during creation

The Create node must validate all assumptions before producing an IdentityMod.

Required checks:

### IMAGE

Expected:

```text
torch.Tensor
B × H × W × C
```

Requirements:

```text
batch size == 1
channels >= 3
height > 0
width > 0
```

If image batch size is greater than one, fail.

Do not silently use the first image.

Example:

```text
Krea2 IdentityMod PoC supports exactly one reference image.
Received image batch size: 4.
```

### target_latent

Must be a ComfyUI LATENT-like dictionary containing:

```python
"samples"
```

`samples` must be a tensor.

Support:

```text
B × C × H × W
```

and, if necessary for compatibility:

```text
B × C × T × H × W
```

For PoC image generation:

```text
T must be 1 if a temporal dimension exists.
```

Otherwise fail.

### VAE result

After VAE encoding:

```text
reference_latent.shape[-2:]
```

must exactly equal:

```text
target_latent["samples"].shape[-2:]
```

The latent channel count must also match the target latent channel count.

If not, fail with an error suggesting an incorrect VAE or incompatible model configuration.

Example:

```text
Encoded reference latent does not match the target Krea2 latent geometry.

Reference latent: C=16, H=128, W=128
Target latent:    C=4, H=128, W=128

Make sure the Krea2-compatible VAE and target latent are being used.
```

Do not attempt to fix this automatically.

---

# 8. IdentityMod internal Python representation

Create an immutable or effectively immutable data container.

Preferred name:

```python
@dataclass(frozen=True)
class ArtemKo7vKrea2IdentityModData:
    ...
```

Suggested fields:

```python
reference_latent: torch.Tensor
metadata: dict[str, str]
```

Optional strongly typed convenience properties may be added:

```python
@property
def target_width(self) -> int:
    ...

@property
def target_height(self) -> int:
    ...

@property
def latent_width(self) -> int:
    ...

@property
def latent_height(self) -> int:
    ...

@property
def identity_name(self) -> str:
    ...
```

The tensor stored inside the object should be:

```text
CPU
contiguous
floating point
batch size 1
```

Preserve its original dtype.

Do not convert to FP16 automatically.

Do not quantize.

---

# 9. Custom ComfyUI datatype

Use a unique custom ComfyUI type:

```text
ARTEMKO7V_KREA2_IDENTITY_MOD
```

Do not use a generic name such as:

```text
IDENTITY_MOD
```

because another custom-node project may eventually define the same type.

---

# 10. `.safetensors` format

IdentityMod files must use `.safetensors`.

Do not use:

```text
pickle
.pt
.pth
torch.save
```

The file must contain exactly one required tensor in PoC v0.1:

```text
appearance_latent
```

Example:

```python
{
    "appearance_latent": reference_latent
}
```

Additional tensor keys must not be added unless explicitly required.

---

# 11. Required safetensors metadata

Safetensors metadata values must all be strings.

Required keys:

```text
format
format_version
creator
model_family
identity_name
description
purpose
target_width
target_height
target_latent_width
target_latent_height
latent_channels
latent_dtype
source_width
source_height
preprocess_mode
vae_downscale_factor
qwen_cache_present
created_at_utc
```

Required values:

```text
format = "krea2_identitymod"
format_version = "0.1.0"
creator = "ComfyUI-Krea2IdentityMod"
model_family = "krea2"
purpose = "appearance_reference"
preprocess_mode = "full_target_grid_near_matched_fit"
vae_downscale_factor = "8"
qwen_cache_present = "false"
```

Example metadata:

```python
{
    "format": "krea2_identitymod",
    "format_version": "0.1.0",
    "creator": "ComfyUI-Krea2IdentityMod",
    "model_family": "krea2",
    "identity_name": "Alice",
    "description": "Primary identity reference",
    "purpose": "appearance_reference",
    "target_width": "1024",
    "target_height": "1024",
    "target_latent_width": "128",
    "target_latent_height": "128",
    "latent_channels": "16",
    "latent_dtype": "torch.float32",
    "source_width": "2048",
    "source_height": "2048",
    "preprocess_mode": "full_target_grid_near_matched_fit",
    "vae_downscale_factor": "8",
    "qwen_cache_present": "false",
    "created_at_utc": "2026-09-15T00:00:00Z",
}
```

`description` is metadata only.

It must not influence inference.

`identity_name` is metadata only.

It must not influence inference.

---

# 12. Format validation on load

Loading must validate:

```text
format == "krea2_identitymod"
```

The loader must verify that:

```text
appearance_latent
```

exists.

For PoC v0.1, accepted version:

```text
0.1.x
```

Do not silently accept a future incompatible major/minor format.

A helper function should parse semantic format compatibility.

For example:

```python
def _validate_format_version(version: str) -> None:
    ...
```

Unknown metadata fields should be tolerated.

Missing required metadata fields should produce a clear error.

The latent must be checked for:

```text
floating-point dtype
rank == 4
batch == 1
valid H/W
expected metadata shape
```

The loader must never execute arbitrary serialized Python code.

---

# 13. Model storage directory

Use:

```text
ComfyUI/models/krea2_identitymods/
```

Register the folder with ComfyUI under a unique folder type:

```text
krea2_identitymods
```

Conceptually:

```python
folder_paths.add_model_folder_path(
    "krea2_identitymods",
    default_path,
    is_default=True,
)
```

The loader node must list `.safetensors` recursively.

If ComfyUI returns files with other extensions because the registered folder has no extension filter, filter the loader options explicitly:

```python
filename.lower().endswith(".safetensors")
```

Subdirectories must be supported:

```text
models/
└── krea2_identitymods/
    ├── alice/
    │   ├── square_1024.safetensors
    │   └── portrait_768x1024.safetensors
    └── bob/
        └── square_1024.safetensors
```

---

# 14. Path security

Saving must never allow path traversal outside:

```text
models/krea2_identitymods/
```

Reject:

```text
../../foo
/absolute/path/foo
C:\outside\foo
```

Normalize all paths and verify with `os.path.commonpath()`.

Subfolders inside the IdentityMod directory are allowed.

Example:

```text
alice/square_1024
```

should create:

```text
models/krea2_identitymods/alice/square_1024.safetensors
```

---

# 15. Required ComfyUI nodes

Implement the following five node classes.

Every class name must use the `ArtemKo7v` prefix.

## 15.1 Create node

Class:

```python
class ArtemKo7vKrea2IdentityModCreate:
```

Mapping key:

```text
ArtemKo7vKrea2IdentityModCreate
```

Display name:

```text
Krea2 IdentityMod Create
```

Category:

```text
ArtemKo7v/Krea2 IdentityMod
```

Required inputs:

```text
image          IMAGE
vae            VAE
target_latent  LATENT
identity_name  STRING
```

Optional input:

```text
description    STRING multiline
```

Suggested defaults:

```text
identity_name = "Identity"
description = ""
```

Return:

```text
ARTEMKO7V_KREA2_IDENTITY_MOD
```

Return name:

```text
identity_mod
```

Function:

```text
create
```

Node description:

```text
Creates a target-resolution-specific Krea2 IdentityMod by preprocessing a reference image and caching its raw VAE appearance latent.
```

Behavior:

1. Validate inputs.
2. Determine target resolution.
3. Verify supported aspect ratio.
4. Preprocess image.
5. VAE encode.
6. Validate resulting latent.
7. Move latent to CPU.
8. Construct metadata.
9. Return `ArtemKo7vKrea2IdentityModData`.

The node must perform the VAE encode exactly once per execution.

---

# 16. Save node

Class:

```python
class ArtemKo7vKrea2IdentityModSave:
```

Mapping key:

```text
ArtemKo7vKrea2IdentityModSave
```

Display name:

```text
Krea2 IdentityMod Save
```

Category:

```text
ArtemKo7v/Krea2 IdentityMod
```

Required inputs:

```text
identity_mod  ARTEMKO7V_KREA2_IDENTITY_MOD
filename      STRING
```

Optional:

```text
overwrite     BOOLEAN
```

Defaults:

```text
filename = "identity_mod"
overwrite = False
```

If `.safetensors` is absent, append it automatically.

If the file exists and:

```text
overwrite == False
```

do not silently overwrite it.

Preferred behavior is automatic increment:

```text
alice.safetensors
alice_001.safetensors
alice_002.safetensors
```

Return:

```text
STRING
```

containing the relative saved filename, for example:

```text
alice/alice_1024x1024.safetensors
```

Set:

```python
OUTPUT_NODE = True
```

Description:

```text
Saves a Krea2 IdentityMod as a portable safetensors file under models/krea2_identitymods.
```

---

# 17. Load node

Class:

```python
class ArtemKo7vKrea2IdentityModLoad:
```

Mapping key:

```text
ArtemKo7vKrea2IdentityModLoad
```

Display name:

```text
Krea2 IdentityMod Load
```

Category:

```text
ArtemKo7v/Krea2 IdentityMod
```

Required input:

```text
identity_mod_file
```

The input must be a dropdown populated from:

```text
models/krea2_identitymods/**/*.safetensors
```

Return:

```text
ARTEMKO7V_KREA2_IDENTITY_MOD
```

Return name:

```text
identity_mod
```

Function:

```text
load
```

Description:

```text
Loads and validates a Krea2 IdentityMod safetensors file.
```

Implement cache invalidation through `IS_CHANGED`.

At minimum use file modification time.

Preferred:

```python
@classmethod
def IS_CHANGED(cls, identity_mod_file: str):
    return os.path.getmtime(...)
```

A file modified on disk must be reloaded rather than using a stale ComfyUI node result.

---

# 18. To Latent node

Class:

```python
class ArtemKo7vKrea2IdentityModToLatent:
```

Mapping key:

```text
ArtemKo7vKrea2IdentityModToLatent
```

Display name:

```text
Krea2 IdentityMod To Latent
```

Category:

```text
ArtemKo7v/Krea2 IdentityMod
```

Required inputs:

```text
identity_mod  ARTEMKO7V_KREA2_IDENTITY_MOD
target_latent LATENT
```

Return:

```text
LATENT
```

Return name:

```text
source_latent
```

Function:

```text
to_latent
```

Description:

```text
Validates an IdentityMod against the current target resolution and exposes its cached appearance latent as a standard ComfyUI LATENT.
```

This node is an important safety boundary.

It must compare current target spatial dimensions with the IdentityMod metadata.

Example:

```text
IdentityMod target:
1024x1024

Current target:
1024x1024
```

→ allowed.

Example:

```text
IdentityMod target:
1024x1024

Current target:
1344x768
```

→ fail.

Error:

```text
IdentityMod target geometry mismatch.

IdentityMod was created for: 1024x1024
Current target is:          1344x768

PoC v0.1 IdentityMods are target-resolution-specific.
Create or load an IdentityMod for the current output geometry.
```

Do not resize the latent.

Do not crop the latent.

Do not interpolate the latent.

Return:

```python
{
    "samples": identity_mod.reference_latent
}
```

The tensor must remain the raw VAE latent.

---

# 19. Info node

Class:

```python
class ArtemKo7vKrea2IdentityModInfo:
```

Mapping key:

```text
ArtemKo7vKrea2IdentityModInfo
```

Display name:

```text
Krea2 IdentityMod Info
```

Category:

```text
ArtemKo7v/Krea2 IdentityMod
```

Required input:

```text
identity_mod  ARTEMKO7V_KREA2_IDENTITY_MOD
```

Return:

```text
STRING
```

Function:

```text
get_info
```

Return a human-readable English summary:

```text
Krea2 IdentityMod

Name: Alice
Format: 0.1.0
Purpose: appearance_reference
Source: 2048x2048
Target: 1024x1024
Latent: 16x128x128
Dtype: torch.float32
Preprocess: full_target_grid_near_matched_fit
Qwen cache: no
```

This node is diagnostic only.

---

# 20. Required node registration

`__init__.py` must export exactly structured mappings similar to:

```python
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
```

Do not use unprefixed mapping IDs.

---

# 21. Naming convention

All public Python classes in this repository must begin with:

```text
ArtemKo7v
```

Examples:

```python
ArtemKo7vKrea2IdentityModCreate
ArtemKo7vKrea2IdentityModLoad
ArtemKo7vKrea2IdentityModData
ArtemKo7vKrea2IdentityModError
```

Private helper functions do not need the prefix and should use leading underscores:

```python
_validate_image(...)
_prepare_reference_image(...)
_parse_metadata(...)
_find_safe_output_path(...)
```

Avoid generic public class names such as:

```text
IdentityMod
Loader
Saver
Metadata
```

---

# 22. Language requirements

All Python code documentation must be English.

This includes:

```text
docstrings
comments
logging
exceptions
tooltips
node descriptions
UI text
README
test names
test comments
metadata descriptions
```

No Russian text must exist in the repository.

The code must remain understandable without reading this specification.

Every public class must have an English docstring.

Every non-trivial public method must have an English docstring.

Every helper implementing important geometry or serialization logic must have an English docstring.

Major logical blocks should have short explanatory comments where the intention is not immediately obvious.

Do not over-comment obvious assignments.

Prefer explaining **why**, not just **what**.

Good:

```python
# Store the raw VAE latent rather than model-scaled samples so the file
# remains independent from the currently loaded Krea2 ModelPatcher.
```

Bad:

```python
# Set latent variable.
latent = ...
```

---

# 23. Logging

Use the Python logging module.

Do not use `print()` for ordinary status messages.

Create:

```python
logger = logging.getLogger(__name__)
```

Use a consistent prefix in important messages:

```text
[ArtemKo7v Krea2IdentityMod]
```

Example:

```text
[ArtemKo7v Krea2IdentityMod] Created IdentityMod 'Alice' for 1024x1024, latent shape=(1, 16, 128, 128).
```

Do not spam the console during every sampling step.

There must be no per-step logging because this repository does not patch the diffusion loop.

---

# 24. Suggested repository layout

Use a clear modular structure.

Recommended:

```text
ComfyUI-Krea2IdentityMod/
├── __init__.py
├── nodes.py
├── constants.py
├── identity_mod.py
├── preprocessing.py
├── serialization.py
├── paths.py
├── README.md
├── LICENSE
├── NOTICE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_serialization.py
│   ├── test_identity_mod.py
│   ├── test_paths.py
│   └── test_nodes.py
└── workflows/
    ├── create_identitymod_poc.json
    └── use_identitymod_poc.json
```

Responsibilities:

```text
constants.py
    Format names, version, tensor key, custom type, folder name,
    VAE scale, geometry constants.

identity_mod.py
    ArtemKo7vKrea2IdentityModData and metadata validation.

preprocessing.py
    IMAGE validation and source-to-target preprocessing.

serialization.py
    safetensors save/load and schema validation.

paths.py
    model-directory registration, filename discovery and safe paths.

nodes.py
    ComfyUI node classes only.

__init__.py
    folder registration + node mappings.
```

Do not put the entire project into `__init__.py`.

---

# 25. Runtime dependencies

Avoid introducing unnecessary dependencies.

Expected runtime dependencies are already present in ComfyUI:

```text
torch
safetensors
ComfyUI APIs
```

Do not add:

```text
numpy
opencv
Pillow
scikit-image
transformers
diffusers
```

unless there is a demonstrated requirement.

Image preprocessing should be implemented with PyTorch.

Safetensors serialization should use the official `safetensors` Python package.

`requirements.txt` may be empty or contain only dependencies not guaranteed by ComfyUI.

Do not pin Torch.

---

# 26. Python style

Target:

```text
Python >= 3.10
```

Use:

```text
type hints
dataclasses where appropriate
pathlib.Path where it improves path clarity
small focused functions
explicit validation
descriptive names
```

Preferred maximum line length:

```text
100
```

Avoid:

```text
deep nesting
large 300-line functions
global mutable state
silent exception handling
bare except
magic strings repeated across modules
```

Constants must be centralized.

Example:

```python
IDENTITY_MOD_FORMAT = "krea2_identitymod"
IDENTITY_MOD_FORMAT_VERSION = "0.1.0"
IDENTITY_MOD_TENSOR_KEY = "appearance_latent"
IDENTITY_MOD_FOLDER_NAME = "krea2_identitymods"
IDENTITY_MOD_COMFY_TYPE = "ARTEMKO7V_KREA2_IDENTITY_MOD"

KREA2_VAE_DOWNSCALE_FACTOR = 8
KREA2_NEAR_MATCH_CROP_TOLERANCE = 0.08
```

---

# 27. Safetensors implementation requirements

Before saving:

```python
tensor = tensor.detach().cpu().contiguous()
```

Preserve dtype.

Do not add gradients.

Do not convert dtype.

Use safetensors metadata.

On loading, force CPU.

Do not automatically move the loaded latent to CUDA.

The existing downstream Krea2 model path should decide device placement.

Roundtrip must satisfy:

```python
torch.equal(original_tensor, loaded_tensor)
```

for the exact saved tensor.

This must be covered by a test.

---

# 28. ComfyUI integration workflow — creating a mod

Expected creation workflow:

```text
Load Image
    │
    ├───────────────┐
    │               │
    ▼               │
Krea2 IdentityMod   │
Create              │
    ▲               │
    │               │
VAE Loader          │
    ▲               │
    │               │
target_latent ◄──── EmptySD3LatentImage
    │
    ▼
Krea2 IdentityMod Save
```

The `target_latent` used when creating the IdentityMod must use the exact intended generation resolution.

Example:

```text
1024 × 1024 generation
→ create a 1024 × 1024 IdentityMod
```

---

# 29. ComfyUI integration workflow — using a mod

Expected generation workflow:

```text
Krea2 IdentityMod Load
        │
        ▼
Krea2 IdentityMod To Latent
        ▲
        │
EmptySD3LatentImage
        │
        └────────────────────────────────────┐
                                             │
IdentityMod source_latent                    │
        │                                    │
        ▼                                    │
Krea2EditModelPatch                          │
        ▲                                    │
        │                                    │
Krea2 model + Identity Edit LoRA             │
                                             │
Reference Image                              │
        │                                    │
        ▼                                    │
Krea2EditGroundedEncode                      │
        │                                    │
        ▼                                    ▼
      CONDITIONING                       KSampler
```

Important runtime configuration:

When using the IdentityMod latent:

```text
Krea2EditModelPatch.source_latent
    = IdentityMod To Latent.source_latent
```

Do not connect:

```text
Krea2EditModelPatch.vae
Krea2EditModelPatch.source_image
```

because the pixel path would override the cached latent.

For the cached full-grid latent, set:

```text
fit_mode = "crop (legacy)"
```

in the existing Krea2EditModelPatch.

This avoids the upstream warning that `fit` requires `vae + source_image`.

Because the IdentityMod reference latent already has exactly the target H/W grid, no latent resize is performed.

Keep normal runtime controls such as:

```text
ref_boost
ref_boost_mask
```

on the existing `Krea2EditModelPatch`.

Do not store `ref_boost` inside the IdentityMod.

`ref_boost` is a runtime generation parameter, not identity data.

---

# 30. Semantic/Qwen path in PoC

Do not modify:

```text
Krea2EditGroundedEncode
```

Do not cache:

```text
Qwen3-VL hidden states
Qwen vision embeddings
prompt-conditioned Qwen states
```

in PoC v0.1.

The same reference image must still be supplied to:

```text
Krea2EditGroundedEncode.image
```

The README must state this prominently.

Example:

```text
PoC limitation:
The IdentityMod currently caches only the VAE appearance reference.
The source image is still required by Krea2EditGroundedEncode for Qwen3-VL semantic grounding.
```

This is not a bug.

It is the intentional PoC scope.

---

# 31. No multi-reference support

PoC v0.1 must support exactly one identity/reference.

Do not implement:

```text
source_latent_b
multiple identities
scene + subject
reference stacking
IdentityMod lists
IdentityMod merging
```

These are future features.

Trying to solve multi-reference in this PoC would make validation significantly harder.

---

# 32. No strength implementation

Do not add an IdentityMod-specific:

```text
strength
weight
copies
blend
```

control.

Reference strength continues to be controlled by the existing Krea2 Identity Edit:

```text
ref_boost
```

PoC should not introduce another competing strength mechanism.

---

# 33. No latent interpolation

Never perform:

```python
F.interpolate(reference_latent, ...)
```

during runtime.

A central design invariant is:

> A PoC IdentityMod either exactly matches the target spatial grid or it is rejected.

This keeps the comparison scientifically useful.

---

# 34. Required unit tests

Tests must not require a full Krea2 checkpoint.

Use small deterministic tensors and fake VAE implementations where appropriate.

## Serialization roundtrip

Create a random CPU tensor.

Save as IdentityMod.

Load it.

Require:

```python
torch.equal(original, loaded)
```

Also verify dtype and shape.

---

## Metadata roundtrip

Save known metadata.

Reload.

Assert required values survive exactly.

---

## Invalid file format

Create a safetensors file with:

```text
format = something_else
```

Loader must reject it.

---

## Unsupported version

Example:

```text
format_version = 1.0.0
```

Loader must reject it.

---

## Missing latent key

File without:

```text
appearance_latent
```

must fail.

---

## Invalid latent rank

Examples:

```text
C × H × W
B × C × T × H × W
```

must fail for the stored PoC format.

Stored format must be exactly 4D:

```text
B × C × H × W
```

---

## Invalid batch

Stored:

```text
B=2
```

must fail.

---

## Path traversal

Attempt to save:

```text
../../evil
```

must fail.

---

## Extension handling

Saving:

```text
alice
```

must produce:

```text
alice.safetensors
```

Saving:

```text
alice.safetensors
```

must not produce:

```text
alice.safetensors.safetensors
```

---

## Existing filename

When overwrite is disabled:

```text
alice.safetensors
```

followed by another save must produce a safe incremented filename.

---

## Source image batch validation

Input batch >1 must fail.

---

## RGB handling

Input with four channels must use RGB only.

Do not fail simply because an alpha channel exists.

---

## Target geometry validation

IdentityMod:

```text
1024x1024
```

Target:

```text
1024x1024
```

must pass.

Target:

```text
768x1024
```

must fail.

---

## Near-matched aspect ratio

Create synthetic source/target dimensions that pass the near-match condition.

Ensure preprocessing results in exact target dimensions.

---

## Unsupported aspect ratio

Example:

```text
source = 1024x1024
target = 1344x768
```

must be rejected by PoC creation.

---

## Wrong VAE output geometry

Fake VAE deliberately returns the wrong spatial shape.

Create node must fail rather than save an invalid IdentityMod.

---

# 35. Fake VAE for unit tests

Use a minimal fake object such as:

```python
class FakeVAE:
    """Deterministic VAE stub used to validate preprocessing and node behavior."""

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        ...
```

The fake must record:

```text
number of encode calls
input shape
```

Tests must verify that the Create node calls VAE encode exactly once.

---

# 36. Required integration/manual test matrix

Perform real ComfyUI A/B testing with actual Krea2 Identity Edit.

Use at least three target geometries:

```text
1024x1024
768x1024
1024x768
```

For each geometry, use a source image with the same or sufficiently near-matched aspect ratio.

Use at least three fixed seeds per geometry.

Use the same:

```text
Krea2 checkpoint
Identity Edit LoRA
LoRA strength
prompt
negative conditioning
seed
sampler
scheduler
steps
CFG
ref_boost
target resolution
grounding_px
source image
```

between baseline and IdentityMod tests.

---

# 37. Baseline A/B workflow

## A — upstream pixel path

Use normal Krea2 Identity Edit v1.2:

```text
Reference IMAGE
    ↓
Krea2EditModelPatch.source_image

VAE
    ↓
Krea2EditModelPatch.vae

Target latent
    ↓
Krea2EditModelPatch.target_latent

fit_mode = fit
```

and use the same IMAGE in:

```text
Krea2EditGroundedEncode
```

---

## B — IdentityMod path

Create the IdentityMod for the same target resolution.

Then:

```text
IdentityMod
    ↓
To Latent
    ↓
Krea2EditModelPatch.source_latent
```

Disconnect:

```text
source_image
vae
target_latent
```

from `Krea2EditModelPatch`.

Set:

```text
fit_mode = crop (legacy)
```

Keep the same source IMAGE in:

```text
Krea2EditGroundedEncode
```

All other parameters must remain identical.

---

# 38. A/B expected result

For a source that uses the supported full-grid near-matched path, the source latent generated during IdentityMod creation should represent the same reference grid as the baseline pixel path.

Primary goal:

```text
Baseline output == IdentityMod output
```

Ideally bit-identical on a deterministic environment.

If not bit-identical, investigate before accepting the PoC.

Potential acceptable reason:

```text
non-deterministic GPU kernels
```

but this must not be used as an automatic excuse.

Before accepting numerical differences, verify:

1. Preprocessed input image tensors match.
2. VAE output shapes match.
3. Raw VAE latents match.
4. Target grid matches.
5. No runtime latent interpolation occurs.
6. Prompt conditioning is identical.
7. Same `ref_boost`.
8. Same seed and sampler state.

The expected visual result must be indistinguishable.

---

# 39. Performance validation

The PoC does not need a formal benchmark suite.

However demonstrate that during IdentityMod generation-time usage:

```text
Krea2EditModelPatch
```

does not call the source VAE.

The appearance reference VAE encode must happen only during IdentityMod creation.

This can be demonstrated by:

```text
disconnecting source_image and vae from Krea2EditModelPatch
```

and successfully generating with the loaded IdentityMod.

Do not claim that the complete pipeline is image-free or VAE-free:

The generation pipeline may still use a VAE for output decoding.

The requirement concerns only **source/reference appearance encoding**.

---

# 40. Acceptance criteria

PoC v0.1 is considered complete only if all of the following are true.

### Repository

The repository loads successfully as a ComfyUI custom node pack.

### Nodes

All five nodes appear:

```text
Krea2 IdentityMod Create
Krea2 IdentityMod Save
Krea2 IdentityMod Load
Krea2 IdentityMod To Latent
Krea2 IdentityMod Info
```

under:

```text
ArtemKo7v/Krea2 IdentityMod
```

### Prefixes

All public class names and all `NODE_CLASS_MAPPINGS` keys use:

```text
ArtemKo7v
```

### Language

All repository-facing texts are English.

### File creation

A reference image can be converted to a target-specific `.safetensors` IdentityMod.

### Roundtrip

The VAE latent survives save/load exactly:

```python
torch.equal(...) is True
```

### Runtime

A loaded IdentityMod can be converted to `LATENT` and connected to the existing:

```text
Krea2EditModelPatch.source_latent
```

### No appearance VAE encode at runtime

The source image and VAE are not required on the Krea2EditModelPatch appearance path.

### Resolution protection

Using the IdentityMod with another target resolution raises an explicit error.

### AR protection

Unsupported aspect-ratio creation raises an explicit error.

### A/B

Supported real-model A/B tests produce equivalent output.

### Tests

All unit tests pass.

### Documentation

README contains installation, creation, usage, limitations and A/B instructions.

---

# 41. README requirements

README must be written entirely in English.

Required sections:

```text
# ComfyUI-Krea2IdentityMod

## What this is

## PoC status

## How it works

## Requirements

## Installation

## Creating an IdentityMod

## Using an IdentityMod

## Current limitations

## File format

## A/B validation

## Troubleshooting

## Development

## License and attribution

## Roadmap
```

At the top, clearly state:

```text
This repository is an experimental Proof of Concept.
```

Clearly explain:

```text
PoC v0.1 caches only the VAE appearance path.
The reference image is still required by Krea2EditGroundedEncode.
```

Clearly explain:

```text
PoC v0.1 IdentityMods are target-resolution-specific.
```

Clearly explain:

```text
Arbitrary mismatched aspect ratios are intentionally not supported yet.
```

Do not market the PoC as a finished replacement for Krea2 reference images.

---

# 42. Troubleshooting requirements

README must include at least these cases.

### Target mismatch

```text
IdentityMod was created for another output resolution.
```

Solution:

```text
Use the matching IdentityMod or create another one.
```

### Aspect ratio rejected

Explain that PoC v0.1 only covers the full-grid near-matched geometry.

### Wrong latent channels

Explain that the wrong VAE/model combination may be connected.

### Krea2 output quality is poor

Confirm:

```text
Krea2EditGroundedEncode still receives the original source image.
Identity Edit LoRA is enabled.
The same target geometry is used.
ref_boost is configured as intended.
```

### Upstream fit warning

When IdentityMod is used through `source_latent` only, configure existing:

```text
fit_mode = crop (legacy)
```

because the cached latent is already a full target-grid latent.

---

# 43. Workflow files

If a runnable ComfyUI environment is available during development, include two exported workflows:

```text
workflows/create_identitymod_poc.json
workflows/use_identitymod_poc.json
```

Do not manually fabricate workflow JSON if it cannot be validated.

If a real ComfyUI environment is unavailable, include precise workflow diagrams in README and add workflow JSON only after validation.

The repository must never ship known-broken example workflows.

---

# 44. License and attribution

Preferred repository license:

```text
Apache-2.0
```

If any preprocessing logic is directly copied or materially adapted from `ComfyUI-Krea2Edit`, retain appropriate Apache-2.0 attribution.

Add a `NOTICE` file explaining compatibility work with:

```text
ComfyUI-Krea2Edit
Krea 2
Krea2 Identity Edit
```

Do not include or redistribute:

```text
Krea2 model weights
Krea2 Identity Edit LoRA weights
Qwen model weights
```

The repository contains nodes only.

---

# 45. Avoid hidden dependency on upstream implementation details

The custom node pack may require `ComfyUI-Krea2Edit` for the example runtime workflow, but internal serialization must not import arbitrary implementation internals from that custom-node repository.

Do not do fragile imports such as:

```python
from comfyui_krea2edit.__init__ import _fit_encode_image
```

Do not dynamically search another custom-node repository and import private underscore functions.

Implement the small, necessary near-match preprocessing logic locally, clearly documented and attributed where appropriate.

This prevents breakage when upstream reorganizes internal code.

---

# 46. Do not implement these features in PoC v0.1

Explicit non-goals:

```text
Qwen vision-feature caching
Qwen hidden-state caching
prompt-independent VLM cache
prompt-conditioned embedding cache
arbitrary aspect ratio FIT references
multiple resolution buckets in one file
automatic bucket selection
multi-reference
two-person support
IdentityMod stacking
strength blending
latent interpolation
reference masks stored in IdentityMod
LoRA loading
Krea2 model loading
custom sampler
custom grounded encoder
custom Krea2 DiT forward
training
fine-tuning
new LoRA
preview image embedding
thumbnail UI
web UI extensions
JavaScript frontend
cloud downloads
Hugging Face downloads
automatic model installation
```

If implementation starts drifting into any of these areas, stop and return to the PoC objective.

---

# 47. Expected future architecture, but do not implement it

The file format should be designed so future versions can add more tensor keys.

Possible future representation:

```text
appearance.square.latent
appearance.portrait.latent
appearance.landscape.latent

qwen.merged
qwen.deepstack.0
qwen.deepstack.1
qwen.deepstack.2
qwen.grid
```

PoC v0.1 must **not** create these.

The versioned file format should simply avoid assumptions that would make expansion impossible.

Unknown future tensor keys should not cause unsafe behavior, but required v0.1 keys must remain validated.

---

# 48. Suggested development sequence for Codex

Implement in this order.

## Stage 1 — Repository scaffold

Create:

```text
package files
constants
README skeleton
license
tests directory
```

Make sure ComfyUI loads the package.

---

## Stage 2 — Data model and serialization

Implement:

```text
ArtemKo7vKrea2IdentityModData
metadata validation
save safetensors
load safetensors
roundtrip tests
```

Do not implement nodes until serialization tests pass.

---

## Stage 3 — Geometry and preprocessing

Implement:

```text
image validation
target latent validation
near-match check
center crop
bicubic resize
VAE encode
latent shape validation
```

Add tests using FakeVAE.

---

## Stage 4 — Create / Save / Load nodes

Implement the three lifecycle nodes.

Test them with lightweight ComfyUI stubs where possible.

---

## Stage 5 — To Latent / Info nodes

Implement strict runtime target validation.

Ensure no interpolation occurs.

---

## Stage 6 — Real workflow validation

Test against actual:

```text
Krea2
Krea2 Identity Edit LoRA
ComfyUI-Krea2Edit
```

Perform baseline vs IdentityMod A/B tests.

---

## Stage 7 — Documentation

Complete README only after actual behavior is confirmed.

Do not document hypothetical functionality.

---

# 49. Code review checklist

Before considering the work finished, manually review:

```text
[ ] No unprefixed public classes.
[ ] No Russian strings.
[ ] No unsafe torch.load.
[ ] No pickle.
[ ] No arbitrary path writes.
[ ] No latent resizing.
[ ] No silent AR fallback.
[ ] No silent target-resolution fallback.
[ ] No Qwen caching.
[ ] No custom Krea2 forward.
[ ] No source VAE call during loaded IdentityMod runtime path.
[ ] Raw VAE latent is stored, not process_latent_in output.
[ ] Save/load preserves tensor dtype exactly.
[ ] Save/load preserves tensor values exactly.
[ ] Invalid files fail with clear messages.
[ ] Every node has an English description.
[ ] Important inputs have English tooltips.
[ ] Public classes and methods are documented in English.
[ ] Unit tests pass.
[ ] README matches actual implementation.
```

---

# 50. Final deliverables

The final Codex implementation must leave the repository in a state where:

```bash
git clone <repo>
```

into:

```text
ComfyUI/custom_nodes/ComfyUI-Krea2IdentityMod
```

followed by a ComfyUI restart exposes all required nodes.

The repository must contain:

```text
working node implementation
working safetensors serialization
strict validation
unit tests
English documentation
license
attribution
example workflows or validated workflow diagrams
```

No model weights are included.

---

# 51. Definition of success

The most important demonstration is this:

### Before

Every generation requires:

```text
reference IMAGE
    ↓
reference preprocessing
    ↓
VAE encode
    ↓
Krea2 source latent
```

### After

Creation happens once:

```text
reference IMAGE
    ↓
reference preprocessing
    ↓
VAE encode
    ↓
alice_1024x1024.safetensors
```

and subsequent supported generations use:

```text
alice_1024x1024.safetensors
    ↓
Load IdentityMod
    ↓
cached raw VAE latent
    ↓
Krea2EditModelPatch.source_latent
```

with no VAE encoding of the appearance reference during generation.

The generated result must remain equivalent to the baseline.

That single result is the core purpose of PoC v0.1.

Do not expand the scope until this is demonstrated.