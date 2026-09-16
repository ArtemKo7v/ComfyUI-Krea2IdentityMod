# ComfyUI-Krea2IdentityMod — PoC v0.2 Technical Task

## Image-free runtime by caching the Qwen3-VL vision path

Repository: `ComfyUI-Krea2IdentityMod`  
Target milestone: `PoC v0.2`  
License: `MIT`

---

## 1. Context

PoC v0.1 has already confirmed the primary IdentityMod hypothesis:

- the Krea2 Identity Edit appearance reference can be represented by the raw source VAE latent;
- the raw latent can be stored losslessly in `.safetensors`;
- loading the saved latent and passing it through `Krea2EditModelPatch.source_latent` reproduces the direct latent path;
- pure VAE roundtrip is clean;
- safetensors roundtrip is exact;
- direct latent runtime and IdentityMod runtime are equivalent;
- decoding the direct latent and the loaded IdentityMod latent produces the same result.

The previously observed noisy / overcooked output was isolated and is **not** an IdentityMod or VAE serialization issue. It occurred when sampling an image with the same prompt and seed used to create the source image. Changing the prompt and/or seed removes the artifact.

Therefore PoC v0.1 should be treated as successful and frozen as the appearance-cache baseline.

The remaining runtime dependency on the source image is the semantic path:

```text
Reference IMAGE
    ↓
Krea2EditGroundedEncode
    ↓
Qwen3-VL vision tower
    ↓
image visual embeddings
    +
current prompt
    ↓
Qwen3-VL language model
    ↓
12 tapped hidden states
    ↓
Krea2 CONDITIONING
```

PoC v0.2 must cache only the **prompt-independent visual part** of this path.

---

## 2. Main goal

Implement a portable Qwen3-VL vision cache inside the existing Krea2 IdentityMod file so that the original reference image is no longer required during generation.

Target runtime:

```text
identity.safetensors
    │
    ├── appearance_latent
    │       ↓
    │   Krea2EditModelPatch.source_latent
    │
    └── cached Qwen3-VL visual features
            +
          prompt
            ↓
      Qwen3-VL language layers
            ↓
        CONDITIONING
            ↓
          KSampler
```

After the IdentityMod has been created, normal generation must not require:

```text
Load Image
source-image VAE Encode
Qwen3-VL vision encoding of the source image
```

The current prompt must still be processed by the Qwen3-VL language model at runtime.

Do **not** cache final prompt-conditioned Krea2 conditioning.

---

## 3. Hypothesis to validate

The PoC must validate:

> The output of the Qwen3-VL vision tower used by Krea2 can be cached independently of the prompt, serialized in the IdentityMod file, re-injected into the normal Qwen3-VL token-processing path, and produce the same Krea2 conditioning as `Krea2EditGroundedEncode` with the original source image.

Primary A/B test:

```text
A:
source IMAGE
→ Krea2EditGroundedEncode
→ CONDITIONING_A

B:
same source IMAGE
→ create cached Qwen visual features once
→ save IdentityMod
→ load IdentityMod
→ ArtemKo7v cached grounded encode
→ CONDITIONING_B
```

For the same Krea2 Qwen3-VL checkpoint, prompt, system prompt, grounding resolution, ComfyUI version, device, and precision, `CONDITIONING_A` and `CONDITIONING_B` should be identical or numerically equivalent with the smallest reasonable tolerance.

---

## 4. Verified current Qwen3-VL cache boundary

At the time this task was written, current ComfyUI exposes a clean prompt-independent boundary.

Conceptually:

```text
IMAGE
  ↓
process_qwen2vl_images(...)
  ↓
Qwen3VL.visual(...)
  ↓
merged
deepstack[0]
deepstack[1]
deepstack[2]
grid
```

Current `Qwen3VL.preprocess_embed()` returns conceptually:

```python
merged, {
    "grid": grid,
    "deepstack": deepstack,
}
```

The language path later uses:

- `merged` as image token embeddings;
- `grid` to build image-aware MRoPE positions;
- DeepStack features injected at image token positions.

Therefore PoC v0.2 must cache **all** of:

```text
merged
grid
deepstack
```

Do not cache only `merged`.

Before implementation, inspect the current upstream files:

```text
ComfyUI/comfy/text_encoders/qwen3vl.py
ComfyUI/comfy/text_encoders/krea2.py
ComfyUI/comfy/sd1_clip.py
ComfyUI/comfy/sd.py
ComfyUI-Krea2Edit/__init__.py
```

If upstream changes, preserve the goal and invariants rather than stale attribute paths.

---

## 5. Architectural constraint

Krea2 currently taps 12 Qwen3-VL hidden states:

```text
2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35
```

Those hidden states are prompt-dependent.

This is invalid:

```text
image + prompt A
→ final CONDITIONING
→ save
→ reuse with prompt B
```

Correct cache boundary:

```text
image
→ Qwen vision tower
→ CACHE

CACHE + arbitrary current prompt
→ Qwen language model
→ Krea2 conditioning
```

---

## 6. Scope

Implement:

1. Qwen3-VL visual cache creation.
2. Storage of the cache in the existing IdentityMod `.safetensors`.
3. Loading and validation of both v0.1 and v0.2 IdentityMods.
4. A cached grounded-encode node that consumes the IdentityMod rather than an image.
5. Numerical A/B validation against upstream `Krea2EditGroundedEncode`.
6. Full image-free single-reference runtime.
7. Unit tests.
8. README documentation.

---

## 7. Explicit non-goals

Do not implement in PoC v0.2:

```text
multiple identities
image_b / second reference
scene + subject dual reference
IdentityMod stacking
final Qwen hidden-state caching
prompt-conditioned conditioning caching
multiple Qwen grounding buckets in one file
multiple appearance-resolution buckets in one file
automatic bucket selection
target-resolution independence
new arbitrary-AR appearance geometry
IdentityMod strength
latent blending
identity merging
training
fine-tuning
new LoRA
custom Krea2 transformer forward
custom sampler
JavaScript UI
remote model downloads
preview image embedding
thumbnail generation
full checkpoint hashing
cross-model compatibility guarantees
```

---

## 8. Preserve PoC v0.1 compatibility

Existing v0.1 files must remain loadable.

Do not rename or break:

```text
ArtemKo7vKrea2IdentityModCreate
ArtemKo7vKrea2IdentityModSave
ArtemKo7vKrea2IdentityModLoad
ArtemKo7vKrea2IdentityModToLatent
ArtemKo7vKrea2IdentityModInfo
```

The custom type remains:

```text
ARTEMKO7V_KREA2_IDENTITY_MOD
```

The existing tensor key remains:

```text
appearance_latent
```

Do not change its meaning.

---

## 9. New node: Add Qwen Vision Cache

Class:

```python
class ArtemKo7vKrea2IdentityModAddQwenVisionCache:
    """Add prompt-independent Qwen3-VL visual features to an IdentityMod."""
```

Mapping key:

```text
ArtemKo7vKrea2IdentityModAddQwenVisionCache
```

Display name:

```text
Krea2 IdentityMod Add Qwen Vision Cache
```

Category:

```text
ArtemKo7v/Krea2 IdentityMod
```

Required inputs:

```text
identity_mod   ARTEMKO7V_KREA2_IDENTITY_MOD
clip           CLIP
image          IMAGE
grounding_px   INT
```

Default:

```text
grounding_px = 768
```

Suggested limits:

```text
min = 0
max = 4096
step = 64
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
add_qwen_vision_cache
```

Description:

```text
Adds a prompt-independent Qwen3-VL visual cache to a Krea2 IdentityMod so the source image is no longer required for grounded encoding at runtime.
```

Return a new IdentityMod data object. Do not mutate a frozen object in place.

---

## 10. New node: Cached Grounded Encode

Class:

```python
class ArtemKo7vKrea2IdentityModGroundedEncode:
    """Encode a Krea2 edit prompt using cached Qwen3-VL visual features."""
```

Mapping key:

```text
ArtemKo7vKrea2IdentityModGroundedEncode
```

Display name:

```text
Krea2 IdentityMod Grounded Encode
```

Category:

```text
ArtemKo7v/Krea2 IdentityMod
```

Required inputs:

```text
clip          CLIP
identity_mod  ARTEMKO7V_KREA2_IDENTITY_MOD
prompt        STRING
```

Optional:

```text
system_prompt STRING
```

`prompt` and `system_prompt` must be multiline strings.

Default:

```text
system_prompt = ""
```

Return:

```text
CONDITIONING
```

Function:

```text
encode
```

Description:

```text
Encodes a Krea2 edit prompt using cached Qwen3-VL visual features from an IdentityMod instead of the original source image.
```

The same node must support:

- normal positive prompts;
- an empty prompt for negative conditioning.

Do not create a separate negative node.

---

## 11. Grounding image preprocessing

Cache creation must reproduce the image preprocessing semantics of current upstream `Krea2EditGroundedEncode`.

Input:

```text
B × H × W × C
```

Requirements:

```text
batch == 1
channels >= 3
```

Use RGB only.

Behavior:

- `grounding_px == 0`: keep native size;
- otherwise, if the longest side exceeds `grounding_px`, downscale while preserving aspect ratio;
- use area-style downscaling;
- do not upscale an already smaller image.

The exact image sent into Qwen3-VL should match upstream behavior.

Do not add:

```text
sharpening
denoise
gamma correction
contrast changes
extra normalization
```

---

## 12. Locate the Krea2 Qwen3-VL transformer

The supplied `CLIP` must be validated.

Expected model family:

```text
Krea2
Qwen3-VL 4B
```

Suggested helper:

```python
def _get_krea2_qwen3vl_transformer(clip):
    """Return the Qwen3-VL transformer used by a Krea2 CLIP object."""
```

Validate each attribute before dereferencing it.

If unsupported:

```text
The supplied CLIP is not a supported Krea2 Qwen3-VL text encoder.

PoC v0.2 requires the Krea2 Qwen3-VL 4B text encoder.
```

---

## 13. Preferred cache extraction path

Prefer the currently loaded transformer's own visual preprocessing boundary.

Conceptually:

```python
merged, extra = transformer.preprocess_embed(
    {
        "type": "image",
        "data": prepared_image,
    },
    device=device,
)

grid = extra["grid"]
deepstack = extra["deepstack"]
```

Requirements:

1. load the CLIP through ComfyUI model management;
2. obtain the correct execution device;
3. use the proper device context;
4. do not instantiate a second Qwen model;
5. do not permanently move or pin the encoder outside ComfyUI management.

---

## 14. Required cache tensors

The v0.2 file must store:

```text
qwen_vision_merged
qwen_vision_grid
qwen_vision_deepstack_0
qwen_vision_deepstack_1
qwen_vision_deepstack_2
```

Current Qwen3-VL 4B has three DeepStack tensors.

PoC v0.2 requires:

```text
deepstack_count == 3
```

If upstream produces another count, fail explicitly rather than creating an ambiguous file.

---

## 15. Tensor handling

Before storing:

```python
tensor = tensor.detach().cpu().contiguous()
```

Preserve dtype exactly.

Do not:

```text
quantize
convert dtype
normalize
compress numerically
```

Expected:

```text
merged      floating point
grid        integer
deepstack   floating point
```

Reject invalid dtypes.

---

## 16. Data model extension

Recommended structure:

```python
@dataclass(frozen=True)
class ArtemKo7vKrea2QwenVisionCache:
    merged: torch.Tensor
    grid: torch.Tensor
    deepstack: tuple[torch.Tensor, ...]
```

Extend:

```python
@dataclass(frozen=True)
class ArtemKo7vKrea2IdentityModData:
    reference_latent: torch.Tensor
    metadata: dict[str, str]
    qwen_vision_cache: ArtemKo7vKrea2QwenVisionCache | None = None
```

Avoid unstructured tensor dictionaries spread across multiple modules.

---

## 17. Safetensors schema v0.2

v0.2 file:

```text
appearance_latent
qwen_vision_merged
qwen_vision_grid
qwen_vision_deepstack_0
qwen_vision_deepstack_1
qwen_vision_deepstack_2
```

v0.1 remains valid with only:

```text
appearance_latent
```

---

## 18. Versioning

Appearance-only object:

```text
format_version = "0.1.0"
qwen_cache_present = "false"
```

Object with complete Qwen cache:

```text
format_version = "0.2.0"
qwen_cache_present = "true"
```

Loader must support both `0.1.x` and `0.2.x`.

A v0.2 file claiming `qwen_cache_present=true` but missing any required cache tensor must fail.

---

## 19. Qwen metadata

When cache is present, add:

```text
qwen_cache_present
qwen_cache_schema
qwen_model_family
qwen_model_type
qwen_grounding_px
qwen_input_width
qwen_input_height
qwen_merged_dtype
qwen_merged_tokens
qwen_merged_width
qwen_grid_dtype
qwen_grid_shape
qwen_deepstack_count
qwen_deepstack_dtype
qwen_deepstack_width
```

Suggested values:

```text
qwen_cache_present = "true"
qwen_cache_schema = "qwen3vl_visual_v1"
qwen_model_family = "qwen3vl"
qwen_model_type = "qwen3vl_4b"
qwen_deepstack_count = "3"
```

All safetensors metadata values remain strings.

---

## 20. Text-encoder compatibility limitation

The cached visual representation depends on the Qwen vision tower that created it.

PoC v0.2 guarantees equivalence only for the Krea2 Qwen3-VL 4B family and, for strict reproducibility, the same text-encoder checkpoint / quantization configuration.

Do not claim universal portability between arbitrary Qwen3-VL checkpoints.

Full checkpoint fingerprinting is not part of this milestone.

README wording should state:

```text
PoC v0.2 Qwen visual caches are intended for the Krea2 Qwen3-VL 4B text encoder family.
For strict reproducibility, create and use the cache with the same text-encoder checkpoint and quantization configuration.
```

---

## 21. Grounded encode architecture

Do not reproduce the whole Qwen language model or Krea2 text-encoder stack.

Preferred architecture:

```text
normal Krea2 tokenization
        ↓
image placeholder
        ↓
replace image preprocessing result with cached representation
        ↓
normal ComfyUI Qwen3-VL processing
        ↓
normal Krea2 12-layer taps
        ↓
normal CONDITIONING
```

Only this step should disappear:

```text
source image → Qwen vision tower
```

---

## 22. Token template parity

The cached path must use the same semantic token layout as current upstream `Krea2EditGroundedEncode`:

```text
system turn
user turn
vision block
current instruction
assistant prefix
```

Inspect the exact current upstream template before implementation.

Do not normalize or casually change whitespace.

Keep:

```text
<|vision_start|>
<|image_pad|>
<|vision_end|>
```

The image placeholder location is required for:

- correct image token positions;
- MRoPE;
- visual position masks;
- DeepStack injection.

---

## 23. Cached image descriptor

Use a namespaced internal marker.

Conceptual example:

```python
{
    "type": "image",
    "artemko7v_qwen3vl_cached": True,
}
```

Requirements:

- downstream logic must still treat it as an image span;
- marker must be unique to this repository;
- ordinary images must continue to use the normal path;
- cached data must not be treated as a generic text embedding.

---

## 24. Cached preprocess interception

Current generic ComfyUI CLIP token processing calls `transformer.preprocess_embed(...)`.

For the repository-specific cached descriptor, return:

```python
cached_merged, {
    "grid": cached_grid,
    "deepstack": cached_deepstack,
}
```

For every other descriptor, call the original function.

The returned structure must match the normal Qwen3-VL visual path.

---

## 25. No persistent monkey patch

If temporary interception of `preprocess_embed` is required, it must be strictly scoped.

Requirements:

1. save the original callable;
2. install wrapper;
3. encode;
4. restore the original in `finally`;
5. restore even on exception;
6. do not permanently modify the shared CLIP object;
7. protect the patch with a module-level `threading.RLock`.

Conceptual:

```python
with _QWEN_PREPROCESS_PATCH_LOCK:
    original = transformer.preprocess_embed
    try:
        transformer.preprocess_embed = patched
        result = clip.encode_from_tokens_scheduled(tokens)
    finally:
        transformer.preprocess_embed = original
```

Do not assume `clip.clone()` isolates the transformer. Current ComfyUI cloning shares `cond_stage_model`.

If a cleaner non-mutating extension point exists in the current ComfyUI version, prefer it and document it.

---

## 26. Device behavior

IdentityMod data loads on CPU.

At cached grounded encode time:

- move `merged` and DeepStack tensors to the Qwen execution device only when needed;
- respect ComfyUI model management;
- do not hard-code CUDA;
- do not permanently store GPU copies;
- do not leave hidden VRAM caches.

Match upstream behavior for `grid`.

---

## 27. Dtype behavior

Preserve cache dtypes in storage.

Do not invent a new precision policy.

Let the stock ComfyUI token-processing and Qwen paths perform their normal casting whenever possible.

Do not unconditionally cast cached features to FP16 or BF16.

---

## 28. System prompt behavior

Match upstream behavior:

```text
system_prompt == ""
```

uses the current default grounding system prompt.

A custom non-empty system prompt must also work.

The cache remains valid across different system prompts.

Do not include the system prompt in the cache.

---

## 29. Prompt-independence validation

Create one cache and reuse it for at least:

```text
"Change her outfit to a black leather jacket."
"Put her in a snowy mountain environment."
"Make this a cinematic night portrait."
""
```

For each prompt compare:

```text
upstream image-grounded encode
vs
cached IdentityMod grounded encode
```

Do not rebuild the cache between prompts.

---

## 30. Grounding resolution semantics

`grounding_px` is a **creation-time cache parameter**.

Do not expose a runtime `grounding_px` that pretends to change an already cached representation.

To use another grounding resolution, rebuild the Qwen cache.

The Info node must display the stored grounding resolution.

---

## 31. Update Save node

Keep:

```text
ArtemKo7vKrea2IdentityModSave
```

The saver must automatically include Qwen cache tensors when present.

Do not create another saver.

Support both v0.1 and v0.2.

---

## 32. Update Load node

Keep:

```text
ArtemKo7vKrea2IdentityModLoad
```

Behavior:

### v0.1

```text
reference_latent loaded
qwen_vision_cache = None
```

### v0.2

Load and validate all Qwen tensors.

Do not accept a partially valid cache.

---

## 33. Update Info node

Example v0.1:

```text
Krea2 IdentityMod

Name: Alice
Format: 0.1.0
Appearance cache: yes
Qwen vision cache: no
Target: 1024x1024
```

Example v0.2:

```text
Krea2 IdentityMod

Name: Alice
Format: 0.2.0
Appearance cache: yes
Qwen vision cache: yes
Target: 1024x1024

Qwen model: qwen3vl_4b
Grounding resolution: 768
Visual tokens: <actual>
Visual width: <actual>
DeepStack tensors: 3
Qwen cache schema: qwen3vl_visual_v1
```

Use real metadata.

---

## 34. Error on v0.1 grounded encode

Connecting a v0.1 IdentityMod to the new grounded encoder must fail clearly:

```text
This IdentityMod does not contain a Qwen3-VL vision cache.

It can still be used as an appearance source through "Krea2 IdentityMod To Latent",
but image-free grounded encoding requires a v0.2 IdentityMod with a Qwen vision cache.

Use "Krea2 IdentityMod Add Qwen Vision Cache" and save the updated IdentityMod.
```

Do not silently fall back to text-only encoding.

---

## 35. Node mappings

Add:

```python
NODE_CLASS_MAPPINGS = {
    ...
    "ArtemKo7vKrea2IdentityModAddQwenVisionCache": (
        ArtemKo7vKrea2IdentityModAddQwenVisionCache
    ),
    "ArtemKo7vKrea2IdentityModGroundedEncode": (
        ArtemKo7vKrea2IdentityModGroundedEncode
    ),
}
```

and equivalent `NODE_DISPLAY_NAME_MAPPINGS`.

All UI text remains English.

---

## 36. Suggested repository structure

```text
ComfyUI-Krea2IdentityMod/
├── __init__.py
├── nodes.py
├── constants.py
├── identity_mod.py
├── preprocessing.py
├── serialization.py
├── paths.py
├── qwen_cache.py
├── qwen_injection.py
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_serialization.py
│   ├── test_identity_mod.py
│   ├── test_paths.py
│   ├── test_nodes.py
│   ├── test_qwen_cache.py
│   └── test_qwen_injection.py
└── workflows/
```

---

## 37. Code quality

All new code must be clear, typed, readable, and documented in English.

No Russian text in:

```text
source code
UI strings
exceptions
logs
README
tests
```

---

## 38. Logging

Use `logging`, not `print()`.

Do not log full tensors.

---

## 39. Serialization tests

Required:

- v0.1 backwards compatibility;
- exact v0.2 tensor roundtrip;
- dtype preservation;
- shape preservation;
- reject missing merged;
- reject missing grid;
- reject missing DeepStack tensor;
- reject wrong DeepStack count;
- reject floating-point grid;
- reject integer merged;
- reject metadata/shape mismatch.

Use `torch.equal()` for saved/loaded tensors.

---

## 40. Cache extraction tests

Use a fake Qwen transformer and verify:

1. preprocessing runs once;
2. `preprocess_embed()` runs once;
3. merged/grid/deepstack are captured;
4. tensors are detached;
5. tensors are moved to CPU;
6. values and dtypes are unchanged.

---

## 41. Injection tests

For a normal image descriptor, the original preprocess path must run.

For an ArtemKo7v cached descriptor, the original vision preprocessing must **not** run.

After the scoped patch, the original callable must be restored.

Also test restoration after exceptions.

---

## 42. Concurrency safety

If temporary method interception is used:

- use one shared `RLock`;
- nested same-thread use must not deadlock;
- original method must always be restored.

---

## 43. Token placeholder validation

Single-reference PoC expects exactly one image placeholder.

Zero or multiple placeholders must fail explicitly.

---

## 44. Main real-model conditioning A/B test

Use the actual Krea2 Qwen3-VL 4B text encoder.

At `grounding_px=768` compare:

```text
A: reference IMAGE → Krea2EditGroundedEncode
B: loaded v0.2 IdentityMod → Krea2 IdentityMod Grounded Encode
```

Compare the actual conditioning tensors and record:

```text
shape
dtype
max absolute error
mean absolute error
torch.equal
torch.allclose
```

Preferred result:

```python
torch.equal(A, B)
```

If exact equality is not available for a legitimate device reason, use a documented tight tolerance.

Do not use only visual image similarity as proof.

---

## 45. Multi-prompt A/B

Reuse the same cache with:

- short prompt;
- long detailed prompt;
- unrelated scene-change prompt;
- empty negative prompt.

Each cached conditioning must match direct image-grounded conditioning.

---

## 46. System prompt A/B

Test default and one custom system prompt.

Both direct and cached paths must match.

---

## 47. Grounding-resolution matrix

Test independently at:

```text
512
768
1024
```

Create a separate cache for each value.

---

## 48. Image-free full runtime test

After saving v0.2, build a generation graph with **no source `Load Image` node**.

Runtime:

```text
Load IdentityMod
    ├── To Latent
    │      ↓
    │   Krea2EditModelPatch.source_latent
    │
    └── IdentityMod Grounded Encode
           ↓
       CONDITIONING
```

Generation must complete.

This is the main product-level acceptance test.

---

## 49. Positive / negative wiring

Use the same IdentityMod for:

```text
positive grounded encode
negative grounded encode with prompt=""
```

The Qwen vision tower must not run for either.

---

## 50. Prove the vision tower is skipped

Instrument the Qwen visual tower so a call raises:

```python
RuntimeError("Vision tower should not run")
```

Cached grounded encoding must still succeed.

Direct image-grounded encoding should fail under the same instrumentation.

---

## 51. Performance measurement

Record basic wall-clock time for:

```text
direct image grounded encode
cached IdentityMod grounded encode
```

Peak VRAM measurement is optional.

Do not promise a speedup unless measured.

---

## 52. Regression note: same prompt + same seed

A misleading visual-quality condition has already been isolated.

If the reference was generated by the same model and generation reuses the exact source-generation prompt and seed, the trajectory can become unusually correlated and produce overcooked texture.

This is not an IdentityMod cache error.

For path-equivalence regression testing, fixed prompt/seed is valid.

For representative visual-quality testing, use changed prompts and/or seeds.

Document this in README.

---

## 53. Do not filter the cache

Do not add blur, denoise, clamp, contrast correction, latent normalization, token normalization, or frequency filtering.

IdentityMod must remain fidelity-neutral.

---

## 54. README changes

Add/update:

```text
## PoC v0.2 status
## Image-free runtime
## Qwen3-VL vision cache
## Creating a full IdentityMod
## Using a full IdentityMod
## Grounding resolution
## Text encoder compatibility
## Backwards compatibility
## A/B validation
## Known testing caveats
```

---

## 55. Suggested creation workflow

```text
Reference IMAGE
      │
      ├──────────────► Krea2 IdentityMod Create
      │                       ▲
      │                      VAE
      │                       ▲
      │                  target_latent
      │
      └──────────────► Add Qwen Vision Cache
                              ▲
                         Krea2 CLIP
                              ▲
                         grounding_px
      ↓
Krea2 IdentityMod Save
```

---

## 56. Rebuilding the cache

Calling `Add Qwen Vision Cache` on a v0.2 object replaces the existing Qwen cache.

Do not merge visual features.

---

## 57. Source-image consistency limitation

PoC v0.2 cannot prove that the image used for the Qwen cache is the same image used for the appearance latent.

Do not implement biometric matching.

README must instruct users to use the same source image for both caches.

---

## 58. Error handling

Required explicit errors include:

```text
unsupported CLIP
missing Qwen cache
invalid DeepStack count
cache metadata mismatch
wrong token placeholder count
invalid source-image batch
```

All errors must be English and actionable.

---

## 59. License / third-party code

Repository remains MIT.

Implementation must be original.

Do not copy or vendor source files from ComfyUI or `ComfyUI-Krea2Edit`.

Using their runtime APIs and independently reproducing compatibility behavior is allowed.

---

## 60. Development sequence

### Stage 1 — freeze v0.1 baseline
Run all existing tests and preserve a known valid v0.1 fixture.

### Stage 2 — Qwen cache data model
Implement cache dataclass and validation.

### Stage 3 — v0.2 serialization
Extend save/load and prove v0.1 compatibility.

### Stage 4 — visual extraction
Implement transformer discovery and cache extraction.

### Stage 5 — cached token injection
Inject cached `merged/grid/deepstack` into stock Qwen processing.

### Stage 6 — conditioning A/B
Compare direct and cached conditioning tensors across prompts.

### Stage 7 — image-free generation
Remove source image completely from runtime.

### Stage 8 — documentation
Document only verified behavior.

---

## 61. Acceptance checklist

### Backwards compatibility

```text
[ ] Existing v0.1 IdentityMods load.
[ ] Existing v0.1 appearance workflow works.
```

### Cache creation

```text
[ ] merged is cached.
[ ] grid is cached.
[ ] all 3 DeepStack tensors are cached.
[ ] cache is prompt-independent.
```

### Serialization

```text
[ ] v0.2 roundtrip is exact.
[ ] dtypes are exact.
[ ] shapes are exact.
[ ] partial caches are rejected.
```

### Grounded encoding

```text
[ ] stock Qwen language path is used.
[ ] Qwen vision tower is skipped.
[ ] positive conditioning works.
[ ] empty negative conditioning works.
[ ] custom system prompt works.
```

### Equivalence

```text
[ ] direct and cached conditioning shapes match.
[ ] direct and cached conditioning numerically match.
[ ] multiple prompts pass.
[ ] 512 / 768 / 1024 tests pass.
```

### Full runtime

```text
[ ] generation succeeds with no source image in the runtime graph.
[ ] appearance comes from appearance_latent.
[ ] semantic grounding comes from cached Qwen visual features.
```

### Code quality

```text
[ ] all public classes use ArtemKo7v prefix.
[ ] all code/UI/error/documentation text is English.
[ ] temporary interception is always restored.
[ ] no unsafe serialization.
[ ] no permanent monkey patch.
```

---

## 62. Definition of success

Creation time, once:

```text
Reference IMAGE
   ├── VAE
   │    ↓
   │ appearance_latent
   │
   └── Qwen3-VL vision tower
        ↓
      merged
      grid
      deepstack
        ↓

identity.safetensors
```

Runtime:

```text
identity.safetensors
    │
    ├── appearance_latent ─────────► Krea2 reference tokens
    │
    └── cached Qwen visual data
                    +
                 new prompt
                    ↓
              Qwen language model
                    ↓
               Krea2 conditioning
                    ↓
                 generation
```

with:

```text
NO source image
NO source VAE encode
NO Qwen vision-tower execution
```

during generation.

Do not expand scope until this result is demonstrated.

---

## 63. Follow-up after v0.2

Do not implement as part of this task.

Likely next priorities:

```text
1. compatibility fingerprints
2. multiple appearance geometry buckets
3. target-resolution independence
4. optional multiple grounding-resolution Qwen caches
5. multi-reference architecture
```

---

## 64. Reference URLs

Current ComfyUI Qwen3-VL implementation:

https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/text_encoders/qwen3vl.py

Current ComfyUI Krea2 text encoder:

https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/text_encoders/krea2.py

Current generic CLIP token processing:

https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/sd1_clip.py

Current ComfyUI CLIP wrapper / model management:

https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/sd.py

Current Krea2 Edit custom nodes:

https://github.com/lbouaraba/comfyui-krea2edit/blob/main/__init__.py

Krea2 Edit documentation:

https://github.com/lbouaraba/comfyui-krea2edit/blob/main/README.md
