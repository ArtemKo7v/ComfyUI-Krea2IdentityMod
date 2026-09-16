# ComfyUI-Krea2IdentityMod

This repository is an experimental Proof of Concept.

PoC v0.2 adds a prompt-independent Qwen3-VL vision cache to the successful v0.1
appearance-cache baseline. Full IdentityMods are designed for image-free generation;
appearance-only v0.1 files still need the image in upstream grounded encoding.
Each IdentityMod is target-resolution-specific. Arbitrary mismatched aspect ratios
are intentionally not supported yet.

## What this is

A small ComfyUI node pack that encodes one reference image once, saves its raw VAE
latent in a portable `.safetensors` file, and exposes the loaded reference as a
standard `LATENT` for the existing `Krea2EditModelPatch.source_latent` input.

The file can also hold Qwen visual features. The current prompt is always processed
again by the Qwen language model. No training or weight modification is performed.

## PoC v0.2 status

Implemented: seven nodes, visual extraction, scoped injection into stock Qwen token
processing, complete-cache validation, and lossless v0.1/v0.2 serialization.
All 53 CPU unit tests pass. v0.1 appearance behavior and its 5D runtime handoff are retained.

v0.1 is the accepted appearance baseline. **v0.2 real-model conditioning equivalence,
image-free generation, and performance are pending**, not established by API-double tests.
See [validation status](VALIDATION.md). No unvalidated workflow JSON is distributed.

## Image-free runtime

```text
Load IdentityMod
    +--> To Latent (+ target LATENT) --> Krea2EditModelPatch.source_latent
    +--> Grounded Encode (+ CLIP + current prompt) --> KSampler.positive
    +--> Grounded Encode (+ CLIP + empty prompt) ----> KSampler.negative
```

The generation graph needs neither the source image nor its VAE encoder nor Qwen
vision-tower execution. It still needs the Qwen language model and an output VAE decoder.
ComfyUI may load the combined text/vision checkpoint: this PoC skips vision execution,
not necessarily its weight allocation. No reduced-VRAM or speedup claim is made.

## Qwen3-VL vision cache

The cache stores `merged`, integer `grid`, and all three DeepStack tensors, captured
from the loaded transformer's `preprocess_embed` boundary. It does not store the
12 prompt-conditioned Krea2 hidden states. CPU storage preserves values and dtypes;
there is no filtering, feature normalization, or quantization.

Runtime uses stock tokenization, image spans, MRoPE, DeepStack injection, language
layers, 12 Krea2 taps, and scheduled conditioning. A namespaced per-call descriptor
temporarily replaces only image preprocessing. An `RLock` serializes this pack's
extraction/injection calls; `finally` restores the original method even on failure.
Ordinary image descriptors delegate unchanged. This is not a global lock for unrelated
extensions operating on the same CLIP concurrently; run validation in an idle process.
No persistent GPU feature copies or patched methods are retained.

## Appearance-only baseline

```text
Creation:
IMAGE + VAE + target LATENT
    -> near-match check -> center crop -> bicubic pixel resize -> VAE encode once
    -> raw CPU latent + metadata -> Save -> identity.safetensors

Generation:
Load -> To Latent (+ current target LATENT) -> Krea2EditModelPatch.source_latent
Original IMAGE ----------------------------> Krea2EditGroundedEncode.image
```

The file stores the raw output of `vae.encode()`, before model latent scaling.
`Krea2EditModelPatch` remains responsible for scaling, device placement, reference
strength (`ref_boost`), and inference. Output decoding can still use a VAE.

Nodes appear under `ArtemKo7v/Krea2 IdentityMod`:

| Node | Inputs | Output |
| --- | --- | --- |
| Krea2 IdentityMod Create | image, vae, target_latent, identity_name; optional description | identity_mod |
| Krea2 IdentityMod Save | identity_mod, filename; optional overwrite | relative filename |
| Krea2 IdentityMod Load | identity_mod_file | identity_mod |
| Krea2 IdentityMod To Latent | identity_mod, target_latent | source_latent |
| Krea2 IdentityMod Info | identity_mod | info string |
| Krea2 IdentityMod Add Qwen Vision Cache | identity_mod, clip, image, grounding_px | new identity_mod |
| Krea2 IdentityMod Grounded Encode | clip, identity_mod, prompt; optional system_prompt | CONDITIONING |

The internal socket type is `ARTEMKO7V_KREA2_IDENTITY_MOD`. Every node ID and public
class has the `ArtemKo7v` prefix. Info returns a string; connect a string display node
to view it. Save is an output node and can terminate the creation workflow.

## Requirements

- Python 3.10 or newer and a working ComfyUI installation.
- PyTorch and `safetensors`, already supplied by ComfyUI.
- A Krea2-compatible VAE for creation and a compatible target latent, normally from
  `EmptySD3LatentImage` (16 channels for Krea2).
- For generation: Krea2, the matching Identity Edit LoRA, a Qwen3-VL encoder with
  vision support, and [ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit).
  The upstream patch must expose `source_latent` and `fit_mode = crop (legacy)`.

Serialization does not import ComfyUI-Krea2Edit internals. No model weights are included
or downloaded, and no additional Python dependencies are required by this node pack.

## Installation

From your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/ArtemKo7v/ComfyUI-Krea2IdentityMod.git
```

Restart ComfyUI. Search for `Krea2 IdentityMod`. Files are saved under
`ComfyUI/models/krea2_identitymods/`; the directory is created on the first save.
You can copy compatible files there manually, including into subdirectories.

The loader also searches folders registered under `krea2_identitymods` in ComfyUI,
including extra model paths. Saving always uses the default `models/krea2_identitymods`
directory. Do not put IdentityMod files in the LoRA directory.

## Creating an IdentityMod

1. Load one reference image, such as a square portrait for a `1024x1024` target.
2. Load the Krea2-compatible VAE.
3. Create a target latent at the intended output resolution.
4. Connect these three outputs to Create and enter an identity name and optional description.
5. Connect Create's `identity_mod` to Save. Set a filename such as `alice/square_1024`.
6. Queue the workflow. Keep the returned filename for generation.

```text
Load Image.IMAGE ----------------> Create.image
VAE Loader.VAE -----------------> Create.vae
EmptySD3LatentImage.LATENT ------> Create.target_latent
Create.identity_mod ------------> Save.identity_mod
```

The file will be `models/krea2_identitymods/alice/square_1024.safetensors`.
With overwrite disabled, subsequent executions select `_001`, `_002`, and so on.
ComfyUI can cache unchanged node executions: queueing an unchanged workflow may reuse
the previous result. Change an input when another save execution is needed.

Preprocessing uses target pixel dimensions equal to latent H/W multiplied by 8. The
source must satisfy the near-match test in both axes:

```python
scale_fit = min(target_height / source_height, target_width / source_width)
source_height * scale_fit >= target_height * 0.92
source_width * scale_fit >= target_width * 0.92
```

Supported images are center-cropped and resized to the exact target grid with
float32 bicubic interpolation and antialiasing. Only RGB is encoded; alpha is discarded
and pixels are clamped to `[0, 1]`. A genuinely mismatched aspect ratio raises an error.

## Creating a full IdentityMod

1. Create the appearance IdentityMod as above, or load an existing v0.1 file.
2. Connect it to **Krea2 IdentityMod Add Qwen Vision Cache**.
3. Connect the Krea2 Qwen3-VL 4B `CLIP` and the **same original source IMAGE**.
4. Set `grounding_px`, normally `768`, matching the direct grounded-encoding baseline.
5. Connect the updated `identity_mod` output to the existing Save node. Save under a
   new name, for example `alice/square_1024_qwen768`, to retain the old file.
6. Info should report format `0.2.0`, Qwen cache `yes`, and three DeepStack tensors.

The pack cannot prove both caches came from the same image. You must supply the same
source yourself. Calling Add Qwen Vision Cache again replaces, rather than merges,
the visual cache and leaves the input object and appearance values unchanged.

## Using a full IdentityMod

1. Load the saved v0.2 file in a separate generation workflow.
2. Wire Load to To Latent and keep the matching target connected to To Latent and KSampler.
3. Wire To Latent to `Krea2EditModelPatch.source_latent`, select `crop (legacy)`, and
   disconnect the patch's source image/VAE inputs so they cannot override the cache.
4. Add two **Krea2 IdentityMod Grounded Encode** nodes. Connect the same loaded
   IdentityMod and Krea2 CLIP to both. Enter the edit prompt in the positive node;
   leave the negative prompt empty. Connect their outputs to KSampler.
5. Keep the model, Identity Edit LoRA, sampler, and output VAE decoding connections.
6. Remove the source Load Image, creation nodes, source VAE Encode, and upstream
   image-grounded encoders from the runtime graph. Queue generation.

Empty or whitespace-only `system_prompt` uses the upstream default. A custom system
prompt is applied at runtime; neither it nor the edit prompt is stored in the cache.

## Grounding resolution

`grounding_px` belongs to cache creation only. It caps the longest image side using
area downscaling without upscaling; `0` preserves native input dimensions. Qwen then
performs its normal patch-grid preprocessing. Info reports both the cap and the
image dimensions passed to Qwen (not the internal patch-rounded dimensions).
Rebuild the Qwen cache for another resolution. The runtime node has no resolution
control. Appearance target geometry and grounding resolution are independent.

## Text encoder compatibility

PoC v0.2 Qwen visual caches are intended for the Krea2 Qwen3-VL 4B text encoder family.
For strict reproducibility, create and use the cache with the same text-encoder
checkpoint and quantization configuration. Family, width, tap layers, and schema are
checked; checkpoint fingerprints are not. A generic Qwen encoder is not a substitute
for the Krea2 encoder. Unsupported API/schema changes fail explicitly.

## Backwards compatibility

Create still produces appearance-only format `0.1.0`. Existing v0.1.x files load with
no vision cache and work through To Latent exactly as before. Grounded Encode rejects
them with instructions to add a cache; it never falls back silently to text-only encoding.
Adding a complete vision cache produces `0.2.0`. The same Save/Load nodes handle both.

## Using an appearance-only IdentityMod

Start with a working single-reference Krea2 Identity Edit workflow and use these connections:

```text
Load.identity_mod ----------------------> To Latent.identity_mod
EmptySD3LatentImage.LATENT --------------> To Latent.target_latent
EmptySD3LatentImage.LATENT --------------> KSampler.latent_image
To Latent.source_latent ----------------> Krea2EditModelPatch.source_latent
Krea2 model + Identity Edit LoRA --------> Krea2EditModelPatch.model
Krea2EditModelPatch.MODEL ---------------> KSampler.model
Original reference IMAGE ---------------> Krea2EditGroundedEncode.image
Krea2EditGroundedEncode.CONDITIONING ----> KSampler.positive
```

Keep your baseline negative conditioning and output decoding connections. If the
negative branch uses grounded encoding, keep the same image there as well.

Set `Krea2EditModelPatch.fit_mode` to **`crop (legacy)`**. Disconnect its `source_image`,
`vae`, and `target_latent` inputs for this cached path. The current target still connects
to **To Latent** and the sampler. Supplying `vae + source_image` on the patch would
activate its pixel path and override the cached reference.

Keep `ref_boost` and any `ref_boost_mask` on the existing patch. They are not stored in
the file. Identity name and description are metadata only.

To Latent checks spatial dimensions and channels, then returns an independent copy of
the raw CPU tensor with shape `1 x C x 1 x H x W`, restoring the singleton frame axis
required by Krea2's channel normalization. It does not resize, crop, quantize, or change dtype.
The target
may have a larger generation batch, but the stored reference always has batch size one.

Version 0.1.1 fixes the original 4D runtime handoff: Krea2's Wan21 normalization would
broadcast a `1 x 16 x H x W` reference to `1 x 16 x 16 x H x W`, causing incorrect
channel normalization. Existing `.safetensors` files remain compatible; update the node
pack and restart ComfyUI. The file format version remains `0.1.0`.

## Current limitations

- One reference per file; no reference lists, merging, strength blending, or training.
- One fixed output geometry per file. A `1024x1024` file cannot be used at `768x1024`.
- Only the supported near-matched aspect-ratio branch is implemented.
- Target pixel H/W must be divisible by **16**, equivalent to even latent H/W. Krea2's
  2x2 latent patches otherwise cause upstream padding and potentially latent interpolation.
- Stored latents are exactly `1 x C x H x W`. Image-only `B x C x 1 x H x W` targets
  and VAE results are accepted; the VAE result's singleton frame axis is removed.
- One Qwen grounding resolution per file; no multi-reference support, embedded preview,
  frontend extension, model downloads, or sampling modifications.
- File geometry validation cannot identify which VAE weights were used. Use the same
  compatible VAE and model configuration for a valid comparison.
- Atomic non-overwriting saves require filesystem hard-link support (for example NTFS
  or ext4). Unsupported filesystems fail explicitly rather than overwriting an existing file.

## File format

Safetensors format `krea2_identitymod`. The loader accepts stable `0.1.x` and `0.2.x`
versions and rejects other major/minor versions. Appearance-only files contain one tensor;
full-cache files contain all six:

| Key | Representation |
| --- | --- |
| `appearance_latent` | CPU-contiguous floating-point raw VAE latent, `1 x C x H x W` |
| `qwen_vision_merged` | Floating-point merged visual features, `tokens x 2560` |
| `qwen_vision_grid` | Integer `1 x 3` tensor containing `T, H, W`; `T=1` |
| `qwen_vision_deepstack_0/1/2` | Three floating-point tensors, each `tokens x 2560` |

Original dtype and values are preserved; no FP16 conversion or quantization occurs.
Metadata values are strings. Required fields are:

| Fields | Meaning or required value |
| --- | --- |
| `format`, `format_version` | `krea2_identitymod`; `0.1.0` or `0.2.0` when writing |
| `creator`, `model_family` | `ComfyUI-Krea2IdentityMod`, `krea2` |
| `identity_name`, `description` | User-supplied descriptive metadata |
| `purpose` | `appearance_reference` |
| `target_width`, `target_height` | Target pixel dimensions |
| `target_latent_width`, `target_latent_height` | Stored latent spatial dimensions |
| `latent_channels`, `latent_dtype` | Channel count and exact dtype, e.g. `torch.float32` |
| `source_width`, `source_height` | Original image dimensions |
| `preprocess_mode` | `full_target_grid_near_matched_fit` |
| `vae_downscale_factor`, `qwen_cache_present` | `8`; `false` (v0.1) or `true` (v0.2) |
| `created_at_utc` | ISO 8601 UTC creation timestamp |

The loader cross-checks metadata against tensor shape and dtype, rejects non-finite
latent values, and tolerates unknown metadata and unrelated tensor keys without using them.
The reserved `qwen_vision_` tensor namespace must be complete and contain exactly the
five defined visual tensors. Partial caches and contradictory versions/flags are rejected.
For full files, required metadata also includes `qwen_cache_schema=qwen3vl_visual_v1`,
`qwen_model_family=qwen3vl`, `qwen_model_type=qwen3vl_4b`, `qwen_grounding_px`,
`qwen_input_width/height`, `qwen_merged_dtype/tokens/width`, `qwen_grid_dtype/shape`,
and `qwen_deepstack_count/dtype/width`. Grid shape is a JSON list string, `[1, 3]`.
It never deserializes Python objects. Reload detection uses resolved path, modification
time, creation/change time, and file size. A file replacement that preserves all of
these attributes is not detected by a content hash.

Saving accepts relative subfolders. Absolute paths, traversal, reserved Windows names,
and links escaping the storage root are rejected. Non-overwriting saves publish a
complete temporary file atomically under an available filename; overwrite uses atomic
replacement. Neither mode intentionally exposes a partial `.safetensors` file.

## A/B validation

The v0.2 primary comparison is **actual CONDITIONING**, not visual similarity:
upstream `Krea2EditGroundedEncode` with the image versus the new encoder with a saved
and reloaded cache. Reuse one cache per resolution across short, long, scene-change,
and empty prompts, with default and custom system prompts; repeat at 512/768/1024.
Record tensor shape, dtype, max/mean absolute error, `torch.equal`, and `torch.allclose`.
Also compare attention masks and scheduled entries. Investigate any nonzero errors.

An opt-in developer helper in [validation.py](validation.py) accepts already loaded
objects and the real upstream `encode` callable. It runs the matrix, save/load,
synchronized encode timings, and a vision-tower-raises guard. It does not access
models or run on import; it is not a standalone command or a GUI node. Usage in an
existing in-process test harness, where `identitymod_package` is this loaded package:

```python
from identitymod_package.validation import run_qwen_ab
records = run_qwen_ab(clip, appearance_identity_mod, image,
                      upstream_grounded_encoder.encode)
```

The returned list is JSON-serializable. Save it with the exact software/model/device
configuration. Timings include encode overhead and may include loading/warm-up;
repeat in alternating order before claiming performance improvements. Full image-free
sampler generation still needs a separate manual workflow run.

The following remains the frozen **appearance-path** comparison for v0.1 regression:

Run the matrix in [VALIDATION.md](VALIDATION.md) with actual Krea2 weights. Both branches
must use the same source image, target resolution, checkpoint, LoRA and strength,
prompt, negative conditioning, seed, sampler, scheduler, steps, CFG, `ref_boost`, masks,
and `grounding_px`.

**A: upstream pixel path.** Connect the image, VAE, and current target to
`Krea2EditModelPatch`; select `fit`. Keep its required `source_latent` input connected
as in your working upstream workflow (normally `VAEEncode` of the reference). The
pixel path overrides that source during sampling.

**B: saved IdentityMod path.** Create and save for the same target. Use a separate
generation graph containing Load and To Latent, wired as above, with `crop (legacy)`.
Keep the original reference in grounded encoding. The generation graph should not
execute Create or a source `VAEEncode` node. Output decoding may still use a VAE.

First verify exact `torch.equal` equality of the preprocessed images and raw VAE
latents, including save/load. Then compare sampler results and decoded outputs.
The goal is identical output in a deterministic environment. Investigate any difference
in tensors, preprocessing, target padding, runtime interpolation, conditioning, or
sampler state before attributing it to nondeterministic GPU kernels. Do not mark the PoC
validated based only on unit tests or visual similarity without investigating differences.

Successful generation in B without the source image/VAE connected to the appearance
patch demonstrates that repeated appearance encoding is unnecessary. It does not
demonstrate an image-free or completely VAE-free pipeline, or a measured end-to-end speedup.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Target geometry mismatch | Load the matching IdentityMod or create one for the current output resolution. |
| Aspect ratio rejected | Prepare the source at the target aspect ratio. PoC v0.1 supports only the full-grid near-match branch. |
| Even latent H/W required | Use pixel dimensions divisible by 16 to avoid upstream patch-padding interpolation. |
| Wrong latent channels or encoded shape | Check the Krea2 VAE, target latent, and model combination. |
| Stripes or distorted appearance with a loaded reference | Update to node pack 0.1.1 or later and restart. To Latent must restore the singleton frame axis before Krea2 normalization. |
| Poor Krea2 output quality | Check the correct vision cache (or source image for v0.1), Identity Edit LoRA, target geometry, and `ref_boost`. |
| Upstream `fit` warning | Use `crop (legacy)` for the cached full-grid latent. |
| Empty Load dropdown | Save/copy a file into `models/krea2_identitymods`, then refresh node definitions or restart ComfyUI. |
| File cannot be saved | Use a valid relative filename, check permissions, and use storage supporting atomic hard links. |
| Unsupported format or version | Select an IdentityMod `0.1.x` or `0.2.x` file, not a checkpoint or LoRA. |
| Missing Qwen cache | Add Qwen Vision Cache to the appearance file and save the updated output. |
| Unsupported CLIP | Use the Krea2 Qwen3-VL 4B encoder with vision support and 12 Krea2 taps. |
| Wrong image placeholder count | Remove extra image/ChatML special tokens from the prompt. |

## Known testing caveats

The previously reported overcooked texture was isolated during v0.1 validation and
is not being treated as a cache/serialization defect. Reusing the source-generation
prompt and seed can be misleading for visual-quality evaluation. Fixed prompt/seed
is appropriate for path-equivalence comparisons; use changed prompts and/or seeds
for representative quality tests. Do not add filtering to either cache to mask it.

## Development

Run from this repository with a Python environment containing `torch` and `safetensors`:

```bash
python -m unittest discover -s tests -t . -v
```

No pytest, checkpoint, or ComfyUI installation is required for unit tests. Tests use
small tensors, a deterministic VAE stub, real safetensors files, and a temporary folder
registry. Windows junctions or POSIX symlinks are used for containment checks where allowed.

Modules separate constants, data/metadata validation, preprocessing, serialization,
paths, and node definitions. `__init__.py` exports the mappings and registers the
model directory. Importing the package does not create directories or load models.

## License and attribution

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for attribution to
[ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit). The local preprocessing
adapts its near-match full-grid branch. Krea 2, Identity Edit, and Qwen weights are not
included; their respective licenses apply separately.

## Roadmap

1. Complete v0.2 real-model conditioning A/B and prove vision execution is skipped.
2. Export and validate creation and generation workflows from that environment.
3. Consider arbitrary reference geometry and multiple resolution buckets after this PoC passes.
4. Consider compatibility fingerprints and multi-reference support only after v0.2 validation.
