# ComfyUI-Krea2IdentityMod

This repository is an experimental Proof of Concept.

PoC v0.1 caches only the VAE appearance path of Krea2 Identity Edit. The reference
image is still required by `Krea2EditGroundedEncode` for Qwen3-VL semantic grounding.
Each IdentityMod is target-resolution-specific. Arbitrary mismatched aspect ratios
are intentionally not supported yet.

## What this is

A small ComfyUI node pack that encodes one reference image once, saves its raw VAE
latent in a portable `.safetensors` file, and exposes the loaded reference as a
standard `LATENT` for the existing `Krea2EditModelPatch.source_latent` input.

The hypothesis is that replacing repeated reference VAE encoding with this lossless
cache preserves the generated result at a matching target geometry. This is not a
finished replacement for reference images, and it does not train or modify weights.

## PoC status

Implemented: all five nodes, preprocessing, format validation, lossless serialization,
safe filenames, and strict runtime geometry checks. Unit tests pass on CPU, and the
package imports with the real ComfyUI folder registry.

Real-model A/B validation is **pending**. Output equivalence has not yet been demonstrated
with a Krea2 checkpoint. See [validation status](VALIDATION.md) for the checks performed
and the remaining acceptance matrix. No unvalidated workflow JSON is distributed.

## How it works

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

## Using an IdentityMod

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
- No Qwen cache, embedded preview image, frontend extension, automatic model loading,
  model downloads, or sampling modifications.
- File geometry validation cannot identify which VAE weights were used. Use the same
  compatible VAE and model configuration for a valid comparison.
- Atomic non-overwriting saves require filesystem hard-link support (for example NTFS
  or ext4). Unsupported filesystems fail explicitly rather than overwriting an existing file.

## File format

Safetensors format `krea2_identitymod`, version `0.1.0`. The loader accepts stable `0.1.x`
versions and rejects other major/minor versions. New files contain exactly one tensor:

| Key | Representation |
| --- | --- |
| `appearance_latent` | CPU-contiguous floating-point raw VAE latent, `1 x C x H x W` |

Original dtype and values are preserved; no FP16 conversion or quantization occurs.
Metadata values are strings. Required fields are:

| Fields | Meaning or required value |
| --- | --- |
| `format`, `format_version` | `krea2_identitymod`, `0.1.0` when writing |
| `creator`, `model_family` | `ComfyUI-Krea2IdentityMod`, `krea2` |
| `identity_name`, `description` | User-supplied descriptive metadata |
| `purpose` | `appearance_reference` |
| `target_width`, `target_height` | Target pixel dimensions |
| `target_latent_width`, `target_latent_height` | Stored latent spatial dimensions |
| `latent_channels`, `latent_dtype` | Channel count and exact dtype, e.g. `torch.float32` |
| `source_width`, `source_height` | Original image dimensions |
| `preprocess_mode` | `full_target_grid_near_matched_fit` |
| `vae_downscale_factor`, `qwen_cache_present` | `8`, `false` |
| `created_at_utc` | ISO 8601 UTC creation timestamp |

The loader cross-checks metadata against tensor shape and dtype, rejects non-finite
latent values, and tolerates unknown metadata and tensor keys without using them.
It never deserializes Python objects. Reload detection uses resolved path, modification
time, creation/change time, and file size. A file replacement that preserves all of
these attributes is not detected by a content hash.

Saving accepts relative subfolders. Absolute paths, traversal, reserved Windows names,
and links escaping the storage root are rejected. Non-overwriting saves publish a
complete temporary file atomically under an available filename; overwrite uses atomic
replacement. Neither mode intentionally exposes a partial `.safetensors` file.

## A/B validation

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
| Poor Krea2 output quality | Ensure grounded encoding still receives the original image, Identity Edit LoRA is enabled, geometry matches, and `ref_boost` is intended. |
| Upstream `fit` warning | Use `crop (legacy)` for the cached full-grid latent. |
| Empty Load dropdown | Save/copy a file into `models/krea2_identitymods`, then refresh node definitions or restart ComfyUI. |
| File cannot be saved | Use a valid relative filename, check permissions, and use storage supporting atomic hard links. |
| Unsupported format or version | Select an IdentityMod `0.1.x` file, not a checkpoint or LoRA. |

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

1. Complete the real-model A/B matrix and confirm equivalence and absence of source VAE calls.
2. Export and validate creation and generation workflows from that environment.
3. Consider arbitrary reference geometry and multiple resolution buckets after this PoC passes.
4. Investigate Qwen visual-feature caching separately; it is outside v0.1.
