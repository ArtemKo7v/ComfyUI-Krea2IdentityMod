# PoC v0.1 validation

## Current result

Implementation and unit validation are complete. The core hypothesis of equivalent
real-model generation is **not yet validated**.

Checks performed on Windows with Python 3.13.3, PyTorch 2.7.0+cu126, and safetensors 0.7.0:

- All 31 unit tests passed with no skips.
- Exact serialization roundtrip for float32, float16, bfloat16, and float64.
- Required metadata, invalid files, format versions, tensor dimensions, and dtype checks.
- Near-match boundaries, RGB/RGBA preprocessing, wrong VAE output, and one encode per creation.
- Full Create/Save/Load/To Latent/Info lifecycle; loaded handoff invokes no VAE or interpolation.
- Concurrent non-overwriting saves, overwrite failure recovery, and Windows junction containment.
- Package import and five node contracts with the real installed ComfyUI `folder_paths` module.

The package-import check above was performed before the 0.1.1 runtime-shape fix.

## Runtime-shape regression fixed in 0.1.1

The supplied `res/sam002.png` shows stripe artifacts. Its workflow uses a saved
4D IdentityMod reference. The supplied safetensors file passes format validation and
an exact in-memory serialization roundtrip. Same-size preprocessing of the supplied
1024x1024 image preserves every pixel value.

A code-level integration defect was reproduced: Krea2 uses
[Wan21 latent normalization](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/latent_formats.py),
whose channel means and standard deviations have shape `1 x 16 x 1 x 1 x 1`.
The original To Latent output `1 x 16 x H x W` broadcasts to `1 x 16 x 16 x H x W`.
The upstream edit patch's frame flattening and reference batch selection then use
incorrectly normalized channels. Krea2's choice of Wan21 is declared in
[supported_models.py](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/supported_models.py).

To Latent now restores T=1 before handoff, always emitting `1 x C x 1 x H x W`, even
when the target is 4D. Storage remains 4D and existing files need no recreation. This
corrects the runtime shape prescribed by the initial specification while retaining
its raw-value and lossless-storage requirements.

A regression test with distinct per-channel normalization constants failed before
the change and passes afterward for both 4D and 5D target latents. It verifies shape,
values, and correct per-channel normalization without importing ComfyUI internals.
This establishes the integration bug and its tensor-level fix, not yet the visual
result after rerunning the user's model.

The supplied A/B workflows also differ in additional reference inputs, Identity Edit
LoRA strength, the empty latent node, and the input image filename/hash. Use one
reference, identical images and settings, and the same target node for the next A/B run.

## Earlier environment check

Full ComfyUI startup through the available Python environment failed before node loading:
`comfy_aimdo.vram_buffer` was unavailable; `comfy_kitchen` also reported incompatible APIs.
The inspected ComfyUI installation and shared model folders contained no Krea2 checkpoint,
Identity Edit LoRA, or Qwen3-VL weights. These environment issues were not modified.

Consequently, UI visibility, real-model execution, end-to-end output equivalence, and
performance have not been verified. No exported workflow JSON is claimed to be validated.
Use the explicit connection diagrams in README until workflows can be exported and tested.

## Compatibility review

The full-grid preprocessing was checked against
[ComfyUI-Krea2Edit source](https://github.com/lbouaraba/comfyui-krea2edit/blob/main/__init__.py),
reviewed on 2026-09-15. The relevant behavior is the near-match branch of
`_fit_encode_image`: crop tolerance 0.08, center crop, float32 bicubic interpolation
with antialiasing, RGB selection, and clamp before VAE encoding.

One additional geometry guard is necessary: latent H/W must be even. The upstream
edit forward pads the target to its patch size before checking whether the legacy
source needs resizing. A matching but odd source grid can therefore be resized.
The [Krea2 model](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/krea2/model.py)
uses 2x2 latent patches. Requiring pixel dimensions divisible by 16 prevents this case.

This repository does not import those private helpers or implement a transformer forward.
Source review is not a substitute for real-model validation against the installed revisions.

## Required real-model matrix

Use one suitably matched reference for each geometry. All dimensions below are width x height.
Seeds are fixed suggestions; record replacements if different seeds are used.

| Target | Seed | Pixel/latent equality | Output comparison | No appearance VAE call in B |
| --- | --- | --- | --- | --- |
| 1024x1024 | 123 | Pending | Pending | Pending |
| 1024x1024 | 456 | Pending | Pending | Pending |
| 1024x1024 | 789 | Pending | Pending | Pending |
| 768x1024 | 123 | Pending | Pending | Pending |
| 768x1024 | 456 | Pending | Pending | Pending |
| 768x1024 | 789 | Pending | Pending | Pending |
| 1024x768 | 123 | Pending | Pending | Pending |
| 1024x768 | 456 | Pending | Pending | Pending |
| 1024x768 | 789 | Pending | Pending | Pending |

For each pair, record:

- ComfyUI, ComfyUI-Krea2Edit, and IdentityMod revisions.
- Checkpoint, LoRA, VAE, text encoder, and source image identifiers or hashes.
- LoRA strength, prompt, negative conditioning, seed, sampler, scheduler, steps, CFG,
  reference boost/mask, target size, and grounding resolution.
- Equality of preprocessed pixel tensors, raw VAE tensors, and saved/loaded tensors.
- Equality of sampler latents and decoded output tensors, plus maximum absolute
  difference if nonzero. Record hardware, dtype, and attention backend.
- Confirmation that B executes Load/To Latent without source VAE encoding and without
  upstream latent resizing. Keep VAE decoding and grounded image encoding in both branches.

Investigate nonzero differences before accepting perceptually equivalent output.
Once all rows pass, export the two runnable workflows as
`workflows/create_identitymod_poc.json` and `workflows/use_identitymod_poc.json`, reload
them in ComfyUI, execute them, and update this report with the actual evidence.
