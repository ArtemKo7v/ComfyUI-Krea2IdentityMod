# IdentityMod validation

## PoC v0.2 implementation result (2026-09-15)

The appearance baseline is frozen. This change adds the prompt-independent Qwen
cache and the image-free grounded encoder, not a new appearance algorithm.

Verified locally with Python 3.13.3, PyTorch 2.7.0+cu126, safetensors 0.7.0:

- All **53 CPU unit tests pass**, including the original 31 tests.
- A frozen, base64-encoded v0.1 safetensors fixture loads and roundtrips unchanged.
- Complete v0.2 roundtrip is exact for float16, bfloat16, float32, and float64.
- Missing/extra cache tensors, invalid grid, wrong DeepStack count, invalid dtypes,
  metadata mismatches, and contradictory version/cache flags are rejected.
- Fake extraction calls preprocessing and the vision boundary once; language encoding
  is not called during creation. Returned data is detached, CPU-owned, and contiguous.
- Scoped injection delegates ordinary descriptors and restores the prior method on
  success, inference failure, nested scopes, and serialized concurrent use.
- Both new nodes pass a Save/Load/To Latent/Info/positive/negative lifecycle test.
- Loaded cache reuse across prompts and system prompts, and a vision-raises guard,
  pass with API doubles. These tests **do not prove actual Qwen numerical parity**.
- The opt-in `validation.run_qwen_ab` matrix driver is itself tested with doubles.

The local standalone PyTorch build reports the installed GPU's `sm_120` capability
as unsupported; an attempted GPU-only tensor check failed with no compatible kernel.
The completed unit suite is deliberately CPU-only. No environment was changed, and
the user's installed ComfyUI files, models, API, and history were not accessed for v0.2.

The existing user-supplied `res/identity_mod_sagalova.safetensors` was preserved:
SHA-256 `bca4519e830616ae468a6c2d180bcd33708bb81639321463f770f5c3cf3c9943`.
The portable synthetic fixture in `tests/fixtures` avoids depending on private images
or untracked `res` files in the unit suite.

## Accepted v0.1 baseline

The v0.2 task records successful direct-vs-cached appearance validation. The supplied
`sam004` pair was pixel-identical for direct and cached VAE decoding; the supplied
`sam003`/`sam005` pair was also pixel-identical for the compared generated outputs.
This supersedes the earlier pending status below for those tested cases, not for
every geometry or model configuration. The 5D handoff fix remains covered by tests.
The texture investigation is closed and was not reopened for this implementation.

## Pending v0.2 real-model acceptance

Use the same image, actual Krea2 Qwen3-VL 4B checkpoint/quantization, software revisions,
device, attention backend, and prompt/system prompt in both paths. Record these names
alongside results; no checkpoint fingerprint is embedded in the cache.

| Grounding cap | Short / scene-change / long / empty prompts | Default / custom system | Vision-raises guard | Timing |
| --- | --- | --- | --- | --- |
| 512 | Pending | Pending | Pending | Pending |
| 768 | Pending | Pending | Pending | Pending |
| 1024 | Pending | Pending | Pending | Pending |

For each row create and save/load **one** cache and reuse it across all prompts.
Compare every scheduled conditioning tensor and masks: shape, dtype, max absolute
error, mean absolute error, `torch.equal`, and `torch.allclose`. The helper reports
diagnostic `atol=1e-6, rtol=1e-5`; this is not a pre-approved acceptance tolerance.
Investigate every non-exact result and document any justified tolerance explicitly.

`validation.run_qwen_ab(clip, appearance, image, upstream_node.encode)` accepts live
objects and the **actual upstream** node method; it does not reproduce the reference
implementation. It returns JSON-ready records, including synchronized wall-clock
times and evidence that disabling the real tower still permits cached encode while
blocking direct encode. Run only in an otherwise idle process: instrumentation is
temporary but shared CLIP models are not isolated by cloning. The helper has not been
run here against actual weights. It is a developer helper, not a registered UI node.

Then build the generation workflow described in README with **no source Load Image**,
no creation nodes, and no source VAE Encode. Use the same loaded full IdentityMod
for appearance, positive, and empty negative; retain output VAE decoding. Queue a
full generation and record the result. Export a workflow JSON only after that graph
has actually run successfully. This product-level test remains pending.

Do not claim v0.2 accepted or faster until those tests are performed. GPU kernel
behavior, installed-version integration, real conditioning parity, and full image-free
sampling cannot be established with the CPU/API-double tests.

## v0.2 compatibility sources

Public upstream sources were reviewed on 2026-09-15; local installation files were not used:

- [Qwen3-VL preprocessing and DeepStack](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/text_encoders/qwen3vl.py).
- [Krea2 4B encoder and 12 taps](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/text_encoders/krea2.py).
- [Generic token processing](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/sd1_clip.py)
  and [CLIP model loading/scheduled encode](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/sd.py).
- [Krea2Edit grounded template and area downscaling](https://github.com/lbouaraba/comfyui-krea2edit/blob/main/__init__.py).

These are moving upstream branches, not a guarantee of compatibility with every
installed revision. Model/schema discovery fails explicitly when the expected API is absent.

## Historical v0.1 implementation report (superseded where noted above)

### Result at initial v0.1 implementation

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

### Runtime-shape regression fixed in 0.1.1

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

### Earlier environment check

Full ComfyUI startup through the available Python environment failed before node loading:
`comfy_aimdo.vram_buffer` was unavailable; `comfy_kitchen` also reported incompatible APIs.
The inspected ComfyUI installation and shared model folders contained no Krea2 checkpoint,
Identity Edit LoRA, or Qwen3-VL weights. These environment issues were not modified.

Consequently, UI visibility, real-model execution, end-to-end output equivalence, and
performance have not been verified. No exported workflow JSON is claimed to be validated.
Use the explicit connection diagrams in README until workflows can be exported and tested.

### Compatibility review

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

### Originally proposed v0.1 real-model matrix

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
