"""Small API doubles; these do not model real Qwen attention or prove model parity."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.support import make_mod
from artemko7v_identitymod_test.constants import KREA2_QWEN_TAP_LAYERS
from artemko7v_identitymod_test.identity_mod import (
    ArtemKo7vKrea2IdentityModData, ArtemKo7vKrea2QwenVisionCache,
)


def make_cache(dtype=torch.float32):
    merged = torch.arange(4 * 2560, dtype=torch.float32).reshape(4, 2560).div(1024).to(dtype)
    return ArtemKo7vKrea2QwenVisionCache(
        merged, torch.tensor([[1, 4, 4]]), tuple(merged + i for i in (1, 2, 3))
    )


def make_full_mod(dtype=torch.float32):
    appearance = make_mod(dtype)
    cache = make_cache(dtype)
    metadata = dict(appearance.metadata, **cache.tensor_metadata())
    metadata.update(format_version="0.2.0", qwen_grounding_px="768",
                    qwen_input_width="64", qwen_input_height="64")
    return ArtemKo7vKrea2IdentityModData(appearance.reference_latent, metadata, cache)


class ArtemKo7vFakeQwen(torch.nn.Module):
    """Produce deterministic visual values with real schema shapes."""

    model_type = "qwen3vl_4b"

    def __init__(self):
        super().__init__()
        self.preprocess_calls = 0
        self.image = None

    def visual(self, image):
        cache = make_cache()
        delta = image.mean()
        return cache.merged + delta, [tensor + delta for tensor in cache.deepstack]

    def preprocess_embed(self, embed, device):
        self.preprocess_calls += 1
        self.image = embed["data"].clone()
        merged, deepstack = self.visual(embed["data"])
        self.last_output = (merged.to(device), {"grid": torch.tensor([[1, 4, 4]]),
                                               "deepstack": [tensor.to(device) for tensor in deepstack]})
        return self.last_output


class ArtemKo7vFakeClip:
    """Minimal tokenizer/encoder stand-in for testing scoped feature delivery."""

    def __init__(self):
        self.transformer = ArtemKo7vFakeQwen()
        self.cond_stage_model = SimpleNamespace(qwen3vl_4b=SimpleNamespace(
            transformer=self.transformer, layer=list(KREA2_QWEN_TAP_LAYERS)))
        self.patcher = SimpleNamespace(load_device=torch.device("cpu"))
        self.loads = 0
        self.encodes = 0

    def load_model(self):
        self.loads += 1

    def tokenize(self, prompt, images, llama_template):
        text = llama_template.format(prompt)
        parts = text.split("<|image_pad|>")
        row = []
        for index, part in enumerate(parts):
            row.extend((ord(char), 1.0) for char in part)
            if index < len(parts) - 1:
                token = ({"type": "image", "data": images[index], "original_type": "image"}
                         if index < len(images) else 151655)
                row.append((token, 1.0))
        return {"qwen3vl_4b": [row]}

    def encode_from_tokens_scheduled(self, tokens):
        self.encodes += 1
        text_sum = 0
        result = None
        for token, weight in tokens["qwen3vl_4b"][0]:
            if isinstance(token, dict):
                merged, extra = self.transformer.preprocess_embed(token, self.patcher.load_device)
                result = merged.float() + sum(extra["deepstack"]).float() + extra["grid"].sum()
            else:
                text_sum += token
        return [[result.unsqueeze(0) + text_sum / 1000, {"attention_mask": torch.ones(1, 4)}]]


def install_model_management_stub(test_case):
    stub = SimpleNamespace(model_management=SimpleNamespace(cuda_device_context=nullcontext))
    binding = patch.dict("sys.modules", {"comfy": stub})
    binding.start()
    test_case.addCleanup(binding.stop)
