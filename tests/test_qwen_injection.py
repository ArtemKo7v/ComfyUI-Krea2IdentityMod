"""Feature delivery, prompt reuse, isolation, and guaranteed method restoration."""

from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

import torch

from tests.support import make_mod
from tests.qwen_support import make_cache, make_full_mod, ArtemKo7vFakeClip, install_model_management_stub
from artemko7v_identitymod_test.identity_mod import ArtemKo7vKrea2IdentityModError
from artemko7v_identitymod_test.qwen_cache import _add_qwen_vision_cache
from artemko7v_identitymod_test.qwen_injection import (
    _CACHE_MARKER, _cached_preprocess, _cached_tokens, _encode_cached_grounded, _grounding_template,
)
from artemko7v_identitymod_test.serialization import _load_identity_mod, _serialize_identity_mod


class ArtemKo7vQwenInjectionTests(unittest.TestCase):
    """No real tokenizer/attention claims: exercise the adapter's public API contracts."""

    def test_only_matching_image_descriptor_skips_original(self):
        transformer = ArtemKo7vFakeClip().transformer
        original = transformer.preprocess_embed
        cache = make_cache()
        marker = object()
        with _cached_preprocess(transformer, cache, marker):
            merged, extra = transformer.preprocess_embed(
                {"type": "image", _CACHE_MARKER: marker}, device="cpu")
            self.assertEqual(transformer.preprocess_calls, 0)
            self.assertTrue(torch.equal(merged, cache.merged))
            self.assertTrue(torch.equal(extra["grid"], cache.grid))
            for actual, expected in zip(extra["deepstack"], cache.deepstack):
                self.assertTrue(torch.equal(actual, expected))
            merged.zero_()
            extra["grid"].zero_()
            extra["deepstack"][0].zero_()
            cache.validate()
            self.assertTrue(torch.equal(cache.merged, make_cache().merged))
            transformer.preprocess_embed({"type": "image", "data": torch.ones(1, 64, 64, 3)}, "cpu")
            transformer.preprocess_embed(
                {"type": "image", _CACHE_MARKER: object(), "data": torch.ones(1, 64, 64, 3)}, "cpu")
            self.assertEqual(transformer.preprocess_calls, 2)
        self.assertEqual(transformer.preprocess_embed, original)
        self.assertNotIn("preprocess_embed", vars(transformer))

    def test_restore_after_exception_and_nested_interception(self):
        transformer = ArtemKo7vFakeClip().transformer
        outer_marker, inner_marker = object(), object()
        original = transformer.preprocess_embed
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with _cached_preprocess(transformer, make_cache(), outer_marker):
                outer = transformer.preprocess_embed
                with _cached_preprocess(transformer, make_cache(torch.float64), inner_marker):
                    inner, _ = transformer.preprocess_embed(
                        {"type": "image", _CACHE_MARKER: inner_marker}, "cpu")
                    delegated, _ = transformer.preprocess_embed(
                        {"type": "image", _CACHE_MARKER: outer_marker}, "cpu")
                    self.assertEqual(inner.dtype, torch.float64)
                    self.assertEqual(delegated.dtype, torch.float32)
                self.assertIs(transformer.preprocess_embed, outer)
                raise RuntimeError("intentional")
        self.assertEqual(transformer.preprocess_embed, original)
        self.assertNotIn("preprocess_embed", vars(transformer))

    def test_restore_preexisting_instance_override(self):
        transformer = ArtemKo7vFakeClip().transformer
        override = lambda embed, device: (None, None)
        transformer.preprocess_embed = override
        with _cached_preprocess(transformer, make_cache(), object()):
            pass
        self.assertIs(transformer.preprocess_embed, override)

    def test_concurrent_scopes_are_serialized(self):
        transformer = ArtemKo7vFakeClip().transformer
        first_entered, release_first, second_entered, second_started = (Event() for _ in range(4))

        def first():
            with _cached_preprocess(transformer, make_cache(), object()):
                first_entered.set()
                if not release_first.wait(5):
                    raise AssertionError("First scope was not released")

        def second():
            second_started.set()
            with _cached_preprocess(transformer, make_cache(), object()):
                second_entered.set()

        with ThreadPoolExecutor(max_workers=2) as pool:
            one = pool.submit(first)
            self.assertTrue(first_entered.wait(5))
            two = pool.submit(second)
            try:
                self.assertTrue(second_started.wait(5))
                self.assertFalse(second_entered.wait(0.05))
            finally:
                release_first.set()
            one.result(timeout=5)
            two.result(timeout=5)
        self.assertTrue(second_entered.is_set())
        self.assertNotIn("preprocess_embed", vars(transformer))

    def test_placeholder_validation(self):
        clip = ArtemKo7vFakeClip()
        tokens, marker = _cached_tokens(clip, "", "")
        descriptors = [t[0] for t in tokens["qwen3vl_4b"][0] if isinstance(t[0], dict)]
        self.assertEqual(len(descriptors), 1)
        self.assertIs(descriptors[0][_CACHE_MARKER], marker)
        self.assertEqual(descriptors[0]["type"], "image")
        self.assertNotIn("data", descriptors[0])
        with self.assertRaisesRegex(ArtemKo7vKrea2IdentityModError, "exactly one"):
            _cached_tokens(clip, "extra <|image_pad|>", "")
        with patch.object(clip, "tokenize", return_value={"qwen3vl_4b": [[(42, 1.0)]]}):
            with self.assertRaisesRegex(ArtemKo7vKrea2IdentityModError, "exactly one"):
                _cached_tokens(clip, "test", "")
        with patch.object(clip, "tokenize", return_value={"other": []}):
            with self.assertRaisesRegex(ArtemKo7vKrea2IdentityModError, "Unsupported token layout"):
                _cached_tokens(clip, "test", "")

    def test_missing_cache_never_falls_back(self):
        clip = ArtemKo7vFakeClip()
        with self.assertRaisesRegex(ArtemKo7vKrea2IdentityModError, "Add Qwen Vision Cache"):
            _encode_cached_grounded(clip, make_mod(), "test")
        self.assertEqual(clip.encodes, 0)

    def test_stock_encoder_exception_restores_method(self):
        clip = ArtemKo7vFakeClip()
        with patch.object(clip, "encode_from_tokens_scheduled", side_effect=RuntimeError("inference")):
            with self.assertRaisesRegex(RuntimeError, "inference"):
                _encode_cached_grounded(clip, make_full_mod(), "test")
        self.assertNotIn("preprocess_embed", vars(clip.transformer))

    def test_template_whitespace_and_custom_system(self):
        self.assertEqual(_grounding_template(""), _grounding_template(" \n "))
        self.assertEqual(_grounding_template("  custom\nline  "),
                         "<|im_start|>system\ncustom\nline<|im_end|>\n<|im_start|>user\n"
                         "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
                         "<|im_start|>assistant\n")

    def test_loaded_cache_reused_across_prompts_without_vision(self):
        install_model_management_stub(self)
        clip = ArtemKo7vFakeClip()
        image = torch.rand(1, 64, 64, 3)
        full = _add_qwen_vision_cache(make_mod(), clip, image, 768)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full.safetensors"
            path.write_bytes(_serialize_identity_mod(full))
            loaded = _load_identity_mod(path)
        outputs = []
        for system in ("", "Follow this detailed editing instruction."):
            for prompt in (
                "Change her outfit to a black leather jacket.",
                "Put her in a snowy mountain environment.",
                "Make this a cinematic night portrait.", "",
            ):
                tokens = clip.tokenize(prompt, images=[image], llama_template=_grounding_template(system))
                direct = clip.encode_from_tokens_scheduled(tokens)
                with patch.object(clip.transformer, "visual", side_effect=RuntimeError("Vision tower should not run")):
                    cached = _encode_cached_grounded(clip, loaded, prompt, system)
                    with self.assertRaisesRegex(RuntimeError, "Vision tower should not run"):
                        clip.encode_from_tokens_scheduled(tokens)
                self.assertTrue(torch.equal(direct[0][0], cached[0][0]))
                self.assertTrue(torch.equal(direct[0][1]["attention_mask"], cached[0][1]["attention_mask"]))
                outputs.append(cached[0][0])
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
