"""Check the opt-in A/B harness with API doubles, not actual model weights."""

import unittest

import torch

from tests.support import make_mod
from tests.qwen_support import ArtemKo7vFakeClip, install_model_management_stub
from artemko7v_identitymod_test.qwen_injection import _grounding_template
from artemko7v_identitymod_test.qwen_cache import _prepare_grounding_image
from artemko7v_identitymod_test.validation import run_qwen_ab, _compare_tensors


class ArtemKo7vValidationTests(unittest.TestCase):
    """Ensure reports expose discrepancies and the guard is actually tested."""

    def test_matrix_driver(self):
        install_model_management_stub(self)
        clip = ArtemKo7vFakeClip()

        def direct_encode(clip, prompt, image, grounding_px, system_prompt):
            tokens = clip.tokenize(prompt, images=[_prepare_grounding_image(image, grounding_px)],
                                   llama_template=_grounding_template(system_prompt))
            return (clip.encode_from_tokens_scheduled(tokens),)

        report = run_qwen_ab(clip, make_mod(), torch.ones(1, 64, 64, 3), direct_encode)
        self.assertEqual(len(report), 24)
        self.assertEqual(clip.loads, 3)
        for row in report:
            self.assertTrue(all(metric["equal"] for metric in row["tensors"]))
            self.assertTrue(row["direct_fails_with_vision_disabled"])
            self.assertTrue(row["cached_succeeds_with_vision_disabled"])
        self.assertNotIn("preprocess_embed", vars(clip.transformer))

    def test_metrics_detect_differences(self):
        report = _compare_tensors(torch.ones(2), torch.zeros(2), 1e-6, 1e-5)
        self.assertFalse(report["equal"])
        self.assertFalse(report["allclose"])
        self.assertEqual(report["max_absolute_error"], 1.0)
        report = _compare_tensors(torch.ones(2), torch.ones(3), 1e-6, 1e-5)
        self.assertFalse(report["equal"])
        self.assertIsNone(report["mean_absolute_error"])
