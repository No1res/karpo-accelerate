from importlib.util import find_spec
from unittest import TestCase, skipUnless


@skipUnless(find_spec("torch") is not None, "requires the optional PyTorch runtime")
class CalibrationTests(TestCase):
    def test_linear_fit_with_both_solver_shapes(self):
        import torch

        from karpo_accel.calibrate import fit_linear

        features = torch.tensor([[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]])
        for x in (features, torch.nn.functional.pad(features, (0, 4))):
            weight, bias = fit_linear(x, [0, 1, 0, 1], 2)
            self.assertEqual((x @ weight + bias).argmax(dim=1).tolist(), [0, 1, 0, 1])

    def test_priority_threshold_uses_false_positives(self):
        from karpo_accel.calibrate import thresholds

        scores = [("harmful", 0.8), ("harmful", 0.4), ("safe", 0.5)]
        labels = ["harmful", "safe", "safe"]
        fallback = ["safe", "safe", "safe"]
        choices = ["safe", "harmful", "other"]
        usual = thresholds(scores, fallback, labels, choices, set(), 0.02)
        priority = thresholds(scores, fallback, labels, choices, {"harmful"}, 0.02)
        self.assertAlmostEqual(usual[1], 0.82)
        self.assertAlmostEqual(priority[1], 0.42)
        self.assertEqual(usual[0], priority[0])
        self.assertIsNone(usual[2])
