import unittest

import numpy as np

try:
    from src.services.predictor import TrafficPredictor
except ModuleNotFoundError:
    TrafficPredictor = None


@unittest.skipIf(TrafficPredictor is None, "PyTorch is not installed in the current environment")
class PredictorValidationTests(unittest.TestCase):
    def setUp(self):
        self.predictor = TrafficPredictor.__new__(TrafficPredictor)
        self.predictor.num_nodes = 3
        self.predictor.history_length = 4

    def test_check_and_format_window_accepts_2d_input(self):
        window = np.arange(12, dtype=np.float32).reshape(3, 4)
        formatted = self.predictor._check_and_format_window(window)
        self.assertEqual(formatted.shape, (3, 4, 1))

    def test_check_and_format_window_rejects_invalid_num_nodes(self):
        window = np.arange(8, dtype=np.float32).reshape(2, 4)
        with self.assertRaisesRegex(ValueError, "num_nodes mismatch"):
            self.predictor._check_and_format_window(window)

    def test_check_and_format_window_rejects_invalid_history_length(self):
        window = np.arange(15, dtype=np.float32).reshape(3, 5)
        with self.assertRaisesRegex(ValueError, "history_length mismatch"):
            self.predictor._check_and_format_window(window)


if __name__ == "__main__":
    unittest.main()
