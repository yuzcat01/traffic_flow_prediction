import unittest
from uuid import uuid4
from pathlib import Path

import numpy as np

from src.datasets.traffic_dataset import get_flow_data, resolve_preprocess_config


class DatasetPreprocessTests(unittest.TestCase):
    @staticmethod
    def _workspace_tmp_root() -> Path:
        root = Path.cwd() / ".codex_tmp" / "unit_tests"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _write_flow_npz(self, root: Path, data: np.ndarray) -> Path:
        flow_path = root / f"flow_{uuid4().hex}.npz"
        np.savez(flow_path, data=data.astype(np.float32))
        return flow_path

    def test_resolve_preprocess_aliases(self):
        cfg = resolve_preprocess_config({"missing_strategy": "ffill", "clip_min": 0, "clip_max_quantile": 0.95})
        self.assertEqual(cfg["missing_strategy"], "forward_fill")
        self.assertEqual(cfg["clip_min"], 0.0)
        self.assertAlmostEqual(cfg["clip_max_quantile"], 0.95)

    def test_get_flow_data_applies_missing_fill_and_clip(self):
        raw = np.array(
            [
                [[1.0], [2.0]],
                [[np.nan], [5.0]],
                [[9.0], [50.0]],
                [[4.0], [7.0]],
            ],
            dtype=np.float32,
        )  # [T, N, D]

        flow_path = self._write_flow_npz(self._workspace_tmp_root(), raw)
        flow_data, stats = get_flow_data(
            str(flow_path),
            preprocess_cfg={
                "missing_strategy": "mean_fill",
                "clip_min": 0.0,
                "clip_max_quantile": 0.8,
            },
            return_stats=True,
        )

        self.assertEqual(flow_data.shape, (2, 4, 1))
        self.assertEqual(stats["missing_value_count_before"], 1)
        self.assertEqual(stats["missing_value_count_after"], 0)
        self.assertTrue(np.isfinite(flow_data).all())
        self.assertGreater(stats["clipped_value_count"], 0)
        self.assertIsNotNone(stats["clip_max_value"])


if __name__ == "__main__":
    unittest.main()
