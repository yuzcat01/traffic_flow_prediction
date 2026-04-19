import csv
import unittest
from uuid import uuid4
from pathlib import Path

import numpy as np

from src.datasets.traffic_dataset import build_adjacency_matrix


class GraphBuildingTests(unittest.TestCase):
    @staticmethod
    def _workspace_tmp_root() -> Path:
        root = Path.cwd() / ".codex_tmp" / "unit_tests"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _write_distance_csv(self, root: Path) -> Path:
        graph_path = root / f"graph_{uuid4().hex}.csv"
        with open(graph_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["from", "to", "distance"])
            writer.writerow([0, 1, 10.0])
            writer.writerow([1, 2, 5.0])
        return graph_path

    def test_build_correlation_adjacency_is_symmetric(self):
        flow_data = np.array(
            [
                [[1.0], [2.0], [3.0], [4.0]],
                [[1.0], [2.0], [3.0], [4.0]],
                [[4.0], [3.0], [2.0], [1.0]],
            ],
            dtype=np.float32,
        )

        graph_path = self._write_distance_csv(self._workspace_tmp_root())
        adj = build_adjacency_matrix(
            distance_file=str(graph_path),
            num_nodes=3,
            graph_cfg={
                "type": "correlation",
                "correlation_topk": 2,
                "correlation_threshold": 0.1,
                "use_abs_corr": False,
            },
            flow_data=flow_data,
        )

        self.assertEqual(adj.shape, (3, 3))
        self.assertTrue(np.allclose(adj, adj.T))
        self.assertTrue(np.all(np.diag(adj) == 0.0))

    def test_build_distance_correlation_adjacency_returns_nonzero_weights(self):
        flow_data = np.array(
            [
                [[1.0], [2.0], [3.0], [4.0]],
                [[1.0], [2.0], [3.0], [4.0]],
                [[2.0], [2.1], [1.9], [2.0]],
            ],
            dtype=np.float32,
        )

        graph_path = self._write_distance_csv(self._workspace_tmp_root())
        adj = build_adjacency_matrix(
            distance_file=str(graph_path),
            num_nodes=3,
            graph_cfg={
                "type": "distance_correlation",
                "correlation_topk": 2,
                "correlation_threshold": 0.0,
                "fusion_alpha": 0.5,
            },
            flow_data=flow_data,
        )

        self.assertEqual(adj.shape, (3, 3))
        self.assertGreater(int(np.count_nonzero(adj)), 0)


if __name__ == "__main__":
    unittest.main()
