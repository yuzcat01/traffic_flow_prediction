import csv
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.services.route_service import RouteRecommendationService


class RouteRecommendationServiceTests(unittest.TestCase):
    @staticmethod
    def _workspace_tmp_root() -> Path:
        root = Path.cwd() / ".codex_tmp" / "unit_tests"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _write_route_graph(self) -> Path:
        graph_path = self._workspace_tmp_root() / f"route_graph_{uuid4().hex}.csv"
        with open(graph_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["from", "to", "cost"])
            writer.writerow([0, 1, 1.0])
            writer.writerow([1, 3, 1.0])
            writer.writerow([0, 2, 1.0])
            writer.writerow([2, 3, 1.0])
        return graph_path

    def test_congestion_strategy_avoids_high_risk_node(self):
        service = RouteRecommendationService(
            graph_path=str(self._write_route_graph()),
            num_nodes=4,
            flow_path=None,
        )
        prediction = np.array([0.2, 5.0, 0.2, 0.2], dtype=np.float32)

        result = service.recommend_route(
            prediction=prediction,
            source=0,
            target=3,
            strategy=RouteRecommendationService.STRATEGY_CONGESTION,
            alpha=1.0,
        )

        self.assertTrue(result["reachable"])
        self.assertEqual(result["path"], [0, 2, 3])
        self.assertEqual(result["high_risk_node_count"], 0)

    def test_top_risk_nodes_are_sorted_by_congestion_score(self):
        service = RouteRecommendationService(
            graph_path=str(self._write_route_graph()),
            num_nodes=4,
            flow_path=None,
        )
        prediction = np.array([0.5, 3.0, 1.5, 0.2], dtype=np.float32)

        risks = service.top_risk_nodes(prediction, topk=2)

        self.assertEqual([row["node_id"] for row in risks], [1, 2])
        self.assertEqual(risks[0]["risk_level"], "严重")

    def test_recommend_routes_returns_multiple_candidates(self):
        service = RouteRecommendationService(
            graph_path=str(self._write_route_graph()),
            num_nodes=4,
            flow_path=None,
        )

        result = service.recommend_routes(
            prediction=np.ones(4, dtype=np.float32),
            source=0,
            target=3,
            candidate_count=2,
        )

        self.assertTrue(result["reachable"])
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["candidates"][0]["path"][0], 0)
        self.assertEqual(result["candidates"][0]["path"][-1], 3)

    def test_network_preview_exposes_nodes_and_edges(self):
        service = RouteRecommendationService(
            graph_path=str(self._write_route_graph()),
            num_nodes=4,
            flow_path=None,
        )

        preview = service.get_network_preview()

        self.assertEqual(len(preview["nodes"]), 4)
        self.assertEqual(len(preview["edges"]), 4)
        self.assertIn("x", preview["nodes"][0])
        self.assertIn("y", preview["nodes"][0])

    def test_query_reachability_returns_minimum_hop_path(self):
        service = RouteRecommendationService(
            graph_path=str(self._write_route_graph()),
            num_nodes=4,
            flow_path=None,
        )

        result = service.query_reachability(0, 3)

        self.assertTrue(result["reachable"])
        self.assertEqual(result["hop_count"], 2)
        self.assertEqual(result["path"][0], 0)
        self.assertEqual(result["path"][-1], 3)

    def test_rejects_same_source_and_target(self):
        service = RouteRecommendationService(
            graph_path=str(self._write_route_graph()),
            num_nodes=4,
            flow_path=None,
        )

        with self.assertRaisesRegex(ValueError, "source and target"):
            service.recommend_route(
                prediction=np.ones(4, dtype=np.float32),
                source=1,
                target=1,
            )


if __name__ == "__main__":
    unittest.main()
