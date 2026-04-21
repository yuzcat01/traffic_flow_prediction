import unittest
from pathlib import Path

import numpy as np

from src.services.route_service import RouteRecommendationService


class SimDemoDataTests(unittest.TestCase):
    def test_sim_demo_route_service_uses_generic_graph_and_flow_files(self):
        graph_path = Path("data/raw/sim_demo/sim_demo.csv")
        flow_path = Path("data/raw/sim_demo/sim_demo.npz")

        self.assertTrue(graph_path.exists())
        self.assertTrue(flow_path.exists())

        service = RouteRecommendationService(
            graph_path=str(graph_path),
            flow_path=str(flow_path),
            num_nodes=24,
        )
        flow = np.load(flow_path)["data"]
        prediction = flow[-1, :, 0][:, np.newaxis]

        preview = service.get_network_preview()
        result = service.recommend_routes(
            prediction=prediction,
            source=0,
            target=23,
            candidate_count=3,
        )

        self.assertEqual(len(preview["nodes"]), 24)
        self.assertEqual(len(preview["edges"]), 43)
        self.assertTrue(result["reachable"])
        self.assertGreaterEqual(len(result["candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
