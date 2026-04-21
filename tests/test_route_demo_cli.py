import subprocess
import sys
import unittest


class RouteDemoCliTests(unittest.TestCase):
    def test_demo_route_recommendation_runs_on_sim_demo(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/demo_route_recommendation.py",
                "--data_cfg",
                "configs/data/sim_demo.yaml",
                "--source",
                "0",
                "--target",
                "23",
                "--candidate_count",
                "2",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        self.assertIn("Dataset: SimDemo", completed.stdout)
        self.assertIn("Reachability: reachable", completed.stdout)
        self.assertIn("候选路线:", completed.stdout)
        self.assertIn("路线 #1 推荐原因:", completed.stdout)

    def test_demo_route_reachability_only_mode(self):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/demo_route_recommendation.py",
                "--data_cfg",
                "configs/data/sim_demo.yaml",
                "--source",
                "0",
                "--target",
                "23",
                "--check_reachability",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        self.assertIn("Reachability: reachable", completed.stdout)
        self.assertIn("Path:", completed.stdout)
        self.assertNotIn("候选路线:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
