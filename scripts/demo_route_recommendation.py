import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.datasets.traffic_dataset import get_flow_data
from src.services.route_service import RouteRecommendationService
from src.utils.config import load_yaml
from src.project_paths import resolve_project_path


def _format_horizon(horizon_idx: int, time_interval: int) -> str:
    steps = int(horizon_idx) + 1
    minutes = steps * int(time_interval)
    if minutes >= 60 and minutes % 60 == 0:
        return f"h{steps} / future {minutes // 60} hour(s)"
    if minutes >= 60:
        return f"h{steps} / future {minutes / 60.0:.1f} hours"
    return f"h{steps} / future {minutes} minutes"


def main():
    parser = argparse.ArgumentParser(
        description="Demo prediction-driven route recommendation on a generic traffic dataset."
    )
    parser.add_argument("--data_cfg", default="configs/data/sim_demo.yaml")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=23)
    parser.add_argument("--horizon", type=int, default=-1, help="-1 means the farthest available horizon")
    parser.add_argument("--candidate_count", type=int, default=3)
    parser.add_argument(
        "--strategy",
        default=RouteRecommendationService.STRATEGY_BALANCED,
        choices=[
            RouteRecommendationService.STRATEGY_DISTANCE,
            RouteRecommendationService.STRATEGY_CONGESTION,
            RouteRecommendationService.STRATEGY_BALANCED,
        ],
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--check_reachability", action="store_true", help="Only check graph reachability and exit")
    args = parser.parse_args()

    cfg = load_yaml(resolve_project_path(args.data_cfg))
    dataset_cfg = cfg["dataset"]
    graph_path = dataset_cfg["graph_path"]
    flow_path = dataset_cfg["flow_path"]
    num_nodes = int(dataset_cfg["num_nodes"])
    time_interval = int(dataset_cfg.get("time_interval", 5))
    preprocess_cfg = dataset_cfg.get("preprocess", {})

    flow_data = get_flow_data(
        str(resolve_project_path(flow_path)),
        preprocess_cfg=preprocess_cfg,
    )  # [N, T, 1]

    # This demo uses the latest observed flow as a stand-in prediction window.
    # In the GUI, the same route service receives model predictions instead.
    latest_flow = flow_data[:, -1, 0]
    prediction = np.repeat(latest_flow[:, np.newaxis], 3, axis=1)
    horizon_idx = prediction.shape[1] - 1 if args.horizon < 0 else max(0, min(args.horizon, prediction.shape[1] - 1))

    service = RouteRecommendationService(
        graph_path=graph_path,
        flow_path=flow_path,
        num_nodes=num_nodes,
        preprocess_cfg=preprocess_cfg,
    )
    preview = service.get_network_preview()
    reachability = service.query_reachability(args.source, args.target)

    if args.check_reachability:
        print(f"Dataset: {dataset_cfg.get('name', 'Unknown')}")
        print(f"Network: {len(preview['nodes'])} nodes, {len(preview['edges'])} edges")
        print(f"Source -> target: {args.source} -> {args.target}")
        if reachability["reachable"]:
            print(f"Reachability: reachable, minimum hops={reachability['hop_count']}")
            print(f"Path: {reachability['path_text']}")
        else:
            print("Reachability: not reachable")
        return

    result = service.recommend_routes(
        prediction=prediction,
        source=args.source,
        target=args.target,
        horizon_idx=horizon_idx,
        strategy=args.strategy,
        alpha=args.alpha,
        topk=args.topk,
        candidate_count=args.candidate_count,
    )

    print(f"Dataset: {dataset_cfg.get('name', 'Unknown')}")
    print(f"Network: {len(preview['nodes'])} nodes, {len(preview['edges'])} edges")
    print(f"Source -> target: {args.source} -> {args.target}")
    if reachability["reachable"]:
        print(f"Reachability: reachable, minimum hops={reachability['hop_count']}")
    else:
        print("Reachability: not reachable")
    print(f"Horizon: {_format_horizon(horizon_idx, time_interval)}")
    print(f"Strategy: {args.strategy}, alpha={args.alpha:.2f}")
    print()

    if not result.get("reachable"):
        print(result.get("message", "No reachable path was found."))
        return

    print("候选路线:")
    for route in result.get("candidates", []):
        print(
            f"  #{route['route_rank']}: {route['path_text']} | "
            f"distance={route['distance']:.2f}, "
            f"avg_congestion={route['avg_congestion_score']:.3f}, "
            f"max_congestion={route['max_congestion_score']:.3f}, "
            f"high_risk_nodes={route['high_risk_node_count']}"
        )

    selected = result.get("candidates", [result])[0]
    print()
    print("路线 #1 推荐原因:")
    for item in RouteRecommendationService.explain_route(selected):
        print(f"  - {item}")

    print()
    print("高风险节点 Top-K:")
    for row in result.get("top_risk_nodes", []):
        print(
            f"  #{row['rank']} node={row['node_id']} "
            f"score={row['congestion_score']:.3f} "
            f"flow={row['predicted_flow']:.2f} "
            f"level={row['risk_level']}"
        )


if __name__ == "__main__":
    main()
