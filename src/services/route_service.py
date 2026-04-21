import csv
import heapq
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.datasets.traffic_dataset import get_flow_data, resolve_preprocess_config
from src.project_paths import get_project_root, resolve_project_path


class RouteRecommendationService:
    """Prediction-driven congestion scoring and graph route recommendation."""

    STRATEGY_DISTANCE = "distance"
    STRATEGY_CONGESTION = "congestion"
    STRATEGY_BALANCED = "balanced"

    def __init__(
        self,
        graph_path: str,
        num_nodes: int,
        flow_path: Optional[str] = None,
        preprocess_cfg: Optional[Dict[str, Any]] = None,
        project_root: Optional[str] = None,
        baseline_quantile: float = 0.85,
        bidirectional: bool = True,
    ):
        self.project_root = get_project_root(project_root)
        self.graph_path = Path(resolve_project_path(graph_path, self.project_root))
        self.flow_path = Path(resolve_project_path(flow_path, self.project_root)) if flow_path else None
        self.num_nodes = int(num_nodes)
        self.preprocess_cfg = resolve_preprocess_config(preprocess_cfg)
        self.baseline_quantile = float(baseline_quantile)
        self.bidirectional = bool(bidirectional)

        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be > 0")
        if not 0.0 < self.baseline_quantile <= 1.0:
            raise ValueError("baseline_quantile must be in (0, 1]")

        self.edges = self._load_edges(self.graph_path, self.num_nodes, bidirectional=self.bidirectional)
        self.display_edges = self._dedupe_display_edges(self.edges)
        self.graph = self._build_graph(self.edges)
        self.node_baseline = self._build_node_baseline()
        self.node_positions = self._build_network_layout()

    @staticmethod
    def _load_edges(graph_path: Path, num_nodes: int, bidirectional: bool = True) -> List[Dict[str, float]]:
        if not graph_path.exists():
            raise FileNotFoundError(f"graph file not found: {graph_path}")

        edges: List[Dict[str, float]] = []
        with open(graph_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    source = int(row.get("from", row.get("source", "")))
                    target = int(row.get("to", row.get("target", "")))
                    cost_raw = row.get("cost", row.get("distance", row.get("weight", "1")))
                    distance = float(cost_raw)
                except (TypeError, ValueError):
                    continue

                if source < 0 or target < 0 or source >= num_nodes or target >= num_nodes:
                    continue
                if distance <= 0:
                    continue

                edges.append({"from": source, "to": target, "distance": distance})
                if bidirectional and source != target:
                    edges.append({"from": target, "to": source, "distance": distance})

        if not edges:
            raise ValueError(f"no valid edges found in graph file: {graph_path}")
        return edges

    @staticmethod
    def _build_graph(edges: List[Dict[str, float]]) -> Dict[int, List[Tuple[int, float]]]:
        graph: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for edge in edges:
            graph[int(edge["from"])].append((int(edge["to"]), float(edge["distance"])))
        return dict(graph)

    @staticmethod
    def _dedupe_display_edges(edges: List[Dict[str, float]]) -> List[Dict[str, float]]:
        deduped: Dict[Tuple[int, int], float] = {}
        for edge in edges:
            source = int(edge["from"])
            target = int(edge["to"])
            if source == target:
                continue
            key = (min(source, target), max(source, target))
            distance = float(edge["distance"])
            deduped[key] = min(distance, deduped.get(key, distance))

        return [
            {"from": source, "to": target, "distance": distance}
            for (source, target), distance in sorted(deduped.items())
        ]

    def _build_network_layout(self, iterations: int = 80) -> np.ndarray:
        rng = np.random.default_rng(42)
        angles = np.linspace(0.0, 2.0 * np.pi, self.num_nodes, endpoint=False)
        degree = np.zeros(self.num_nodes, dtype=np.float32)
        for edge in self.display_edges:
            degree[int(edge["from"])] += 1.0
            degree[int(edge["to"])] += 1.0

        degree_scale = 1.0 + degree / max(float(np.max(degree)), 1.0)
        radius = 0.45 + 0.20 * (1.0 / degree_scale)
        pos = np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius]).astype(np.float32)
        pos += rng.normal(0.0, 0.015, size=pos.shape).astype(np.float32)

        if self.num_nodes <= 1 or not self.display_edges:
            return pos

        area = 1.0
        k = np.sqrt(area / float(self.num_nodes))
        edge_pairs = np.array(
            [[int(edge["from"]), int(edge["to"])] for edge in self.display_edges],
            dtype=np.int64,
        )
        temperature = 0.12

        for _ in range(max(0, int(iterations))):
            disp = np.zeros_like(pos, dtype=np.float32)

            delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            distance = np.sqrt(np.sum(delta * delta, axis=2) + 1e-4)
            repulsive = (k * k) / distance
            disp += np.sum(delta / distance[:, :, np.newaxis] * repulsive[:, :, np.newaxis], axis=1)

            for source, target in edge_pairs:
                delta_edge = pos[source] - pos[target]
                dist = float(np.sqrt(np.sum(delta_edge * delta_edge)) + 1e-4)
                attractive = (dist * dist) / k
                direction = delta_edge / dist
                disp[source] -= direction * attractive
                disp[target] += direction * attractive

            lengths = np.sqrt(np.sum(disp * disp, axis=1)) + 1e-6
            pos += (disp / lengths[:, np.newaxis]) * np.minimum(lengths, temperature)[:, np.newaxis]
            pos = np.clip(pos, -1.0, 1.0)
            temperature *= 0.94

        min_xy = np.min(pos, axis=0)
        max_xy = np.max(pos, axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        pos = (pos - min_xy) / span
        return pos.astype(np.float32)

    def _build_node_baseline(self) -> np.ndarray:
        if self.flow_path is not None and self.flow_path.exists():
            flow_data = get_flow_data(str(self.flow_path), preprocess_cfg=self.preprocess_cfg)  # [N, T, 1]
            usable = flow_data[: self.num_nodes, :, 0].astype(np.float32)
            baseline = np.quantile(usable, self.baseline_quantile, axis=1).astype(np.float32)
        else:
            baseline = np.ones(self.num_nodes, dtype=np.float32)

        baseline = np.nan_to_num(baseline, nan=1.0, posinf=1.0, neginf=1.0)
        baseline = np.maximum(baseline, 1.0)
        if baseline.shape[0] < self.num_nodes:
            padded = np.ones(self.num_nodes, dtype=np.float32)
            padded[: baseline.shape[0]] = baseline
            baseline = padded
        return baseline[: self.num_nodes].astype(np.float32)

    @staticmethod
    def classify_congestion(score: float) -> str:
        if score < 0.55:
            return "畅通"
        if score < 0.75:
            return "轻度"
        if score < 1.0:
            return "中度"
        return "严重"

    def compute_congestion_scores(
        self,
        prediction: np.ndarray,
        horizon_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.ndim == 3:
            prediction = prediction[:, :, 0]
        if prediction.ndim == 1:
            pred_horizon = prediction
        elif prediction.ndim == 2:
            if prediction.shape[0] != self.num_nodes:
                raise ValueError(f"prediction num_nodes mismatch: expected {self.num_nodes}, got {prediction.shape[0]}")
            horizon_idx = max(0, min(int(horizon_idx), prediction.shape[1] - 1))
            pred_horizon = prediction[:, horizon_idx]
        else:
            raise ValueError("prediction must be [N], [N, H], or [N, H, 1]")

        if pred_horizon.shape[0] != self.num_nodes:
            raise ValueError(f"prediction num_nodes mismatch: expected {self.num_nodes}, got {pred_horizon.shape[0]}")

        pred_horizon = np.maximum(np.nan_to_num(pred_horizon, nan=0.0), 0.0)
        scores = pred_horizon / self.node_baseline
        scores = np.nan_to_num(scores, nan=0.0, posinf=3.0, neginf=0.0)
        return pred_horizon.astype(np.float32), scores.astype(np.float32)

    def top_risk_nodes(
        self,
        prediction: np.ndarray,
        horizon_idx: int = 0,
        topk: int = 10,
    ) -> List[Dict[str, Any]]:
        pred_horizon, scores = self.compute_congestion_scores(prediction, horizon_idx=horizon_idx)
        limit = max(0, min(int(topk), self.num_nodes))
        indices = np.argsort(-scores)[:limit]
        return [
            {
                "rank": rank,
                "node_id": int(node_id),
                "predicted_flow": float(pred_horizon[node_id]),
                "baseline_flow": float(self.node_baseline[node_id]),
                "congestion_score": float(scores[node_id]),
                "risk_level": self.classify_congestion(float(scores[node_id])),
            }
            for rank, node_id in enumerate(indices, start=1)
        ]

    def _edge_weight(
        self,
        source: int,
        target: int,
        distance: float,
        scores: np.ndarray,
        strategy: str,
        alpha: float,
    ) -> float:
        strategy = strategy.strip().lower()
        edge_score = float((scores[source] + scores[target]) / 2.0)
        alpha = min(max(float(alpha), 0.0), 5.0)

        if strategy == self.STRATEGY_DISTANCE:
            return float(distance)
        if strategy == self.STRATEGY_CONGESTION:
            return float(1.0 + alpha * edge_score)
        if strategy == self.STRATEGY_BALANCED:
            return float(distance * (1.0 + alpha * edge_score))
        raise ValueError("strategy must be one of: distance, congestion, balanced")

    def _shortest_path(
        self,
        source: int,
        target: int,
        scores: np.ndarray,
        strategy: str,
        alpha: float,
    ) -> Tuple[List[int], float]:
        queue: List[Tuple[float, int]] = [(0.0, source)]
        distances = {source: 0.0}
        previous: Dict[int, int] = {}
        visited = set()

        while queue:
            current_cost, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            if node == target:
                break

            for next_node, edge_distance in self.graph.get(node, []):
                new_cost = current_cost + self._edge_weight(
                    node,
                    next_node,
                    edge_distance,
                    scores=scores,
                    strategy=strategy,
                    alpha=alpha,
                )
                if new_cost < distances.get(next_node, float("inf")):
                    distances[next_node] = new_cost
                    previous[next_node] = node
                    heapq.heappush(queue, (new_cost, next_node))

        if target not in distances:
            return [], float("inf")

        path = [target]
        while path[-1] != source:
            path.append(previous[path[-1]])
        path.reverse()
        return path, float(distances[target])

    def _candidate_paths(
        self,
        source: int,
        target: int,
        scores: np.ndarray,
        strategy: str,
        alpha: float,
        max_routes: int,
        max_expansions: int = 12000,
    ) -> List[Tuple[List[int], float]]:
        max_routes = max(1, int(max_routes))
        queue: List[Tuple[float, Tuple[int, ...]]] = [(0.0, (source,))]
        candidates: List[Tuple[List[int], float]] = []
        seen_complete = set()
        expansions = 0

        while queue and len(candidates) < max_routes and expansions < max_expansions:
            cost, path_tuple = heapq.heappop(queue)
            expansions += 1
            node = path_tuple[-1]

            if node == target:
                if path_tuple not in seen_complete:
                    seen_complete.add(path_tuple)
                    candidates.append((list(path_tuple), float(cost)))
                continue

            path_set = set(path_tuple)
            neighbors = []
            for next_node, edge_distance in self.graph.get(node, []):
                if next_node in path_set:
                    continue
                edge_weight = self._edge_weight(
                    node,
                    next_node,
                    edge_distance,
                    scores=scores,
                    strategy=strategy,
                    alpha=alpha,
                )
                neighbors.append((edge_weight, next_node))

            for edge_weight, next_node in sorted(neighbors):
                heapq.heappush(queue, (cost + edge_weight, path_tuple + (next_node,)))

        if not candidates:
            shortest_path, shortest_cost = self._shortest_path(
                source,
                target,
                scores=scores,
                strategy=strategy,
                alpha=alpha,
            )
            if shortest_path:
                candidates.append((shortest_path, shortest_cost))

        return candidates

    def _path_distance(self, path: List[int]) -> float:
        total = 0.0
        for source, target in zip(path[:-1], path[1:]):
            match = next((distance for node, distance in self.graph.get(source, []) if node == target), None)
            if match is None:
                return float("inf")
            total += float(match)
        return total

    def query_reachability(self, source: int, target: int) -> Dict[str, Any]:
        source = int(source)
        target = int(target)
        if source < 0 or source >= self.num_nodes:
            raise ValueError(f"source node out of range: {source}")
        if target < 0 or target >= self.num_nodes:
            raise ValueError(f"target node out of range: {target}")

        if source == target:
            return {
                "source": source,
                "target": target,
                "reachable": True,
                "hop_count": 0,
                "path": [source],
                "path_text": str(source),
                "message": "Source and target are the same node.",
            }

        queue: List[int] = [source]
        previous: Dict[int, int] = {}
        visited = {source}
        cursor = 0

        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            if node == target:
                break
            for next_node, _ in self.graph.get(node, []):
                if next_node in visited:
                    continue
                visited.add(next_node)
                previous[next_node] = node
                queue.append(next_node)

        if target not in visited:
            return {
                "source": source,
                "target": target,
                "reachable": False,
                "hop_count": None,
                "path": [],
                "path_text": "",
                "message": "Target is not reachable from source in the current graph.",
            }

        path = [target]
        while path[-1] != source:
            path.append(previous[path[-1]])
        path.reverse()
        return {
            "source": source,
            "target": target,
            "reachable": True,
            "hop_count": len(path) - 1,
            "path": [int(node) for node in path],
            "path_text": " -> ".join(str(node) for node in path),
            "message": "Target is reachable from source.",
        }

    def _build_route_result(
        self,
        route: List[int],
        route_cost: float,
        source: int,
        target: int,
        horizon_idx: int,
        strategy: str,
        alpha: float,
        pred_horizon: np.ndarray,
        scores: np.ndarray,
        shortest_route: List[int],
        shortest_cost: float,
        route_rank: int = 1,
    ) -> Dict[str, Any]:
        route_scores = scores[route]
        route_flows = pred_horizon[route]
        route_distance = self._path_distance(route)
        high_risk_nodes = [node for node in route if scores[node] >= 1.0]

        shortest_distance = self._path_distance(shortest_route) if shortest_route else float("inf")
        shortest_scores = scores[shortest_route] if shortest_route else np.array([], dtype=np.float32)
        shortest_avg_score = float(np.mean(shortest_scores)) if shortest_scores.size else 0.0
        shortest_max_score = float(np.max(shortest_scores)) if shortest_scores.size else 0.0
        shortest_high_risk_count = int(np.count_nonzero(shortest_scores >= 1.0)) if shortest_scores.size else 0
        avg_score = float(np.mean(route_scores)) if len(route_scores) else 0.0

        return {
            "route_rank": int(route_rank),
            "source": source,
            "target": target,
            "horizon_idx": int(horizon_idx),
            "strategy": strategy,
            "alpha": float(alpha),
            "path": [int(node) for node in route],
            "path_text": " -> ".join(str(node) for node in route),
            "reachable": True,
            "cost": float(route_cost),
            "distance": float(route_distance),
            "node_count": len(route),
            "avg_congestion_score": avg_score,
            "max_congestion_score": float(np.max(route_scores)) if len(route_scores) else 0.0,
            "high_risk_node_count": len(high_risk_nodes),
            "high_risk_nodes": [int(node) for node in high_risk_nodes],
            "route_nodes": [
                {
                    "order": idx + 1,
                    "node_id": int(node),
                    "predicted_flow": float(route_flows[idx]),
                    "baseline_flow": float(self.node_baseline[node]),
                    "congestion_score": float(route_scores[idx]),
                    "risk_level": self.classify_congestion(float(route_scores[idx])),
                }
                for idx, node in enumerate(route)
            ],
            "shortest_path": [int(node) for node in shortest_route],
            "shortest_path_text": " -> ".join(str(node) for node in shortest_route),
            "shortest_cost": float(shortest_cost),
            "shortest_distance": float(shortest_distance),
            "shortest_avg_congestion_score": shortest_avg_score,
            "shortest_max_congestion_score": shortest_max_score,
            "shortest_high_risk_node_count": shortest_high_risk_count,
            "shortest_node_count": len(shortest_route),
            "distance_delta": float(route_distance - shortest_distance),
            "congestion_delta": float(avg_score - shortest_avg_score),
        }

    def get_network_preview(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": int(node_id),
                    "x": float(self.node_positions[node_id, 0]),
                    "y": float(self.node_positions[node_id, 1]),
                    "degree": int(len(self.graph.get(node_id, []))),
                }
                for node_id in range(self.num_nodes)
            ],
            "edges": [
                {
                    "from": int(edge["from"]),
                    "to": int(edge["to"]),
                    "distance": float(edge["distance"]),
                }
                for edge in self.display_edges
            ],
        }

    def recommend_routes(
        self,
        prediction: np.ndarray,
        source: int,
        target: int,
        horizon_idx: int = 0,
        strategy: str = STRATEGY_BALANCED,
        alpha: float = 1.0,
        topk: int = 10,
        candidate_count: int = 3,
    ) -> Dict[str, Any]:
        source = int(source)
        target = int(target)
        if source < 0 or source >= self.num_nodes:
            raise ValueError(f"source node out of range: {source}")
        if target < 0 or target >= self.num_nodes:
            raise ValueError(f"target node out of range: {target}")
        if source == target:
            raise ValueError("source and target must be different")

        pred_horizon, scores = self.compute_congestion_scores(prediction, horizon_idx=horizon_idx)
        path_candidates = self._candidate_paths(
            source=source,
            target=target,
            scores=scores,
            strategy=strategy,
            alpha=alpha,
            max_routes=candidate_count,
        )
        shortest_route, shortest_cost = self._shortest_path(
            source,
            target,
            scores=scores,
            strategy=self.STRATEGY_DISTANCE,
            alpha=alpha,
        )
        top_risk_nodes = self.top_risk_nodes(prediction, horizon_idx=horizon_idx, topk=topk)

        if not path_candidates:
            return {
                "source": source,
                "target": target,
                "horizon_idx": int(horizon_idx),
                "strategy": strategy,
                "alpha": float(alpha),
                "path": [],
                "path_text": "",
                "reachable": False,
                "candidates": [],
                "selected_index": -1,
                "message": "No reachable path was found in the current road network.",
                "top_risk_nodes": top_risk_nodes,
            }

        candidates = [
            self._build_route_result(
                route=path,
                route_cost=cost,
                source=source,
                target=target,
                horizon_idx=horizon_idx,
                strategy=strategy,
                alpha=alpha,
                pred_horizon=pred_horizon,
                scores=scores,
                shortest_route=shortest_route,
                shortest_cost=shortest_cost,
                route_rank=rank,
            )
            for rank, (path, cost) in enumerate(path_candidates, start=1)
        ]
        selected = dict(candidates[0])
        selected.update(
            {
                "candidates": candidates,
                "selected_index": 0,
                "top_risk_nodes": top_risk_nodes,
            }
        )
        return selected

    def recommend_route(
        self,
        prediction: np.ndarray,
        source: int,
        target: int,
        horizon_idx: int = 0,
        strategy: str = STRATEGY_BALANCED,
        alpha: float = 1.0,
        topk: int = 10,
    ) -> Dict[str, Any]:
        return self.recommend_routes(
            prediction=prediction,
            source=source,
            target=target,
            horizon_idx=horizon_idx,
            strategy=strategy,
            alpha=alpha,
            topk=topk,
            candidate_count=1,
        )

    @staticmethod
    def explain_route(route_result: Dict[str, Any]) -> List[str]:
        if not route_result.get("reachable", False):
            return [route_result.get("message", "当前路网中未找到可达路径。")]

        explanations: List[str] = []
        distance_delta = float(route_result.get("distance_delta", 0.0))
        congestion_delta = float(route_result.get("congestion_delta", 0.0))
        high_risk_count = int(route_result.get("high_risk_node_count", 0))
        shortest_high_risk_count = int(route_result.get("shortest_high_risk_node_count", 0))
        avg_score = float(route_result.get("avg_congestion_score", 0.0))
        max_score = float(route_result.get("max_congestion_score", 0.0))
        high_risk_nodes = route_result.get("high_risk_nodes", [])

        if congestion_delta < -0.02:
            explanations.append(
                f"相比最短路线，平均拥堵指数降低 {abs(congestion_delta):.3f}。"
            )
        elif congestion_delta > 0.02:
            explanations.append(
                f"相比最短路线，平均拥堵指数增加 {congestion_delta:.3f}。"
            )
        else:
            explanations.append("平均拥堵指数与最短路线基本接近。")

        if distance_delta > 0.01:
            explanations.append(
                f"相对最短路线，距离增加 {distance_delta:.2f} 个距离单位。"
            )
        elif distance_delta < -0.01:
            explanations.append(
                f"相对最短路线，距离减少 {abs(distance_delta):.2f} 个距离单位。"
            )
        else:
            explanations.append("路线距离与最短路线基本一致。")

        if high_risk_count < shortest_high_risk_count:
            explanations.append(
                f"相比最短路线，避开了 {shortest_high_risk_count - high_risk_count} 个严重拥堵风险节点。"
            )
        elif high_risk_count > 0:
            sample_nodes = ", ".join(str(node) for node in high_risk_nodes[:5])
            explanations.append(f"当前路线仍经过 {high_risk_count} 个严重拥堵风险节点：{sample_nodes}。")
        else:
            explanations.append("当前预测步长下，路线中没有严重拥堵风险节点。")

        if max_score >= 1.0:
            explanations.append(f"路线最高拥堵指数为 {max_score:.3f}，建议人工复核或考虑其他候选路线。")
        elif avg_score >= 0.75:
            explanations.append("路线整体存在中等拥堵压力，更适合作为备选方案。")
        else:
            explanations.append("当前预测下路线整体拥堵压力较低，适合作为推荐方案。")

        return explanations
