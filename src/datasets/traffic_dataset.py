import csv
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_IMPORT_ERROR = None
except Exception as _e:
    torch = None
    _TORCH_IMPORT_ERROR = _e

    class Dataset:  # type: ignore[misc]
        pass


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError(
            "No module named 'torch'. Please install PyTorch or build executable with torch included."
        ) from _TORCH_IMPORT_ERROR


def _load_distance_adjacency(
    distance_file: str,
    num_nodes: int,
    id_file: Optional[str] = None,
    graph_type: str = "connect",
) -> np.ndarray:
    if not os.path.exists(distance_file):
        raise FileNotFoundError(f"distance_file not found: {distance_file}")

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    if id_file is not None:
        if not os.path.exists(id_file):
            raise FileNotFoundError(f"id_file not found: {id_file}")

        with open(id_file, "r", encoding="utf-8") as f:
            node_ids = f.read().strip().split("\n")
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(node_ids)}

        with open(distance_file, "r", encoding="utf-8") as f:
            f.readline()
            reader = csv.reader(f)
            for item in reader:
                if len(item) != 3:
                    continue

                i, j, distance = int(item[0]), int(item[1]), float(item[2])

                if i not in node_id_dict or j not in node_id_dict:
                    continue

                ii, jj = node_id_dict[i], node_id_dict[j]

                if graph_type == "connect":
                    A[ii, jj] = 1.0
                    A[jj, ii] = 1.0
                elif graph_type == "distance":
                    if distance > 0:
                        A[ii, jj] = 1.0 / distance
                        A[jj, ii] = 1.0 / distance
                else:
                    raise ValueError("graph_type must be 'connect' or 'distance'")
    else:
        with open(distance_file, "r", encoding="utf-8") as f:
            f.readline()
            reader = csv.reader(f)
            for item in reader:
                if len(item) != 3:
                    continue

                i, j, distance = int(item[0]), int(item[1]), float(item[2])

                if i >= num_nodes or j >= num_nodes:
                    continue

                if graph_type == "connect":
                    A[i, j] = 1.0
                    A[j, i] = 1.0
                elif graph_type == "distance":
                    if distance > 0:
                        A[i, j] = 1.0 / distance
                        A[j, i] = 1.0 / distance
                else:
                    raise ValueError("graph_type must be 'connect' or 'distance'")

    return A


def _normalize_weight_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=True)
    np.fill_diagonal(matrix, 0.0)
    max_value = float(np.max(matrix)) if matrix.size > 0 else 0.0
    if max_value > 0:
        matrix /= max_value
    return matrix


def _symmetrize_by_max(matrix: np.ndarray) -> np.ndarray:
    return np.maximum(matrix, matrix.T).astype(np.float32)


def _keep_topk_per_row(matrix: np.ndarray, topk: int) -> np.ndarray:
    if topk <= 0 or topk >= matrix.shape[1]:
        return matrix

    trimmed = np.zeros_like(matrix, dtype=np.float32)
    for row_idx in range(matrix.shape[0]):
        row = matrix[row_idx]
        positive_indices = np.flatnonzero(row > 0)
        if len(positive_indices) <= topk:
            trimmed[row_idx, positive_indices] = row[positive_indices]
            continue

        top_indices = positive_indices[np.argsort(row[positive_indices])[-topk:]]
        trimmed[row_idx, top_indices] = row[top_indices]

    return trimmed


def _build_correlation_adjacency(
    flow_data: np.ndarray,
    num_nodes: int,
    topk: int,
    threshold: float,
    use_abs_corr: bool,
) -> np.ndarray:
    series = flow_data[:num_nodes, :, 0].astype(np.float32)
    corr = np.corrcoef(series)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if use_abs_corr:
        corr = np.abs(corr)
    else:
        corr = np.maximum(corr, 0.0)

    np.fill_diagonal(corr, 0.0)

    if threshold > 0:
        corr[corr < threshold] = 0.0

    corr = _keep_topk_per_row(corr, topk=topk)
    corr = _symmetrize_by_max(corr)
    return _normalize_weight_matrix(corr)


def resolve_graph_config(graph_cfg: Optional[Any] = None, graph_type: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(graph_cfg, dict):
        resolved = dict(graph_cfg)
    else:
        resolved = {}

    resolved_type = str(resolved.get("type", graph_type or "connect")).strip().lower()
    resolved["type"] = resolved_type
    resolved.setdefault("correlation_topk", 8)
    resolved.setdefault("correlation_threshold", 0.3)
    resolved.setdefault("use_abs_corr", False)
    resolved.setdefault("fusion_alpha", 0.5)
    return resolved


def resolve_preprocess_config(preprocess_cfg: Optional[Any] = None) -> Dict[str, Any]:
    if isinstance(preprocess_cfg, dict):
        resolved = dict(preprocess_cfg)
    else:
        resolved = {}

    missing_strategy = str(resolved.get("missing_strategy", "none")).strip().lower()
    alias_map = {
        "linear": "linear_interpolate",
        "linear_interp": "linear_interpolate",
        "ffill": "forward_fill",
        "mean": "mean_fill",
    }
    missing_strategy = alias_map.get(missing_strategy, missing_strategy)
    if missing_strategy not in {"none", "linear_interpolate", "forward_fill", "mean_fill"}:
        raise ValueError(
            "dataset.preprocess.missing_strategy must be one of: "
            "none, linear_interpolate, forward_fill, mean_fill"
        )

    clip_min = resolved.get("clip_min", 0.0)
    clip_min = None if clip_min is None else float(clip_min)

    clip_max_quantile = resolved.get("clip_max_quantile")
    if clip_max_quantile in {"", None}:
        clip_max_quantile = None
    else:
        clip_max_quantile = float(clip_max_quantile)
        if not 0.0 < clip_max_quantile < 1.0:
            raise ValueError("dataset.preprocess.clip_max_quantile must be in (0, 1)")

    return {
        "missing_strategy": missing_strategy,
        "clip_min": clip_min,
        "clip_max_quantile": clip_max_quantile,
    }


def _fill_missing_series(values: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "none":
        return np.nan_to_num(values, nan=0.0).astype(np.float32, copy=False)

    mask = np.isnan(values)
    if not np.any(mask):
        return values.astype(np.float32, copy=False)

    valid_idx = np.flatnonzero(~mask)
    if valid_idx.size == 0:
        return np.zeros_like(values, dtype=np.float32)

    filled = values.astype(np.float32, copy=True)
    time_idx = np.arange(values.shape[0], dtype=np.float32)

    if strategy == "linear_interpolate":
        filled[mask] = np.interp(time_idx[mask], time_idx[valid_idx], filled[valid_idx]).astype(np.float32)
        return filled

    if strategy == "forward_fill":
        last_value = float(filled[valid_idx[0]])
        for idx in range(filled.shape[0]):
            if np.isnan(filled[idx]):
                filled[idx] = last_value
            else:
                last_value = float(filled[idx])
        return filled

    if strategy == "mean_fill":
        mean_value = float(np.mean(filled[valid_idx]))
        filled[mask] = mean_value
        return filled

    raise ValueError(f"Unsupported missing strategy: {strategy}")


def _fill_missing_values(flow_data: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "none" and not np.isnan(flow_data).any():
        return flow_data.astype(np.float32, copy=False)

    cleaned = np.empty_like(flow_data, dtype=np.float32)
    num_nodes, _, num_features = flow_data.shape
    for node_idx in range(num_nodes):
        for feat_idx in range(num_features):
            cleaned[node_idx, :, feat_idx] = _fill_missing_series(
                flow_data[node_idx, :, feat_idx],
                strategy=strategy,
            )
    return cleaned


def _clip_flow_values(
    flow_data: np.ndarray,
    clip_min: Optional[float],
    clip_max_quantile: Optional[float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    clipped = flow_data.astype(np.float32, copy=True)
    stats: Dict[str, Any] = {
        "clip_min": clip_min,
        "clip_max_quantile": clip_max_quantile,
        "clip_max_value": None,
        "clipped_value_count": 0,
    }

    if clip_min is not None:
        below_mask = clipped < clip_min
        stats["clipped_value_count"] += int(np.count_nonzero(below_mask))
        clipped[below_mask] = clip_min

    if clip_max_quantile is not None and clipped.size > 0:
        max_value = float(np.quantile(clipped, clip_max_quantile))
        above_mask = clipped > max_value
        stats["clip_max_value"] = max_value
        stats["clipped_value_count"] += int(np.count_nonzero(above_mask))
        clipped[above_mask] = max_value

    return clipped, stats


def build_adjacency_matrix(
    distance_file: str,
    num_nodes: int,
    id_file: Optional[str] = None,
    graph_cfg: Optional[Dict[str, Any]] = None,
    flow_file: Optional[str] = None,
    flow_data: Optional[np.ndarray] = None,
    flow_slice: Optional[Tuple[int, int]] = None,
    preprocess_cfg: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    graph_cfg = resolve_graph_config(graph_cfg=graph_cfg)
    graph_type = graph_cfg["type"]

    if graph_type in {"connect", "distance"}:
        return _load_distance_adjacency(
            distance_file=distance_file,
            num_nodes=num_nodes,
            id_file=id_file,
            graph_type=graph_type,
        )

    if graph_type not in {"correlation", "distance_correlation"}:
        raise ValueError(
            "graph.type must be one of: connect, distance, correlation, distance_correlation"
        )

    if flow_data is None:
        if flow_file is None:
            raise ValueError(f"graph.type={graph_type} requires flow_file or flow_data")
        flow_data = get_flow_data(flow_file, preprocess_cfg=preprocess_cfg)

    if flow_slice is not None:
        start_t, end_t = flow_slice
        sliced_flow = flow_data[:, start_t:end_t, :]
        if sliced_flow.shape[1] > 1:
            flow_data = sliced_flow

    corr_adj = _build_correlation_adjacency(
        flow_data=flow_data,
        num_nodes=num_nodes,
        topk=int(graph_cfg.get("correlation_topk", 8)),
        threshold=float(graph_cfg.get("correlation_threshold", 0.3)),
        use_abs_corr=bool(graph_cfg.get("use_abs_corr", False)),
    )

    if graph_type == "correlation":
        return corr_adj

    distance_adj = _load_distance_adjacency(
        distance_file=distance_file,
        num_nodes=num_nodes,
        id_file=id_file,
        graph_type="distance",
    )
    distance_adj = _normalize_weight_matrix(distance_adj)

    fusion_alpha = float(graph_cfg.get("fusion_alpha", 0.5))
    fusion_alpha = min(max(fusion_alpha, 0.0), 1.0)

    mixed_adj = fusion_alpha * distance_adj + (1.0 - fusion_alpha) * corr_adj
    mixed_adj = _keep_topk_per_row(mixed_adj, topk=int(graph_cfg.get("correlation_topk", 8)))
    mixed_adj = _symmetrize_by_max(mixed_adj)
    return _normalize_weight_matrix(mixed_adj)


def get_adjacent_matrix(
    distance_file: str,
    num_nodes: int,
    id_file: Optional[str] = None,
    graph_type: str = "connect",
) -> np.ndarray:
    return build_adjacency_matrix(
        distance_file=distance_file,
        num_nodes=num_nodes,
        id_file=id_file,
        graph_cfg={"type": graph_type},
    )


def get_flow_data(
    flow_file: str,
    preprocess_cfg: Optional[Dict[str, Any]] = None,
    return_stats: bool = False,
):
    if not os.path.exists(flow_file):
        raise FileNotFoundError(f"flow_file not found: {flow_file}")

    with np.load(flow_file) as data:
        if "data" not in data:
            raise KeyError(f"'data' key not found in {flow_file}")
        flow_data = data["data"]  # [T, N, D]
    flow_data = flow_data.transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]  # [N, T, 1]
    flow_data = flow_data.astype(np.float32)

    preprocess_cfg = resolve_preprocess_config(preprocess_cfg=preprocess_cfg)
    missing_before = int(np.isnan(flow_data).sum())

    flow_data = _fill_missing_values(flow_data, strategy=preprocess_cfg["missing_strategy"])
    flow_data, clip_stats = _clip_flow_values(
        flow_data=flow_data,
        clip_min=preprocess_cfg["clip_min"],
        clip_max_quantile=preprocess_cfg["clip_max_quantile"],
    )

    stats = {
        "missing_strategy": preprocess_cfg["missing_strategy"],
        "missing_value_count_before": missing_before,
        "missing_value_count_after": int(np.isnan(flow_data).sum()),
        "clip_min": clip_stats["clip_min"],
        "clip_max_quantile": clip_stats["clip_max_quantile"],
        "clip_max_value": clip_stats["clip_max_value"],
        "clipped_value_count": clip_stats["clipped_value_count"],
    }

    if return_stats:
        return flow_data, stats
    return flow_data


def traffic_batch_collate(batch):
    _require_torch()
    if not batch:
        raise ValueError("traffic_batch_collate received an empty batch")

    return {
        "graph": batch[0]["graph"],
        "flow_x": torch.stack([item["flow_x"] for item in batch], dim=0),
        "flow_y": torch.stack([item["flow_y"] for item in batch], dim=0),
    }


class LoadData(Dataset):
    def __init__(
        self,
        data_path,
        num_nodes: int,
        divide_days,
        time_interval: int,
        history_length: int,
        train_mode: str,
        predict_steps: int = 1,
        graph_type: str = "connect",
        graph_cfg: Optional[Dict[str, Any]] = None,
        preprocess_cfg: Optional[Dict[str, Any]] = None,
        id_file: Optional[str] = None,
        norm_base: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        norm_source_range: Optional[Tuple[int, int]] = None,
    ):
        _require_torch()
        if train_mode not in ["train", "test"]:
            raise ValueError("train_mode must be 'train' or 'test'")

        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]
        self.test_days = divide_days[1]
        self.history_length = history_length
        self.predict_steps = int(predict_steps)
        if self.predict_steps <= 0:
            raise ValueError("predict_steps must be > 0")
        self.time_interval = time_interval
        self.one_day_length = int(24 * 60 / self.time_interval)
        self.graph_cfg = resolve_graph_config(graph_cfg=graph_cfg, graph_type=graph_type)
        self.preprocess_cfg = resolve_preprocess_config(preprocess_cfg=preprocess_cfg)

        self.requested_train_steps = int(self.train_days * self.one_day_length)
        self.requested_test_steps = int(self.test_days * self.one_day_length)

        raw_flow_data, self.preprocess_stats = get_flow_data(
            data_path[1],
            preprocess_cfg=self.preprocess_cfg,
            return_stats=True,
        )
        self.total_steps = int(raw_flow_data.shape[1])

        self.train_total_steps = min(self.requested_train_steps, self.total_steps)
        remain_steps = max(self.total_steps - self.train_total_steps, 0)
        self.test_total_steps = min(self.requested_test_steps, remain_steps)

        graph_flow_end = self.train_total_steps

        self.graph = build_adjacency_matrix(
            distance_file=data_path[0],
            num_nodes=num_nodes,
            id_file=id_file,
            graph_cfg=self.graph_cfg,
            flow_data=raw_flow_data,
            flow_slice=(0, graph_flow_end),
            preprocess_cfg=self.preprocess_cfg,
        )

        if norm_base is None:
            if norm_source_range is None:
                start_t = 0
                end_t = self.train_total_steps
            else:
                start_t, end_t = norm_source_range

            start_t = int(max(start_t, 0))
            end_t = int(min(end_t, self.total_steps))
            if end_t <= start_t:
                start_t = 0
                end_t = max(1, self.train_total_steps)

            norm_source = raw_flow_data[:, start_t:end_t, :]
            if norm_source.shape[1] <= 0:
                raise ValueError(
                    "No valid normalization range found. "
                    f"total_steps={self.total_steps}, start_t={start_t}, end_t={end_t}"
                )
            self.flow_norm = self.normalize_base(norm_source, norm_dim=1)
        else:
            self.flow_norm = norm_base

        self.flow_data = self.normalize_data(
            max_data=self.flow_norm[0],
            min_data=self.flow_norm[1],
            data=raw_flow_data,
        )
        self.graph_tensor = torch.from_numpy(self.graph.astype(np.float32, copy=False))
        self.flow_tensor = torch.from_numpy(self.flow_data)

    def __len__(self):
        if self.train_mode == "train":
            return max(self.train_total_steps - self.history_length - self.predict_steps + 1, 0)
        elif self.train_mode == "test":
            if self.train_total_steps < self.history_length:
                return 0
            return max(self.test_total_steps - self.predict_steps + 1, 0)
        else:
            raise ValueError(f"train_mode [{self.train_mode}] is not defined")

    def __getitem__(self, index: int):
        if self.train_mode == "train":
            data_index = index
        elif self.train_mode == "test":
            data_index = index + self.train_total_steps
        else:
            raise ValueError(f"train_mode [{self.train_mode}] is not defined")

        data_x, data_y = LoadData.slice_data(
            data=self.flow_tensor,
            history_length=self.history_length,
            predict_steps=self.predict_steps,
            index=data_index,
            train_mode=self.train_mode,
        )

        if data_x.shape[1] != self.history_length:
            raise IndexError(
                f"Invalid flow_x length at index={index}: expected {self.history_length}, got {data_x.shape[1]}"
            )
        if data_y.shape[1] != self.predict_steps:
            raise IndexError(
                f"Invalid flow_y length at index={index}: expected {self.predict_steps}, got {data_y.shape[1]}"
            )

        return {
            "graph": self.graph_tensor,
            "flow_x": data_x,
            "flow_y": data_y,
        }

    @staticmethod
    def slice_data(
        data: np.ndarray,
        history_length: int,
        predict_steps: int,
        index: int,
        train_mode: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if train_mode == "train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError(f"train_mode [{train_mode}] is not defined")

        data_x = data[:, start_index:end_index]
        data_y = data[:, end_index:end_index + predict_steps]

        return data_x, data_y

    @staticmethod
    def normalize_base(data: np.ndarray, norm_dim: int):
        max_data = np.max(data, axis=norm_dim, keepdims=True)
        min_data = np.min(data, axis=norm_dim, keepdims=True)
        return max_data.astype(np.float32), min_data.astype(np.float32)

    @staticmethod
    def normalize_data(max_data: np.ndarray, min_data: np.ndarray, data: np.ndarray):
        base = max_data - min_data
        base = np.where(base == 0, 1.0, base)
        normalized_data = (data - min_data) / base
        return normalized_data.astype(np.float32)

    @staticmethod
    def recover_data(max_data: np.ndarray, min_data: np.ndarray, data: np.ndarray):
        base = max_data - min_data
        base = np.where(base == 0, 1.0, base)
        recovered_data = data * base + min_data
        return recovered_data

    @staticmethod
    def to_tensor(data):
        _require_torch()
        if isinstance(data, torch.Tensor):
            return data.to(dtype=torch.float32)
        return torch.from_numpy(np.asarray(data, dtype=np.float32))
