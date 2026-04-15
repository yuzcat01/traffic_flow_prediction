import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import yaml

from datasets.traffic_dataset import build_adjacency_matrix, get_flow_data, resolve_graph_config
from utils.config import load_yaml


class DataService:
    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root).resolve()

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _to_project_relative_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path.resolve())

    def import_data_file(self, src_path: str, data_kind: str = "flow") -> Dict[str, str]:
        src = Path(src_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"file not found: {src}")

        data_kind = data_kind.strip().lower()
        if data_kind not in {"graph", "flow"}:
            raise ValueError("data_kind must be 'graph' or 'flow'")

        if data_kind == "graph" and src.suffix.lower() not in {".csv", ".txt"}:
            raise ValueError("graph file should be .csv or .txt")
        if data_kind == "flow" and src.suffix.lower() not in {".npz"}:
            raise ValueError("flow file should be .npz")

        target_dir = self.project_root / "data" / "raw" / "imported"
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{data_kind}_{timestamp}_{src.name}"
        dst = (target_dir / safe_name).resolve()
        shutil.copy2(src, dst)

        return {
            "source_path": str(src),
            "target_path": str(dst),
            "target_relative_path": self._to_project_relative_path(dst),
        }

    def create_data_config(
        self,
        config_name: str,
        dataset_name: str,
        graph_path: str,
        flow_path: str,
        num_nodes: int,
        train_days: int,
        test_days: int,
        time_interval: int,
    ) -> Dict[str, str]:
        config_name = config_name.strip()
        dataset_name = dataset_name.strip()

        if not config_name:
            raise ValueError("config_name can not be empty")
        if not dataset_name:
            raise ValueError("dataset_name can not be empty")
        if num_nodes <= 0:
            raise ValueError("num_nodes must be > 0")
        if train_days <= 0 or test_days <= 0:
            raise ValueError("train_days and test_days must be > 0")
        if time_interval <= 0:
            raise ValueError("time_interval must be > 0")

        graph_path_resolved = self._resolve_path(graph_path)
        flow_path_resolved = self._resolve_path(flow_path)

        if not graph_path_resolved.exists():
            raise FileNotFoundError(f"graph file not found: {graph_path_resolved}")
        if not flow_path_resolved.exists():
            raise FileNotFoundError(f"flow file not found: {flow_path_resolved}")

        cfg = {
            "dataset": {
                "name": dataset_name,
                "graph_path": self._to_project_relative_path(graph_path_resolved),
                "flow_path": self._to_project_relative_path(flow_path_resolved),
                "num_nodes": int(num_nodes),
                "divide_days": [int(train_days), int(test_days)],
                "time_interval": int(time_interval),
            }
        }

        cfg_dir = self.project_root / "configs" / "data"
        cfg_dir.mkdir(parents=True, exist_ok=True)

        file_name = config_name if config_name.endswith(".yaml") else f"{config_name}.yaml"
        cfg_path = (cfg_dir / file_name).resolve()

        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        return {
            "config_path": str(cfg_path),
            "config_relative_path": self._to_project_relative_path(cfg_path),
        }

    def export_preview_summary(self, preview: Dict[str, Any], save_path: str):
        out_path = Path(save_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {}
        for key, value in preview.items():
            if key in {"flow_data", "adjacency"}:
                continue
            if isinstance(value, np.ndarray):
                export_data[key] = value.tolist()
            else:
                export_data[key] = value

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    def export_node_series(
        self,
        preview: Dict[str, Any],
        node_id: int,
        max_points: Optional[int],
        save_path: str,
    ):
        series = self.get_node_series(preview=preview, node_id=node_id, max_points=max_points)
        out_path = Path(save_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = np.column_stack([np.arange(len(series), dtype=np.int64), series.astype(np.float64)])
        header = "time_step,traffic_flow"
        np.savetxt(out_path, rows, fmt=["%d", "%.6f"], delimiter=",", header=header, comments="")

    def load_preview(self, data_cfg_path: str, model_cfg_path: Optional[str] = None) -> Dict[str, Any]:
        data_cfg_all = load_yaml(data_cfg_path)
        data_cfg = data_cfg_all.get("dataset", {})

        model_cfg = {}
        if model_cfg_path:
            model_cfg_all = load_yaml(model_cfg_path)
            model_cfg = model_cfg_all.get("model", {})

        dataset_name = data_cfg.get("name", "UnknownDataset")
        graph_path = self._resolve_path(data_cfg.get("graph_path", ""))
        flow_path = self._resolve_path(data_cfg.get("flow_path", ""))
        num_nodes = int(data_cfg.get("num_nodes", 0))
        divide_days = data_cfg.get("divide_days", [0, 0])
        time_interval = int(data_cfg.get("time_interval", 5))

        graph_cfg = resolve_graph_config(model_cfg.get("graph", {}))
        history_length = int(model_cfg.get("input", {}).get("history_length", 12))
        predict_steps = int(model_cfg.get("output", {}).get("predict_steps", 1))

        flow_data = get_flow_data(str(flow_path))           # [N, T, 1]
        train_days = int(divide_days[0]) if len(divide_days) > 0 else 0
        test_days = int(divide_days[1]) if len(divide_days) > 1 else 0
        one_day_length = int(24 * 60 / time_interval) if time_interval > 0 else 0

        adjacency = build_adjacency_matrix(
            distance_file=str(graph_path),
            num_nodes=num_nodes,
            graph_cfg=graph_cfg,
            flow_data=flow_data,
            flow_slice=(0, train_days * one_day_length),
        )

        n_nodes, total_steps, input_dim = flow_data.shape

        train_steps = train_days * one_day_length
        test_steps_cfg = test_days * one_day_length

        train_samples = max(train_steps - history_length - predict_steps + 1, 0)
        available_test_steps = max(total_steps - train_steps, 0)
        test_steps_actual = min(test_steps_cfg, available_test_steps)
        test_samples = max(test_steps_actual - predict_steps + 1, 0)

        nonzero_edges = int(np.count_nonzero(adjacency))
        density = float(nonzero_edges / (n_nodes * n_nodes)) if n_nodes > 0 else 0.0

        return {
            "dataset_name": dataset_name,
            "graph_path": str(graph_path),
            "flow_path": str(flow_path),
            "num_nodes_cfg": num_nodes,
            "num_nodes_actual": n_nodes,
            "total_steps": total_steps,
            "input_dim": input_dim,
            "divide_days": divide_days,
            "time_interval": time_interval,
            "one_day_length": one_day_length,
            "train_days": train_days,
            "test_days": test_days,
            "train_steps": train_steps,
            "test_steps_cfg": test_steps_cfg,
            "test_steps_actual": test_steps_actual,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "graph_type": graph_cfg["type"],
            "graph_cfg": graph_cfg,
            "history_length": history_length,
            "predict_steps": predict_steps,
            "adjacency_shape": adjacency.shape,
            "nonzero_edges": nonzero_edges,
            "density": density,
            "flow_data": flow_data,
            "adjacency": adjacency,
        }

    def get_node_series(self, preview: Dict[str, Any], node_id: int, max_points: Optional[int] = None):
        flow_data = preview["flow_data"]
        node_id = max(0, min(node_id, flow_data.shape[0] - 1))

        series = flow_data[node_id, :, 0]
        if max_points is not None and max_points > 0 and len(series) > max_points:
            series = series[:max_points]

        return series.astype(float)
