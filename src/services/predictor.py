import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.traffic_dataset import LoadData, traffic_batch_collate
from src.models.builder import build_model
from src.project_paths import get_project_root, resolve_project_path
from src.utils.metrics import Evaluation


class TrafficPredictor:
    def __init__(
        self,
        run_config_path: str,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.project_root = get_project_root()
        run_config_path = str(resolve_project_path(run_config_path, self.project_root))
        if not os.path.exists(run_config_path):
            raise FileNotFoundError(f"run_config not found: {run_config_path}")

        with open(run_config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        if device == "auto":
            train_device = str(self.cfg["train"].get("device", "auto")).strip().lower()
            use_cuda = torch.cuda.is_available() and train_device in {"auto", "cuda"}
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device(device)

        self.dataset_cfg = self.cfg["dataset"]
        self.model_cfg = self.cfg["model"]
        self.train_cfg = self.cfg["train"]
        self.graph_cfg = self.model_cfg.get("graph", {"type": "connect"})
        self.preprocess_cfg = self.dataset_cfg.get("preprocess", {})
        self.graph_path = str(resolve_project_path(self.dataset_cfg["graph_path"], self.project_root))
        self.flow_path = str(resolve_project_path(self.dataset_cfg["flow_path"], self.project_root))
        self.save_dir = str(resolve_project_path(self.train_cfg["save_dir"], self.project_root))

        self.history_length = int(self.model_cfg["input"]["history_length"])
        self.predict_steps = int(self.model_cfg.get("output", {}).get("predict_steps", 1))
        if self.predict_steps <= 0:
            raise ValueError("model.output.predict_steps must be > 0")

        self.num_nodes = int(self.dataset_cfg["num_nodes"])
        self.graph_type = self.graph_cfg["type"]
        self.model_name = self.model_cfg["name"]

        self.model = build_model(self.cfg).to(self.device)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.save_dir,
                "checkpoints",
                f"{self.model_name}_best.pth",
            )
        else:
            checkpoint_path = str(resolve_project_path(checkpoint_path, self.project_root))

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(
                checkpoint_path,
                map_location=self.device,
            )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.norm_base = self._build_norm_base()
        self.graph = self._build_graph_tensor()
        self.test_data = self._build_test_dataset()

    def _build_norm_base(self) -> Tuple[np.ndarray, np.ndarray]:
        one_day_length = int(24 * 60 / self.dataset_cfg["time_interval"])
        total_train_steps = self.dataset_cfg["divide_days"][0] * one_day_length
        full_train_sample_num = total_train_steps - self.history_length - self.predict_steps + 1

        val_ratio = float(self.train_cfg.get("val_ratio", 0.1))
        if val_ratio > 0:
            val_sample_num = max(1, int(full_train_sample_num * val_ratio))
            train_sample_num = full_train_sample_num - val_sample_num
        else:
            train_sample_num = full_train_sample_num

        norm_end_t = train_sample_num + self.history_length + self.predict_steps - 1

        ref_train_data = LoadData(
            data_path=[self.graph_path, self.flow_path],
            num_nodes=self.dataset_cfg["num_nodes"],
            divide_days=self.dataset_cfg["divide_days"],
            time_interval=self.dataset_cfg["time_interval"],
            history_length=self.history_length,
            predict_steps=self.predict_steps,
            train_mode="train",
            graph_cfg=self.graph_cfg,
            preprocess_cfg=self.preprocess_cfg,
            norm_source_range=(0, norm_end_t),
        )
        return ref_train_data.flow_norm

    def _build_graph_tensor(self) -> torch.Tensor:
        ref_train_data = LoadData(
            data_path=[self.graph_path, self.flow_path],
            num_nodes=self.dataset_cfg["num_nodes"],
            divide_days=self.dataset_cfg["divide_days"],
            time_interval=self.dataset_cfg["time_interval"],
            history_length=self.history_length,
            predict_steps=self.predict_steps,
            train_mode="train",
            graph_cfg=self.graph_cfg,
            preprocess_cfg=self.preprocess_cfg,
            norm_base=self.norm_base,
        )
        return torch.tensor(ref_train_data.graph, dtype=torch.float32, device=self.device)

    def _build_test_dataset(self) -> LoadData:
        return LoadData(
            data_path=[self.graph_path, self.flow_path],
            num_nodes=self.dataset_cfg["num_nodes"],
            divide_days=self.dataset_cfg["divide_days"],
            time_interval=self.dataset_cfg["time_interval"],
            history_length=self.history_length,
            predict_steps=self.predict_steps,
            train_mode="test",
            graph_cfg=self.graph_cfg,
            preprocess_cfg=self.preprocess_cfg,
            norm_base=self.norm_base,
        )

    def _check_and_format_window(self, window: np.ndarray) -> np.ndarray:
        window = np.asarray(window, dtype=np.float32)

        if window.ndim == 2:
            window = window[:, :, np.newaxis]

        if window.ndim != 3:
            raise ValueError("window must be [N, H] or [N, H, 1]")

        if window.shape[0] != self.num_nodes:
            raise ValueError(f"window num_nodes mismatch: expected {self.num_nodes}, got {window.shape[0]}")

        if window.shape[1] != self.history_length:
            raise ValueError(
                f"window history_length mismatch: expected {self.history_length}, got {window.shape[1]}"
            )

        if window.shape[2] != 1:
            raise ValueError("input_dim currently must be 1")

        return window

    def get_test_size(self) -> int:
        return len(self.test_data)

    def get_node_count(self) -> int:
        return self.num_nodes

    def get_predict_steps(self) -> int:
        return self.predict_steps

    def predict_window(self, window: np.ndarray, input_normalized: bool = False) -> np.ndarray:
        """
        Input:
            window: [N, H] or [N, H, 1]
            input_normalized:
                False -> input raw flow values, normalize inside this function
                True  -> input already-normalized window
        Output:
            pred: [N, H_out, 1]
        """
        window = self._check_and_format_window(window)

        if input_normalized:
            norm_window = window.astype(np.float32)
        else:
            norm_window = LoadData.normalize_data(
                max_data=self.norm_base[0],
                min_data=self.norm_base[1],
                data=window,
            )

        flow_x = torch.tensor(norm_window, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, N, H, 1]

        data = {
            "graph": self.graph,
            "flow_x": flow_x,
        }

        with torch.no_grad():
            pred = self.model(data).detach().cpu().numpy()[0]  # [N, H_out, 1]

        pred = LoadData.recover_data(self.norm_base[0], self.norm_base[1], pred)
        return pred

    def predict_test_sample(self, index: int) -> Dict[str, np.ndarray]:
        sample = self.test_data[index]

        flow_x = sample["flow_x"].numpy()
        target = sample["flow_y"].numpy()

        pred = self.predict_window(flow_x, input_normalized=True)   # [N, H_out, 1]
        target = LoadData.recover_data(self.norm_base[0], self.norm_base[1], target)

        return {
            "prediction": pred[:, 0, 0],
            "target": target[:, 0, 0],
            "prediction_full": pred[:, :, 0],
            "target_full": target[:, :, 0],
        }

    def get_test_sample_detail(self, index: int, node_id: int, horizon_idx: int = 0) -> Dict[str, np.ndarray]:
        if index < 0 or index >= len(self.test_data):
            raise IndexError(f"sample index out of range: {index}")

        if node_id < 0 or node_id >= self.num_nodes:
            raise IndexError(f"node_id out of range: {node_id}")

        if horizon_idx < 0 or horizon_idx >= self.predict_steps:
            raise IndexError(f"horizon_idx out of range: {horizon_idx}")

        sample = self.test_data[index]

        flow_x = sample["flow_x"].numpy()   # [N, H, 1]
        target = sample["flow_y"].numpy()   # [N, H_out, 1]

        pred_all = self.predict_window(flow_x, input_normalized=True)  # [N, H_out, 1]

        history = LoadData.recover_data(self.norm_base[0], self.norm_base[1], flow_x)   # [N, H, 1]
        target = LoadData.recover_data(self.norm_base[0], self.norm_base[1], target)    # [N, H_out, 1]

        history_node = history[node_id, :, 0]
        target_node = float(target[node_id, horizon_idx, 0])
        pred_node = float(pred_all[node_id, horizon_idx, 0])

        pred_selected_h = pred_all[:, horizon_idx, 0]
        target_selected_h = target[:, horizon_idx, 0]

        return {
            "history": history_node,
            "prediction": pred_node,
            "target": target_node,
            "abs_error": abs(pred_node - target_node),
            "predict_steps": self.predict_steps,
            "horizon_idx": horizon_idx,
            "prediction_all_nodes": pred_selected_h,
            "target_all_nodes": target_selected_h,
            "prediction_all_horizons": pred_all[:, :, 0],
            "target_all_horizons": target[:, :, 0],
            "prediction_node_all_horizons": pred_all[node_id, :, 0],
            "target_node_all_horizons": target[node_id, :, 0],
        }

    def evaluate_test_set(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        if len(self.test_data) <= 0:
            raise ValueError("No valid test samples in current run config.")

        loader = DataLoader(
            self.test_data,
            batch_size=batch_size or self.train_cfg["batch_size"],
            shuffle=False,
            num_workers=self.train_cfg["num_workers"],
            pin_memory=self.device.type == "cuda",
            persistent_workers=int(self.train_cfg["num_workers"]) > 0,
            collate_fn=traffic_batch_collate,
        )

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data in loader:
                batch = {
                    "graph": data["graph"].to(self.device, non_blocking=self.device.type == "cuda"),
                    "flow_x": data["flow_x"].to(self.device, non_blocking=self.device.type == "cuda"),
                    "flow_y": data["flow_y"].to(self.device, non_blocking=self.device.type == "cuda"),
                }
                pred = self.model(batch).detach().cpu().numpy()
                target = batch["flow_y"].detach().cpu().numpy()

                pred = LoadData.recover_data(self.norm_base[0], self.norm_base[1], pred)
                target = LoadData.recover_data(self.norm_base[0], self.norm_base[1], target)

                all_preds.append(pred)
                all_targets.append(target)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        mae, mape, rmse = Evaluation.total(
            all_targets.reshape(-1),
            all_preds.reshape(-1),
        )

        return {
            "mae": float(mae),
            "mape": float(mape),
            "rmse": float(rmse),
        }
