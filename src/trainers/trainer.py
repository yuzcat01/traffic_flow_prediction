import os
import time
import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.datasets.traffic_dataset import LoadData, traffic_batch_collate
from src.models.builder import build_model
from src.project_paths import get_project_root, resolve_project_path
from src.utils.metrics import Evaluation
from src.utils.recorder import append_result, save_run_config
from src.utils.visualize import plot_prediction_overview, plot_prediction_vs_target, plot_loss_curve


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.project_root = get_project_root()

        train_cfg = cfg["train"]
        self.seed = int(train_cfg.get("seed", 42))
        self._set_random_seed(self.seed)

        requested_device = str(cfg["train"].get("device", "auto")).strip().lower()
        if requested_device == "auto":
            use_cuda = torch.cuda.is_available()
        else:
            use_cuda = torch.cuda.is_available() and requested_device == "cuda"
        self.device = torch.device("cuda" if use_cuda else "cpu")

        dataset_cfg = cfg["dataset"]
        model_cfg = cfg["model"]
        self.graph_path = str(resolve_project_path(dataset_cfg["graph_path"], self.project_root))
        self.flow_path = str(resolve_project_path(dataset_cfg["flow_path"], self.project_root))

        self.history_length = model_cfg["input"]["history_length"]
        self.predict_steps = int(model_cfg.get("output", {}).get("predict_steps", 1))
        if self.predict_steps <= 0:
            raise ValueError("model.output.predict_steps must be > 0")
        self.graph_cfg = model_cfg.get("graph", {"type": "connect"})
        self.preprocess_cfg = dataset_cfg.get("preprocess", {})
        self.model_name = model_cfg["name"]
        self.save_dir = str(resolve_project_path(train_cfg["save_dir"], self.project_root))

        self.one_day_length = int(24 * 60 / dataset_cfg["time_interval"])
        self.total_train_steps = dataset_cfg["divide_days"][0] * self.one_day_length
        estimated_full_train_sample_num = self.total_train_steps - self.history_length - self.predict_steps + 1

        self.val_ratio = float(train_cfg.get("val_ratio", 0.1))
        self.early_stop_patience = int(train_cfg.get("early_stop_patience", 8))
        self.early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 1e-4))
        self.figure_horizon_step = int(train_cfg.get("figure_horizon_step", 0))
        self.figure_horizon_step = min(max(self.figure_horizon_step, 0), self.predict_steps - 1)

        # normalization only uses the estimated training segment
        norm_end_t = max(estimated_full_train_sample_num, 1) + self.history_length + self.predict_steps - 1

        self.full_train_data = LoadData(
            data_path=[self.graph_path, self.flow_path],
            num_nodes=dataset_cfg["num_nodes"],
            divide_days=dataset_cfg["divide_days"],
            time_interval=dataset_cfg["time_interval"],
            history_length=self.history_length,
            predict_steps=self.predict_steps,
            train_mode="train",
            graph_cfg=self.graph_cfg,
            preprocess_cfg=self.preprocess_cfg,
            norm_source_range=(0, norm_end_t),
        )

        self.full_train_sample_num = len(self.full_train_data)
        if self.full_train_sample_num <= 0:
            raise ValueError(
                "No valid training samples. Please check divide_days/history_length/predict_steps "
                f"(train_total_steps={self.full_train_data.train_total_steps}, "
                f"history_length={self.history_length}, predict_steps={self.predict_steps})"
            )

        if self.val_ratio > 0 and self.full_train_sample_num >= 2:
            self.val_sample_num = max(1, int(self.full_train_sample_num * self.val_ratio))
            self.val_sample_num = min(self.val_sample_num, self.full_train_sample_num - 1)
            self.train_sample_num = self.full_train_sample_num - self.val_sample_num
        else:
            self.val_sample_num = 0
            self.train_sample_num = self.full_train_sample_num

        self.norm_base = self.full_train_data.flow_norm

        train_indices = list(range(0, self.train_sample_num))
        self.train_data = Subset(self.full_train_data, train_indices)

        if self.val_sample_num > 0:
            val_indices = list(range(self.train_sample_num, self.full_train_sample_num))
            self.val_data = Subset(self.full_train_data, val_indices)
        else:
            self.val_data = None

        self.test_data = LoadData(
            data_path=[self.graph_path, self.flow_path],
            num_nodes=dataset_cfg["num_nodes"],
            divide_days=dataset_cfg["divide_days"],
            time_interval=dataset_cfg["time_interval"],
            history_length=self.history_length,
            predict_steps=self.predict_steps,
            train_mode="test",
            graph_cfg=self.graph_cfg,
            preprocess_cfg=self.preprocess_cfg,
            norm_base=self.norm_base,
        )

        self.loader_num_workers = int(train_cfg.get("num_workers", 0))
        self.pin_memory = self.device.type == "cuda"
        self.persistent_workers = self.loader_num_workers > 0

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=train_cfg["batch_size"],
            shuffle=train_cfg["shuffle"],
            num_workers=self.loader_num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=traffic_batch_collate,
        )

        self.val_loader = None
        if self.val_data is not None:
            self.val_loader = DataLoader(
                self.val_data,
                batch_size=train_cfg["batch_size"],
                shuffle=False,
                num_workers=self.loader_num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=traffic_batch_collate,
            )

        self.test_loader = DataLoader(
            self.test_data,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=self.loader_num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=traffic_batch_collate,
        )

        self.model = build_model(cfg).to(self.device)
        self.loss_fn = str(train_cfg.get("loss_fn", "mse")).strip().lower()
        self.huber_delta = float(train_cfg.get("huber_delta", 1.0))
        if self.loss_fn not in {"mse", "mae", "huber"}:
            raise ValueError("train.loss_fn must be one of: mse, mae, huber")

        self.horizon_weight_mode = str(train_cfg.get("horizon_weight_mode", "uniform")).strip().lower()
        self.horizon_weight_gamma = float(train_cfg.get("horizon_weight_gamma", 0.9))
        if self.horizon_weight_mode not in {"uniform", "linear_decay", "exp_decay", "custom"}:
            self.horizon_weight_mode = "uniform"
        self.horizon_weights = self._build_horizon_weights(
            mode=self.horizon_weight_mode,
            predict_steps=self.predict_steps,
            gamma=self.horizon_weight_gamma,
            custom_raw=train_cfg.get("horizon_weights", []),
        )
        self.horizon_weights_list = [float(x) for x in self.horizon_weights.detach().cpu().numpy().tolist()]

        self.learning_rate = float(train_cfg["learning_rate"])
        self.weight_decay = float(train_cfg.get("weight_decay", 0.0))
        optimizer_name = str(train_cfg.get("optimizer", "adamw")).strip().lower()
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError("train.optimizer must be one of: adam, adamw")

        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
        self.scheduler_type = str(train_cfg.get("lr_scheduler", "plateau")).strip().lower()
        if self.scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(train_cfg.get("lr_scheduler_factor", 0.5)),
                patience=int(train_cfg.get("lr_scheduler_patience", 3)),
                min_lr=float(train_cfg.get("min_lr", 1e-5)),
            )
        elif self.scheduler_type in {"none", ""}:
            self.scheduler = None
        else:
            raise ValueError("train.lr_scheduler must be one of: plateau, none")

        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        fig_dir = os.path.join(self.save_dir, "figures")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        self.best_ckpt_path = os.path.join(ckpt_dir, f"{self.model_name}_best.pth")
        self.last_ckpt_path = os.path.join(ckpt_dir, f"{self.model_name}_last.pth")
        self.loss_curve_path = os.path.join(fig_dir, f"{self.model_name}_loss_curve.png")
        self.pred_fig_path = os.path.join(
            fig_dir,
            f"{self.model_name}_prediction_overview.png"
        )
        self.pred_focus_fig_path = os.path.join(
            fig_dir,
            f"{self.model_name}_node{train_cfg['figure_node_id']}_h{self.figure_horizon_step + 1}_prediction.png"
        )

        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def _raise_if_non_finite_loss(loss, stage, pred=None, target=None):
        if torch.isfinite(loss).item():
            return

        detail_parts = [f"{stage} loss became non-finite"]

        if pred is not None:
            pred_finite = bool(torch.isfinite(pred).all().item())
            detail_parts.append(f"pred_finite={pred_finite}")
            if pred.numel() > 0:
                pred_abs_max = float(torch.nan_to_num(pred.detach(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().item())
                detail_parts.append(f"pred_abs_max={pred_abs_max:.6f}")

        if target is not None:
            target_finite = bool(torch.isfinite(target).all().item())
            detail_parts.append(f"target_finite={target_finite}")
            if target.numel() > 0:
                target_abs_max = float(torch.nan_to_num(target.detach(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().item())
                detail_parts.append(f"target_abs_max={target_abs_max:.6f}")

        raise RuntimeError(" | ".join(detail_parts))

    def _move_batch_to_device(self, data):
        non_blocking = self.pin_memory
        return {
            "graph": data["graph"].to(self.device, non_blocking=non_blocking),
            "flow_x": data["flow_x"].to(self.device, non_blocking=non_blocking),
            "flow_y": data["flow_y"].to(self.device, non_blocking=non_blocking),
        }

    @staticmethod
    def _set_random_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _parse_custom_horizon_weights(raw_value):
        if raw_value is None:
            return []

        items = []
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if text == "":
                return []
            items = [token.strip() for token in text.replace(";", ",").split(",")]
        elif isinstance(raw_value, (list, tuple)):
            items = list(raw_value)
        else:
            raise ValueError("train.horizon_weights must be a list/tuple or comma-separated string")

        result = []
        for item in items:
            if item is None:
                continue
            token = str(item).strip()
            if token == "":
                continue
            value = float(token)
            if value <= 0:
                raise ValueError("all train.horizon_weights values must be > 0")
            result.append(value)

        return result

    def _build_horizon_weights(self, mode, predict_steps, gamma, custom_raw):
        valid_modes = {"uniform", "linear_decay", "exp_decay", "custom"}
        if mode not in valid_modes:
            mode = "uniform"

        if predict_steps <= 1:
            return torch.tensor([1.0], dtype=torch.float32)

        if mode == "uniform":
            weights = np.ones(predict_steps, dtype=np.float32)
        elif mode == "linear_decay":
            weights = np.arange(predict_steps, 0, -1, dtype=np.float32)
        elif mode == "exp_decay":
            if gamma <= 0:
                raise ValueError("train.horizon_weight_gamma must be > 0 when horizon_weight_mode=exp_decay")
            weights = np.power(gamma, np.arange(predict_steps, dtype=np.float32)).astype(np.float32)
        else:
            custom_weights = self._parse_custom_horizon_weights(custom_raw)
            if len(custom_weights) != predict_steps:
                raise ValueError(
                    f"train.horizon_weights length must equal predict_steps ({predict_steps}) when horizon_weight_mode=custom"
                )
            weights = np.array(custom_weights, dtype=np.float32)

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            raise ValueError("horizon weights sum must be > 0")

        weights = weights / weight_sum
        return torch.tensor(weights, dtype=torch.float32)

    def _compute_loss(self, pred, target):
        if self.loss_fn == "mse":
            elementwise = torch.pow(pred - target, 2)
        elif self.loss_fn == "mae":
            elementwise = torch.abs(pred - target)
        else:
            elementwise = F.huber_loss(pred, target, reduction="none", delta=self.huber_delta)

        if elementwise.dim() != 4 or self.predict_steps <= 1:
            return elementwise.mean()

        # [B, N, H, D] -> [H]
        loss_by_horizon = elementwise.mean(dim=(0, 1, 3))
        weights = self.horizon_weights.to(loss_by_horizon.device)
        if weights.numel() != loss_by_horizon.numel():
            weights = torch.full_like(loss_by_horizon, 1.0 / loss_by_horizon.numel())

        return torch.sum(loss_by_horizon * weights)

    def _evaluate_loss_only(self, loader):
        if len(loader) == 0:
            raise ValueError("empty loader in _evaluate_loss_only")
        self.model.eval()
        total_loss = 0.0

        with torch.inference_mode():
            for data in loader:
                batch = self._move_batch_to_device(data)
                pred = self.model(batch)
                target = batch["flow_y"]
                loss = self._compute_loss(pred, target)
                self._raise_if_non_finite_loss(loss, stage="validation", pred=pred, target=target)
                total_loss += loss.item()

        return total_loss / len(loader)

    def _evaluate_with_metrics(self, loader):
        if len(loader) == 0:
            raise ValueError("empty loader in _evaluate_with_metrics")
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.inference_mode():
            for data in loader:
                batch = self._move_batch_to_device(data)
                pred = self.model(batch)
                target = batch["flow_y"]

                loss = self._compute_loss(pred, target)
                self._raise_if_non_finite_loss(loss, stage="test", pred=pred, target=target)
                total_loss += loss.item()

                pred_np = pred.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()

                pred_np = LoadData.recover_data(self.norm_base[0], self.norm_base[1], pred_np)
                target_np = LoadData.recover_data(self.norm_base[0], self.norm_base[1], target_np)
                if not np.isfinite(pred_np).all():
                    raise RuntimeError("test prediction contains non-finite values after denormalization")
                if not np.isfinite(target_np).all():
                    raise RuntimeError("test target contains non-finite values after denormalization")

                all_preds.append(pred_np)
                all_targets.append(target_np)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        mae, mape, rmse = Evaluation.total(
            all_targets.reshape(-1),
            all_preds.reshape(-1)
        )

        horizon_metrics = []
        for h in range(all_preds.shape[2]):
            h_mae, h_mape, h_rmse = Evaluation.total(
                all_targets[:, :, h, :].reshape(-1),
                all_preds[:, :, h, :].reshape(-1),
            )
            horizon_metrics.append(
                {
                    "horizon_index": int(h),
                    "horizon_step": int(h + 1),
                    "mae": float(h_mae),
                    "mape": float(h_mape),
                    "rmse": float(h_rmse),
                }
            )

        return {
            "loss": total_loss / len(loader),
            "mae": float(mae),
            "mape": float(mape),
            "rmse": float(rmse),
            "horizon_metrics": horizon_metrics,
            "preds": all_preds,
            "targets": all_targets,
        }

    def train(self):
        print("device =", self.device)
        print(f"loss_fn = {self.loss_fn}")
        print(f"predict_steps = {self.predict_steps}")
        print(f"seed = {self.seed}")
        print(f"optimizer = {self.optimizer.__class__.__name__} (lr={self.learning_rate}, weight_decay={self.weight_decay})")
        print(f"grad_clip_norm = {self.grad_clip_norm}")
        print(f"lr_scheduler = {self.scheduler_type}")
        print(f"horizon_weight_mode = {self.horizon_weight_mode}")
        print(f"horizon_weights(normalized) = {self.horizon_weights_list}")
        print("\n========== Start Training ==========")

        best_val_loss = float("inf")
        bad_epoch_count = 0
        total_train_start = time.perf_counter()

        epochs = self.cfg["train"]["epochs"]

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_start = time.perf_counter()
            samples_seen = 0

            for data in self.train_loader:
                self.optimizer.zero_grad(set_to_none=True)

                batch = self._move_batch_to_device(data)
                pred = self.model(batch)
                target = batch["flow_y"]

                loss = self._compute_loss(pred, target)
                self._raise_if_non_finite_loss(loss, stage="train", pred=pred, target=target)
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()

                epoch_loss += loss.item()
                samples_seen += int(batch["flow_x"].size(0))

            train_loss = epoch_loss / len(self.train_loader)
            self.train_losses.append(train_loss)

            current_lr = float(self.optimizer.param_groups[0].get("lr", self.learning_rate))
            epoch_seconds = max(time.perf_counter() - epoch_start, 1e-8)
            samples_per_sec = samples_seen / epoch_seconds
            log_msg = (
                f"Epoch {epoch + 1:03d} | LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.6f} | Time: {epoch_seconds:.2f}s | "
                f"Samples/s: {samples_per_sec:.1f}"
            )

            if self.val_loader is not None:
                val_start = time.perf_counter()
                val_loss = self._evaluate_loss_only(self.val_loader)
                val_seconds = time.perf_counter() - val_start
                self.val_losses.append(val_loss)
                log_msg += f" | Val Loss: {val_loss:.6f} | Val Time: {val_seconds:.2f}s"

                if val_loss < best_val_loss - self.early_stop_min_delta:
                    best_val_loss = val_loss
                    bad_epoch_count = 0
                    torch.save(self.model.state_dict(), self.best_ckpt_path)
                    log_msg += " | best saved"
                else:
                    bad_epoch_count += 1
                    log_msg += f" | patience {bad_epoch_count}/{self.early_stop_patience}"

                print(log_msg)

                if bad_epoch_count >= self.early_stop_patience:
                    print("Early stopping triggered.")
                    break
            else:
                print(log_msg)

            if self.scheduler is not None:
                metric = self.val_losses[-1] if len(self.val_losses) > 0 else train_loss
                self.scheduler.step(metric)

        torch.save(self.model.state_dict(), self.last_ckpt_path)
        print(f"last model saved to {self.last_ckpt_path}")
        print(f"training finished in {time.perf_counter() - total_train_start:.2f}s")

        plot_loss_curve(
            train_losses=self.train_losses,
            val_losses=self.val_losses if len(self.val_losses) > 0 else None,
            save_path=self.loss_curve_path,
            title=f"{self.model_name} Loss Curve"
        )
        print(f"loss curve saved to {self.loss_curve_path}")

        if self.val_loader is not None and os.path.exists(self.best_ckpt_path):
            try:
                state_dict = torch.load(self.best_ckpt_path, map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(self.best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"best model loaded from {self.best_ckpt_path}")

    def test(self):
        print("\n========== Start Testing ==========")

        if len(self.test_data) <= 0:
            raise ValueError(
                "No valid test samples available. Please check divide_days/history_length/predict_steps "
                f"(train_total_steps={self.test_data.train_total_steps}, "
                f"test_total_steps={self.test_data.test_total_steps}, "
                f"history_length={self.history_length}, predict_steps={self.predict_steps})"
            )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        test_start = time.perf_counter()
        test_result = self._evaluate_with_metrics(self.test_loader)
        test_seconds = max(time.perf_counter() - test_start, 1e-8)
        test_samples = int(test_result["preds"].shape[0])

        print(f"Test Loss: {test_result['loss']:.6f}")
        print(
            "Performance | "
            f"MAE: {test_result['mae']:.4f} | "
            f"MAPE: {test_result['mape'] * 100:.4f}% | "
            f"RMSE: {test_result['rmse']:.4f}"
        )
        print(f"Test Time: {test_seconds:.2f}s | Samples/s: {test_samples / test_seconds:.1f}")

        node_id = self.cfg["train"]["figure_node_id"]
        figure_points = self.cfg["train"]["figure_points"]
        horizon_step = self.figure_horizon_step

        pred_curve = test_result["preds"][:, node_id, horizon_step, 0]
        target_curve = test_result["targets"][:, node_id, horizon_step, 0]

        plot_prediction_vs_target(
            target=target_curve[:figure_points],
            prediction=pred_curve[:figure_points],
            save_path=self.pred_focus_fig_path,
            title=f"{self.model_name} Prediction vs Target (Node {node_id}, Horizon {horizon_step + 1})"
        )
        print(f"focused prediction figure saved to {self.pred_focus_fig_path}")

        plot_prediction_overview(
            targets=test_result["targets"],
            predictions=test_result["preds"],
            save_path=self.pred_fig_path,
            title=f"{self.model_name} Prediction Overview",
            figure_points=figure_points,
            focus_node_id=node_id,
            focus_horizon_step=horizon_step,
        )
        print(f"prediction overview saved to {self.pred_fig_path}")

        horizon_metrics_dir = os.path.join(self.save_dir, "horizon_metrics")
        os.makedirs(horizon_metrics_dir, exist_ok=True)
        horizon_metrics_path = os.path.join(horizon_metrics_dir, f"{self.model_name}.json")
        with open(horizon_metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "predict_steps": self.predict_steps,
                    "horizons": test_result.get("horizon_metrics", []),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"horizon metrics saved to {horizon_metrics_path}")

        peak_gpu_mb = 0.0
        if self.device.type == "cuda":
            peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024

        config_path = save_run_config(self.save_dir, self.model_name, self.cfg)

        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.model_name,
            "graph_type": self.cfg["model"]["graph"]["type"],
            "spatial_type": self.cfg["model"]["spatial"]["type"],
            "temporal_type": self.cfg["model"]["temporal"]["type"],
            "loss_fn": self.loss_fn,
            "horizon_weight_mode": self.horizon_weight_mode,
            "horizon_weight_gamma": self.horizon_weight_gamma,
            "horizon_weights": ",".join([f"{w:.6f}" for w in self.horizon_weights_list]),
            "predict_steps": self.predict_steps,
            "val_ratio": self.val_ratio,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_min_delta": self.early_stop_min_delta,
            "correlation_topk": self.cfg["model"]["graph"].get("correlation_topk", ""),
            "correlation_threshold": self.cfg["model"]["graph"].get("correlation_threshold", ""),
            "use_abs_corr": self.cfg["model"]["graph"].get("use_abs_corr", ""),
            "fusion_alpha": self.cfg["model"]["graph"].get("fusion_alpha", ""),
            "history_length": self.cfg["model"]["input"]["history_length"],
            "batch_size": self.cfg["train"]["batch_size"],
            "learning_rate": self.cfg["train"]["learning_rate"],
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer.__class__.__name__,
            "lr_scheduler": self.scheduler_type,
            "epochs": self.cfg["train"]["epochs"],
            "figure_horizon_step": self.figure_horizon_step,
            "num_params": self.num_params,
            "peak_gpu_mb": round(peak_gpu_mb, 2),
            "mae": round(test_result["mae"], 4),
            "mape": round(test_result["mape"] * 100, 4),
            "rmse": round(test_result["rmse"], 4),
            "ckpt_path": self.best_ckpt_path if os.path.exists(self.best_ckpt_path) else self.last_ckpt_path,
            "fig_path": self.pred_fig_path,
            "horizon_metrics_path": horizon_metrics_path,
        }
        append_result(self.save_dir, record)

        print(f"run config saved to {config_path}")
        print(f"result appended to {os.path.join(self.save_dir, 'metrics_summary.csv')}")
