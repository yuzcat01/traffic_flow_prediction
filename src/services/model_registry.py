import csv
import glob
import os
from typing import Any, Dict, List, Optional


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


class ModelRegistry:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.metrics_csv = os.path.join(results_dir, "metrics_summary.csv")
        self.run_cfg_dir = os.path.join(results_dir, "run_configs")

    def _find_latest_run_config(self, model_name: str) -> Optional[str]:
        pattern = os.path.join(self.run_cfg_dir, f"{model_name}_*.json")
        candidates = glob.glob(pattern)
        if not candidates:
            return None
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    def list_models(self, sort_by: str = "rmse") -> List[Dict[str, Any]]:
        if not os.path.exists(self.metrics_csv):
            return []

        rows: List[Dict[str, Any]] = []
        with open(self.metrics_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["mae"] = _safe_float(row.get("mae"))
                row["mape"] = _safe_float(row.get("mape"))
                row["rmse"] = _safe_float(row.get("rmse"))
                row["peak_gpu_mb"] = _safe_float(row.get("peak_gpu_mb"))
                row["learning_rate"] = _safe_float(row.get("learning_rate"))
                row["weight_decay"] = _safe_float(row.get("weight_decay"))
                row["val_ratio"] = _safe_float(row.get("val_ratio"))
                row["early_stop_min_delta"] = _safe_float(row.get("early_stop_min_delta"))
                row["horizon_weight_gamma"] = _safe_float(row.get("horizon_weight_gamma"), default=0.9)
                row["correlation_threshold"] = _safe_float(row.get("correlation_threshold"))
                row["fusion_alpha"] = _safe_float(row.get("fusion_alpha"))
                row["epochs"] = _safe_int(row.get("epochs"))
                row["history_length"] = _safe_int(row.get("history_length"))
                row["predict_steps"] = _safe_int(row.get("predict_steps"), default=1)
                row["batch_size"] = _safe_int(row.get("batch_size"))
                row["num_params"] = _safe_int(row.get("num_params"))
                row["early_stop_patience"] = _safe_int(row.get("early_stop_patience"))
                row["correlation_topk"] = _safe_int(row.get("correlation_topk"))
                row["figure_horizon_step"] = _safe_int(row.get("figure_horizon_step"))
                row["horizon_weight_mode"] = str(row.get("horizon_weight_mode", "uniform"))
                row["horizon_weights"] = str(row.get("horizon_weights", ""))
                row["horizon_metrics_path"] = str(row.get("horizon_metrics_path", ""))
                row["optimizer"] = str(row.get("optimizer", ""))
                row["lr_scheduler"] = str(row.get("lr_scheduler", ""))

                row["run_config_path"] = self._find_latest_run_config(row["model_name"])
                rows.append(row)

        rows.sort(key=lambda x: x.get(sort_by, float("inf")))
        return rows

    def get_best_model(
        self,
        sort_by: str = "rmse",
        filters: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        rows = self.list_models(sort_by=sort_by)
        if filters:
            filtered = []
            for row in rows:
                ok = True
                for k, v in filters.items():
                    if str(row.get(k)) != str(v):
                        ok = False
                        break
                if ok:
                    filtered.append(row)
            rows = filtered

        return rows[0] if rows else None
