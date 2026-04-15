from pathlib import Path
from typing import List, Dict, Optional
import yaml


class ConfigService:
    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root).resolve()

    def _list_yaml(self, relative_dir: str) -> List[Dict[str, str]]:
        folder = self.project_root / relative_dir
        if not folder.exists():
            return []

        items = []
        for path in sorted(folder.glob("*.yaml")):
            items.append({
                "name": path.name,
                "path": str(path),
                "relative_path": str(path.relative_to(self.project_root)),
            })
        return items

    def list_data_configs(self) -> List[Dict[str, str]]:
        return self._list_yaml("configs/data")

    def list_train_configs(self) -> List[Dict[str, str]]:
        return self._list_yaml("configs/train")

    def list_model_configs(self) -> List[Dict[str, str]]:
        return self._list_yaml("configs/model")

    def _write_yaml(self, relative_path: str, payload: dict, overwrite: bool = False) -> Dict[str, str]:
        target = (self.project_root / relative_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists() and not overwrite:
            return {"path": str(target), "status": "exists"}

        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        return {"path": str(target), "status": "created"}

    def create_default_train_config(self, config_name: str = "default_generated", overwrite: bool = False) -> Dict[str, str]:
        name = (config_name or "").strip()
        if not name:
            name = "default_generated"
        if not name.endswith(".yaml"):
            name = f"{name}.yaml"

        payload = {
            "train": {
                "epochs": 50,
                "batch_size": 64,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "optimizer": "adamw",
                "grad_clip_norm": 5.0,
                "lr_scheduler": "plateau",
                "lr_scheduler_factor": 0.5,
                "lr_scheduler_patience": 3,
                "min_lr": 0.00001,
                "seed": 42,
                "num_workers": 0,
                "shuffle": True,
                "device": "auto",
                "val_ratio": 0.1,
                "early_stop_patience": 8,
                "early_stop_min_delta": 0.0001,
                "save_dir": "results",
                "figure_node_id": 0,
                "figure_points": 300,
                "figure_horizon_step": 0,
                "loss_fn": "mse",
                "huber_delta": 1.0,
                "horizon_weight_mode": "uniform",
                "horizon_weight_gamma": 0.9,
                "horizon_weights": [],
            }
        }
        return self._write_yaml(f"configs/train/{name}", payload, overwrite=overwrite)

    def create_default_model_configs(self, overwrite: bool = False) -> List[Dict[str, str]]:
        templates = [
            (
                "configs/model/gcn_gru_generated.yaml",
                {
                    "model": {
                        "name": "gcn_gru_generated",
                        "graph": {"type": "connect"},
                        "input": {"history_length": 12, "input_dim": 1},
                        "spatial": {"type": "gcn", "hidden_dim": 16},
                        "temporal": {"type": "gru", "hidden_dim": 32, "num_layers": 1},
                        "regularization": {"dropout": 0.10},
                        "output": {
                            "output_dim": 1,
                            "predict_steps": 1,
                            "head_type": "horizon_mlp",
                            "pred_hidden_dim": 64,
                            "horizon_emb_dim": 8,
                            "dropout": 0.10,
                            "use_last_value_residual": True,
                        },
                    }
                },
            ),
            (
                "configs/model/gcn_gru_corr_generated.yaml",
                {
                    "model": {
                        "name": "gcn_gru_corr_generated",
                        "graph": {
                            "type": "correlation",
                            "correlation_topk": 8,
                            "correlation_threshold": 0.35,
                            "use_abs_corr": False,
                        },
                        "input": {"history_length": 12, "input_dim": 1},
                        "spatial": {"type": "gcn", "hidden_dim": 16},
                        "temporal": {"type": "gru", "hidden_dim": 32, "num_layers": 1},
                        "regularization": {"dropout": 0.10},
                        "output": {
                            "output_dim": 1,
                            "predict_steps": 1,
                            "head_type": "horizon_mlp",
                            "pred_hidden_dim": 64,
                            "horizon_emb_dim": 8,
                            "dropout": 0.10,
                            "use_last_value_residual": True,
                        },
                    }
                },
            ),
        ]

        result = []
        for rel_path, payload in templates:
            result.append(self._write_yaml(rel_path, payload, overwrite=overwrite))
        return result
