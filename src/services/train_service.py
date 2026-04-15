import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Dict, Any

import yaml

from utils.config import load_yaml


def deep_update_dict(base: dict, updates: dict):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update_dict(base[k], v)
        else:
            base[k] = v
    return base


class TrainService:
    def __init__(
        self,
        data_cfg: str,
        train_cfg: str,
        model_cfg: str,
        overrides: Optional[Dict[str, Any]] = None,
        project_root: Optional[str] = None,
    ):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root).resolve()

        self.data_cfg = str(Path(data_cfg).resolve())
        self.train_cfg = str(Path(train_cfg).resolve())
        self.model_cfg = str(Path(model_cfg).resolve())
        self.overrides = overrides or {}

        self.process: Optional[subprocess.Popen] = None
        self._stop_requested = False

    def _build_temp_configs(self):
        data_cfg = load_yaml(self.data_cfg)
        train_cfg = load_yaml(self.train_cfg)
        model_cfg = load_yaml(self.model_cfg)

        merged_data_cfg = deepcopy(data_cfg)
        merged_train_cfg = deepcopy(train_cfg)
        merged_model_cfg = deepcopy(model_cfg)

        # 覆盖 data / train / model 参数
        if "dataset" in self.overrides:
            deep_update_dict(merged_data_cfg.setdefault("dataset", {}), self.overrides["dataset"])
        if "train" in self.overrides:
            deep_update_dict(merged_train_cfg.setdefault("train", {}), self.overrides["train"])
        if "model" in self.overrides:
            deep_update_dict(merged_model_cfg.setdefault("model", {}), self.overrides["model"])

        # 自动改实验名，避免覆盖旧结果
        base_model_name = merged_model_cfg.get("model", {}).get("name", "experiment")
        run_suffix = self.overrides.get("meta", {}).get("run_suffix", "").strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if run_suffix:
            final_model_name = f"{base_model_name}_{run_suffix}_{timestamp}"
        else:
            final_model_name = f"{base_model_name}_{timestamp}"

        merged_model_cfg["model"]["name"] = final_model_name

        temp_dir = tempfile.TemporaryDirectory(prefix="traffic_train_gui_")
        temp_path = Path(temp_dir.name)

        data_cfg_path = temp_path / "data_cfg.yaml"
        train_cfg_path = temp_path / "train_cfg.yaml"
        model_cfg_path = temp_path / "model_cfg.yaml"

        with open(data_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged_data_cfg, f, allow_unicode=True, sort_keys=False)

        with open(train_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged_train_cfg, f, allow_unicode=True, sort_keys=False)

        with open(model_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged_model_cfg, f, allow_unicode=True, sort_keys=False)

        return temp_dir, str(data_cfg_path), str(train_cfg_path), str(model_cfg_path), final_model_name

    def build_command(self, data_cfg_path: str, train_cfg_path: str, model_cfg_path: str):
        return [
            sys.executable,
            "-u",
            "train.py",
            "--data_cfg", data_cfg_path,
            "--train_cfg", train_cfg_path,
            "--model_cfg", model_cfg_path,
        ]

    def run(self, line_callback: Optional[Callable[[str], None]] = None):
        temp_dir, data_cfg_path, train_cfg_path, model_cfg_path, final_model_name = self._build_temp_configs()
        cmd = self.build_command(data_cfg_path, train_cfg_path, model_cfg_path)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        if line_callback is not None:
            line_callback(">>> 启动训练命令：")
            line_callback(" ".join(cmd))
            line_callback(f">>> 实际实验名：{final_model_name}")
            line_callback("")

        try:
            if self.process.stdout is not None:
                for raw_line in self.process.stdout:
                    line = raw_line.rstrip("\n")
                    if line_callback is not None:
                        line_callback(line)

            return_code = self.process.wait()

            if self._stop_requested:
                return {
                    "status": "stopped",
                    "return_code": return_code,
                    "model_name": final_model_name,
                }

            if return_code != 0:
                raise RuntimeError(f"训练进程退出异常，return code = {return_code}")

            return {
                "status": "success",
                "return_code": return_code,
                "model_name": final_model_name,
            }

        finally:
            self.process = None
            temp_dir.cleanup()

    def stop(self):
        self._stop_requested = True

        if self.process is None:
            return

        if self.process.poll() is not None:
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
