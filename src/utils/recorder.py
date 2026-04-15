import csv
import json
import os
import time


def append_result(save_dir: str, record: dict):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics_summary.csv")

    fieldnames = [
        "time",
        "model_name",
        "graph_type",
        "spatial_type",
        "temporal_type",
        "loss_fn",
        "horizon_weight_mode",
        "horizon_weight_gamma",
        "horizon_weights",
        "val_ratio",
        "early_stop_patience",
        "early_stop_min_delta",
        "correlation_topk",
        "correlation_threshold",
        "use_abs_corr",
        "fusion_alpha",
        "predict_steps",
        "history_length",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "lr_scheduler",
        "epochs",
        "figure_horizon_step",
        "num_params",
        "peak_gpu_mb",
        "mae",
        "mape",
        "rmse",
        "ckpt_path",
        "fig_path",
        "horizon_metrics_path",
    ]

    existing_rows = []
    need_rewrite = False

    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                old_fields = reader.fieldnames or []
                if old_fields != fieldnames:
                    need_rewrite = True
                for row in reader:
                    existing_rows.append(row)
        except Exception:
            need_rewrite = True

    normalized_record = {k: record.get(k, "") for k in fieldnames}

    if need_rewrite:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
            writer.writerow(normalized_record)
        return

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(normalized_record)


def save_run_config(save_dir: str, model_name: str, cfg: dict):
    cfg_dir = os.path.join(save_dir, "run_configs")
    os.makedirs(cfg_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cfg_path = os.path.join(cfg_dir, f"{model_name}_{timestamp}.json")

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    return cfg_path
