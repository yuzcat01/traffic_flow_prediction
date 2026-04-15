import argparse
import csv
import math
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml


def discover_model_configs(models_dir: Path):
    return sorted([p for p in models_dir.glob("*.yaml") if p.is_file()], key=lambda p: p.name)


def choose_train_cfg(model_cfg: Path, train_default: Path, train_gat: Path):
    return train_gat if model_cfg.name.lower().startswith("gat_") else train_default


def parse_seed_list(text: str) -> List[int]:
    seeds = []
    for token in (text or "").replace(";", ",").split(","):
        token = token.strip()
        if token == "":
            continue
        seeds.append(int(token))
    if not seeds:
        seeds = [42]
    return seeds


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj or {}


def save_yaml(path: Path, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def find_metrics_row(metrics_csv: Path, model_name: str):
    if not metrics_csv.exists():
        return None
    found = None
    with open(metrics_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("model_name", "")).strip() == model_name:
                found = row
    return found


def to_float(v, default=math.nan):
    try:
        return float(v)
    except Exception:
        return default


def mean_std(values: List[float]):
    cleaned = [x for x in values if not math.isnan(x)]
    if not cleaned:
        return math.nan, math.nan
    mean_v = sum(cleaned) / len(cleaned)
    if len(cleaned) <= 1:
        return mean_v, 0.0
    var = sum((x - mean_v) ** 2 for x in cleaned) / (len(cleaned) - 1)
    return mean_v, math.sqrt(var)


def build_parser():
    parser = argparse.ArgumentParser(description="Run baseline experiments with optional multi-seed repeats.")
    parser.add_argument("--data_cfg", type=str, default="configs/data/pems04.yaml")
    parser.add_argument("--models_dir", type=str, default="configs/model")
    parser.add_argument("--train_cfg_default", type=str, default="configs/train/default.yaml")
    parser.add_argument("--train_cfg_gat", type=str, default="configs/train/gat_safe.yaml")
    parser.add_argument(
        "--only_models",
        type=str,
        default="",
        help="Comma-separated model yaml names, e.g. gcn_gru.yaml,gat_gru.yaml",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds, e.g. 42,2026,3407",
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--runs_csv", type=str, default="results/baseline_runs.csv")
    parser.add_argument("--summary_csv", type=str, default="results/baseline_summary.csv")
    parser.add_argument("--fail_fast", action="store_true", help="Stop immediately when one experiment fails.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned commands only, do not execute.")
    return parser


def main():
    args = build_parser().parse_args()

    root = Path(__file__).resolve().parent
    data_cfg = (root / args.data_cfg).resolve()
    models_dir = (root / args.models_dir).resolve()
    train_default = (root / args.train_cfg_default).resolve()
    train_gat = (root / args.train_cfg_gat).resolve()
    results_dir = (root / args.results_dir).resolve()
    metrics_csv = (results_dir / "metrics_summary.csv").resolve()
    runs_csv = (root / args.runs_csv).resolve()
    summary_csv = (root / args.summary_csv).resolve()
    seeds = parse_seed_list(args.seeds)

    if not data_cfg.exists():
        raise FileNotFoundError(f"data cfg not found: {data_cfg}")
    if not models_dir.exists():
        raise FileNotFoundError(f"models dir not found: {models_dir}")
    if not train_default.exists():
        raise FileNotFoundError(f"train cfg default not found: {train_default}")
    if not train_gat.exists():
        raise FileNotFoundError(f"train cfg gat not found: {train_gat}")

    model_cfgs = discover_model_configs(models_dir)
    if not model_cfgs:
        raise RuntimeError(f"no model config found under: {models_dir}")

    if args.only_models.strip():
        wanted = {x.strip() for x in args.only_models.split(",") if x.strip()}
        model_cfgs = [p for p in model_cfgs if p.name in wanted]
        if not model_cfgs:
            raise RuntimeError("no model config matched --only_models")

    jobs = []
    for model_cfg in model_cfgs:
        train_cfg = choose_train_cfg(model_cfg, train_default, train_gat)
        for seed in seeds:
            jobs.append((train_cfg, model_cfg, seed))

    print("=" * 96)
    print(f"Total experiments: {len(jobs)} (models={len(model_cfgs)}, seeds={len(seeds)})")
    for idx, (train_cfg, model_cfg, seed) in enumerate(jobs, start=1):
        print(f"[{idx:02d}] seed={seed} | train={train_cfg.relative_to(root)} | model={model_cfg.relative_to(root)}")

    if args.dry_run:
        print("=" * 96)
        print("Dry run only. No experiment executed.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    run_records = []
    failed_count = 0

    for idx, (train_cfg_path, model_cfg_path, seed) in enumerate(jobs, start=1):
        start = time.time()
        base_model_cfg = load_yaml(model_cfg_path).get("model", {})
        train_cfg_obj = load_yaml(train_cfg_path)
        model_cfg_obj = load_yaml(model_cfg_path)

        base_model_name = str(base_model_cfg.get("name", model_cfg_path.stem))
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{base_model_name}_s{seed}_{run_tag}"

        train_cfg_obj.setdefault("train", {})
        model_cfg_obj.setdefault("model", {})
        train_cfg_obj["train"]["seed"] = int(seed)
        model_cfg_obj["model"]["name"] = model_name

        with tempfile.TemporaryDirectory(prefix="traffic_batch_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            tmp_train = (tmp_dir_path / "train.yaml").resolve()
            tmp_model = (tmp_dir_path / "model.yaml").resolve()
            save_yaml(tmp_train, train_cfg_obj)
            save_yaml(tmp_model, model_cfg_obj)

            cmd = [
                sys.executable,
                "train.py",
                "--data_cfg",
                str(data_cfg),
                "--train_cfg",
                str(tmp_train),
                "--model_cfg",
                str(tmp_model),
            ]

            print("\n" + "=" * 96)
            print(f"[{idx:02d}/{len(jobs):02d}] Running: {' '.join(cmd)}")

            status = "success"
            return_code = 0
            try:
                subprocess.run(cmd, check=True, cwd=str(root))
            except subprocess.CalledProcessError as e:
                status = "failed"
                return_code = int(e.returncode)
                failed_count += 1

            duration = time.time() - start
            metric_row = find_metrics_row(metrics_csv, model_name) if status == "success" else None
            mae = to_float(metric_row.get("mae")) if metric_row else math.nan
            mape = to_float(metric_row.get("mape")) if metric_row else math.nan
            rmse = to_float(metric_row.get("rmse")) if metric_row else math.nan

            rec = {
                "base_model": base_model_name,
                "model_name": model_name,
                "model_cfg": model_cfg_path.name,
                "train_cfg": train_cfg_path.name,
                "seed": int(seed),
                "status": status,
                "return_code": return_code,
                "duration_sec": round(duration, 2),
                "mae": mae,
                "mape": mape,
                "rmse": rmse,
                "time": metric_row.get("time", "") if metric_row else "",
            }
            run_records.append(rec)

            if status == "success":
                print(f"[{idx:02d}] Success in {duration:.1f}s | RMSE={rmse:.4f}")
            else:
                print(f"[{idx:02d}] Failed in {duration:.1f}s, return code={return_code}")
                if args.fail_fast:
                    break

    run_headers = [
        "base_model",
        "model_name",
        "model_cfg",
        "train_cfg",
        "seed",
        "status",
        "return_code",
        "duration_sec",
        "mae",
        "mape",
        "rmse",
        "time",
    ]
    with open(runs_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=run_headers)
        writer.writeheader()
        for row in run_records:
            writer.writerow(row)

    grouped: Dict[str, List[Dict]] = {}
    for row in run_records:
        grouped.setdefault(row["base_model"], []).append(row)

    summary_rows = []
    for base_model, rows in grouped.items():
        success_rows = [r for r in rows if r.get("status") == "success"]
        mae_mean, mae_std = mean_std([to_float(r.get("mae")) for r in success_rows])
        mape_mean, mape_std = mean_std([to_float(r.get("mape")) for r in success_rows])
        rmse_mean, rmse_std = mean_std([to_float(r.get("rmse")) for r in success_rows])
        best_run = min(success_rows, key=lambda r: to_float(r.get("rmse"), default=1e18)) if success_rows else None

        summary_rows.append(
            {
                "base_model": base_model,
                "runs_total": len(rows),
                "runs_success": len(success_rows),
                "seeds": ",".join([str(r.get("seed")) for r in rows]),
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "mape_mean": mape_mean,
                "mape_std": mape_std,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "best_model_name": best_run.get("model_name", "") if best_run else "",
            }
        )

    summary_rows.sort(key=lambda r: to_float(r.get("rmse_mean"), default=1e18))
    for idx, row in enumerate(summary_rows, start=1):
        row["rank"] = idx

    summary_headers = [
        "rank",
        "base_model",
        "runs_total",
        "runs_success",
        "seeds",
        "mae_mean",
        "mae_std",
        "mape_mean",
        "mape_std",
        "rmse_mean",
        "rmse_std",
        "best_model_name",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_headers)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("\n" + "=" * 96)
    print("Batch summary")
    print(f"Runs: {len(run_records)} | Failed: {failed_count}")
    print(f"Run-level CSV: {runs_csv}")
    print(f"Summary CSV:  {summary_csv}")
    if summary_rows:
        top = summary_rows[0]
        print(
            f"Best base model: {top['base_model']} | "
            f"RMSE mean={to_float(top['rmse_mean'], 0.0):.4f} ± {to_float(top['rmse_std'], 0.0):.4f}"
        )

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
