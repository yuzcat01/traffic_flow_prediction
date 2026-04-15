from datetime import datetime
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import csv


class ExperimentReportService:
    def __init__(self, project_root: Optional[str] = None, results_dir: str = "results"):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root).resolve()

        self.results_dir = (self.project_root / results_dir).resolve()
        self.reports_dir = (self.results_dir / "reports").resolve()
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def create_report_dir(self, prefix: str = "report") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = (self.reports_dir / f"{prefix}_{timestamp}").resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    def copy_file_if_exists(self, src_path: str, dst_dir: Path) -> str:
        if not src_path:
            return ""

        src = Path(src_path).resolve()
        if not src.exists() or not src.is_file():
            return ""

        dst = (dst_dir / src.name).resolve()
        shutil.copy2(src, dst)
        return str(dst)

    def save_table_csv(self, rows: List[Dict], headers: List[str], csv_path: Path):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({h: row.get(h, "") for h in headers})

    def generate_markdown_report(
        self,
        report_dir: Path,
        title: str,
        current_model_row: Optional[Dict],
        ranking_rows: List[Dict],
        ranking_meta: Dict,
        selected_rows: List[Dict],
        ranking_chart_file: str = "",
        selected_chart_file: str = "",
        current_pred_fig_file: str = "",
        current_loss_fig_file: str = "",
        baseline_rows: Optional[List[Dict]] = None,
        baseline_metric: str = "rmse_mean",
        baseline_chart_file: str = "",
    ) -> Path:
        def _safe_float(v, default=1e18):
            try:
                return float(v)
            except Exception:
                return default

        lines = []
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"- Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- Group Metric: {ranking_meta.get('metric', '')}")
        lines.append(f"- Group By: {ranking_meta.get('group_by', '')}")
        lines.append(f"- Aggregate: {ranking_meta.get('agg', '')}")
        lines.append("")

        lines.append("## Current Model")
        lines.append("")
        if current_model_row:
            keys = [
                "model_name", "graph_type", "spatial_type", "temporal_type",
                "loss_fn", "history_length", "batch_size", "learning_rate", "epochs",
                "mae", "mape", "rmse", "num_params", "time",
            ]
            for k in keys:
                lines.append(f"- {k}: {current_model_row.get(k, '')}")
        else:
            lines.append("- (No current model loaded)")
        lines.append("")

        if current_pred_fig_file:
            lines.append("### Current Prediction Figure")
            lines.append("")
            lines.append(f"![current_prediction]({Path(current_pred_fig_file).name})")
            lines.append("")

        if current_loss_fig_file:
            lines.append("### Current Loss Figure")
            lines.append("")
            lines.append(f"![current_loss]({Path(current_loss_fig_file).name})")
            lines.append("")

        lines.append("## Group Ranking")
        lines.append("")
        if ranking_rows:
            lines.append("| Rank | Group | Score | Count |")
            lines.append("|---|---|---:|---:|")
            for idx, row in enumerate(ranking_rows, start=1):
                lines.append(f"| {idx} | {row.get('name', '')} | {float(row.get('score', 0.0)):.4f} | {int(row.get('count', 0))} |")
        else:
            lines.append("- (No ranking data)")
        lines.append("")

        if ranking_chart_file:
            lines.append("### Group Ranking Chart")
            lines.append("")
            lines.append(f"![group_ranking_chart]({Path(ranking_chart_file).name})")
            lines.append("")

        lines.append("## Selected Models Comparison")
        lines.append("")
        if selected_rows:
            lines.append("| Model | Graph | Spatial | Temporal | MAE | MAPE | RMSE |")
            lines.append("|---|---|---|---|---:|---:|---:|")
            for row in selected_rows:
                lines.append(
                    f"| {row.get('model_name', '')} | {row.get('graph_type', '')} | {row.get('spatial_type', '')} | "
                    f"{row.get('temporal_type', '')} | {float(row.get('mae', 0.0)):.4f} | "
                    f"{float(row.get('mape', 0.0)):.4f} | {float(row.get('rmse', 0.0)):.4f} |"
                )
        else:
            lines.append("- (No selected models)")
        lines.append("")

        if selected_chart_file:
            lines.append("### Selected Models Chart")
            lines.append("")
            lines.append(f"![selected_models_chart]({Path(selected_chart_file).name})")
            lines.append("")

        lines.append("## Multi-Seed Baseline Summary")
        lines.append("")
        baseline_rows = baseline_rows or []
        if baseline_rows:
            sorted_rows = sorted(
                baseline_rows,
                key=lambda r: _safe_float(r.get(baseline_metric, 1e18)),
            )

            lines.append("| Rank | Base Model | Runs | Success | MAE(mean+/-std) | MAPE(mean+/-std) | RMSE(mean+/-std) |")
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            for idx, row in enumerate(sorted_rows, start=1):
                mae_mean = float(row.get("mae_mean", 0.0))
                mae_std = float(row.get("mae_std", 0.0))
                mape_mean = float(row.get("mape_mean", 0.0))
                mape_std = float(row.get("mape_std", 0.0))
                rmse_mean = float(row.get("rmse_mean", 0.0))
                rmse_std = float(row.get("rmse_std", 0.0))
                lines.append(
                    f"| {idx} | {row.get('base_model', '')} | {row.get('runs_total', 0)} | {row.get('runs_success', 0)} | "
                    f"{mae_mean:.4f}+/-{mae_std:.4f} | {mape_mean:.4f}+/-{mape_std:.4f} | {rmse_mean:.4f}+/-{rmse_std:.4f} |"
                )

            best = sorted_rows[0]
            lines.append("")
            lines.append("### Baseline Conclusion")
            lines.append("")
            lines.append(
                f"- Best base model by {baseline_metric}: **{best.get('base_model', '')}**"
            )
            lines.append(
                f"- RMSE mean+/-std: **{float(best.get('rmse_mean', 0.0)):.4f}+/-{float(best.get('rmse_std', 0.0)):.4f}**"
            )
            lines.append(
                f"- MAE mean+/-std: **{float(best.get('mae_mean', 0.0)):.4f}+/-{float(best.get('mae_std', 0.0)):.4f}**"
            )
            lines.append(
                f"- MAPE mean+/-std: **{float(best.get('mape_mean', 0.0)):.4f}+/-{float(best.get('mape_std', 0.0)):.4f}**"
            )
            lines.append(
                f"- Supporting runs: {best.get('runs_success', 0)}/{best.get('runs_total', 0)}"
            )
        else:
            lines.append("- (No multi-seed baseline summary data)")
        lines.append("")

        if baseline_chart_file:
            lines.append("### Baseline Summary Chart")
            lines.append("")
            lines.append(f"![baseline_summary_chart]({Path(baseline_chart_file).name})")
            lines.append("")

        md_path = (report_dir / "report.md").resolve()
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return md_path
