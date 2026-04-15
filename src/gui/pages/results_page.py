import csv
import json
import os
from collections import defaultdict

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from services.model_registry import ModelRegistry
from services.report_service import ExperimentReportService


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(320)


class ResultsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.registry = ModelRegistry(results_dir="results")
        self.report_service = ExperimentReportService(results_dir="results")

        self.current_model_row = None
        self._pred_fig_path = ""
        self._loss_fig_path = ""

        self._all_rows = []
        self._last_compare_rows = []
        self._last_compare_metric = ""
        self._last_compare_group = ""
        self._last_compare_agg = ""
        self._last_horizon_rows = []
        self._last_horizon_metric = ""
        self._last_horizon_index = 0
        self._baseline_rows = []

        self._init_ui()
        self.refresh_compare_view()
        self.refresh_baseline_summary()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        scroll_content = QWidget()
        scroll_root = QVBoxLayout(scroll_content)
        scroll_root.setContentsMargins(0, 0, 0, 0)
        scroll_root.setSpacing(0)

        panel = QFrame()
        panel.setObjectName("PagePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("结果分析")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        desc = QLabel("查看当前模型摘要、分组排名对比、多模型对比，并支持一键导出实验报告。")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #9ac2e8; line-height: 1.6;")

        top_layout = QHBoxLayout()
        top_layout.setSpacing(16)

        summary_group = QGroupBox("当前模型摘要")
        summary_layout = QVBoxLayout(summary_group)

        self.text_results_info = QTextEdit()
        self.text_results_info.setReadOnly(True)
        self.text_results_info.setMinimumHeight(180)
        self.text_results_info.setMaximumHeight(220)
        self.text_results_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        summary_layout.addWidget(self.text_results_info)

        metric_group = QGroupBox("当前模型指标")
        metric_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        metric_layout = QGridLayout(metric_group)
        metric_layout.setHorizontalSpacing(20)
        metric_layout.setVerticalSpacing(12)

        self.label_mae = self._make_metric_value("-")
        self.label_mape = self._make_metric_value("-")
        self.label_rmse = self._make_metric_value("-")
        self.label_params = self._make_metric_value("-")

        metric_layout.addWidget(self._make_metric_title("MAE"), 0, 0)
        metric_layout.addWidget(self.label_mae, 1, 0)
        metric_layout.addWidget(self._make_metric_title("MAPE"), 0, 1)
        metric_layout.addWidget(self.label_mape, 1, 1)
        metric_layout.addWidget(self._make_metric_title("RMSE"), 2, 0)
        metric_layout.addWidget(self.label_rmse, 3, 0)
        metric_layout.addWidget(self._make_metric_title("参数量"), 2, 1)
        metric_layout.addWidget(self.label_params, 3, 1)

        struct_group = QGroupBox("模型结构卡片")
        struct_layout = QVBoxLayout(struct_group)
        self.text_model_struct = QTextEdit()
        self.text_model_struct.setReadOnly(True)
        self.text_model_struct.setMinimumHeight(180)
        self.text_model_struct.setMaximumHeight(220)
        self.text_model_struct.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        struct_layout.addWidget(self.text_model_struct)

        top_layout.addWidget(summary_group, 2)
        top_layout.addWidget(metric_group, 1)
        top_layout.addWidget(struct_group, 2)

        compare_group = QGroupBox("分组对比设置")
        compare_layout = QGridLayout(compare_group)
        compare_layout.setHorizontalSpacing(12)
        compare_layout.setVerticalSpacing(10)

        self.combo_metric = QComboBox()
        self.combo_metric.addItems(["rmse", "mae", "mape"])
        self.combo_metric.setMaximumWidth(180)

        self.combo_group_by = QComboBox()
        self.combo_group_by.addItems(["spatial_type", "temporal_type", "graph_type", "loss_fn", "model_name"])
        self.combo_group_by.setMaximumWidth(220)

        self.combo_agg = QComboBox()
        self.combo_agg.addItems(["best", "mean"])
        self.combo_agg.setMaximumWidth(160)

        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(3, 50)
        self.spin_topk.setValue(8)
        self.spin_topk.setMaximumWidth(120)

        self.btn_refresh_compare = QPushButton("刷新分组对比")
        self.btn_export_compare_csv = QPushButton("导出分组排名CSV")
        self.btn_export_compare_chart = QPushButton("导出分组图PNG")

        self.btn_refresh_compare.clicked.connect(self.refresh_compare_view)
        self.btn_export_compare_csv.clicked.connect(self.export_compare_csv)
        self.btn_export_compare_chart.clicked.connect(self.export_compare_chart)

        self.combo_metric.currentIndexChanged.connect(self.refresh_compare_view)
        self.combo_group_by.currentIndexChanged.connect(self.refresh_compare_view)
        self.combo_agg.currentIndexChanged.connect(self.refresh_compare_view)
        self.spin_topk.valueChanged.connect(self.refresh_compare_view)

        compare_layout.addWidget(QLabel("指标:"), 0, 0)
        compare_layout.addWidget(self.combo_metric, 0, 1)
        compare_layout.addWidget(QLabel("分组字段:"), 0, 2)
        compare_layout.addWidget(self.combo_group_by, 0, 3)

        compare_layout.addWidget(QLabel("聚合方式:"), 1, 0)
        compare_layout.addWidget(self.combo_agg, 1, 1)
        compare_layout.addWidget(QLabel("Top-N:"), 1, 2)
        compare_layout.addWidget(self.spin_topk, 1, 3)

        compare_layout.addWidget(self.btn_refresh_compare, 0, 4, 2, 1)
        compare_layout.addWidget(self.btn_export_compare_csv, 0, 5)
        compare_layout.addWidget(self.btn_export_compare_chart, 1, 5)

        compare_layout.setColumnStretch(1, 1)
        compare_layout.setColumnStretch(3, 1)

        compare_vis_layout = QHBoxLayout()
        compare_vis_layout.setSpacing(16)

        chart_group = QGroupBox("分组柱状图")
        chart_layout = QVBoxLayout(chart_group)
        self.canvas_compare = MplCanvas(self, width=8, height=4, dpi=100)
        chart_layout.addWidget(self.canvas_compare)

        ranking_group = QGroupBox("分组排名文本")
        ranking_layout = QVBoxLayout(ranking_group)
        self.text_ranking = QTextEdit()
        self.text_ranking.setReadOnly(True)
        self.text_ranking.setMinimumHeight(320)
        ranking_layout.addWidget(self.text_ranking)

        compare_vis_layout.addWidget(chart_group, 3)
        compare_vis_layout.addWidget(ranking_group, 2)

        baseline_group = QGroupBox("基线多随机种子汇总")
        baseline_layout = QVBoxLayout(baseline_group)

        baseline_control = QHBoxLayout()
        self.combo_baseline_metric = QComboBox()
        self.combo_baseline_metric.addItems(["rmse_mean", "mae_mean", "mape_mean"])
        self.combo_baseline_metric.setMaximumWidth(200)
        self.spin_baseline_topk = QSpinBox()
        self.spin_baseline_topk.setRange(3, 50)
        self.spin_baseline_topk.setValue(8)
        self.spin_baseline_topk.setMaximumWidth(120)
        self.btn_refresh_baseline = QPushButton("刷新基线汇总")
        self.btn_export_baseline = QPushButton("导出基线CSV")
        self.btn_refresh_baseline.clicked.connect(self.refresh_baseline_summary)
        self.btn_export_baseline.clicked.connect(self.export_baseline_summary_csv)
        self.combo_baseline_metric.currentIndexChanged.connect(self.refresh_baseline_summary)
        self.spin_baseline_topk.valueChanged.connect(self.refresh_baseline_summary)

        baseline_control.addWidget(QLabel("排名指标:"))
        baseline_control.addWidget(self.combo_baseline_metric)
        baseline_control.addSpacing(10)
        baseline_control.addWidget(QLabel("Top-N:"))
        baseline_control.addWidget(self.spin_baseline_topk)
        baseline_control.addSpacing(10)
        baseline_control.addWidget(self.btn_refresh_baseline)
        baseline_control.addWidget(self.btn_export_baseline)
        baseline_control.addStretch()

        self.table_baseline = QTableWidget()
        self.table_baseline.setColumnCount(8)
        self.table_baseline.setHorizontalHeaderLabels([
            "排名", "基础模型", "运行次数", "成功次数", "MAE mean+/-std", "MAPE mean+/-std", "RMSE mean+/-std", "最佳模型"
        ])
        self.table_baseline.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_baseline.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_baseline.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_baseline.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_baseline.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for idx in [2, 3, 4, 5, 6]:
            self.table_baseline.horizontalHeader().setSectionResizeMode(idx, QHeaderView.ResizeToContents)
        self.table_baseline.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        self.table_baseline.setMinimumHeight(220)

        baseline_vis = QHBoxLayout()
        self.canvas_baseline = MplCanvas(self, width=8, height=3.6, dpi=100)
        self.text_baseline = QTextEdit()
        self.text_baseline.setReadOnly(True)
        self.text_baseline.setMinimumHeight(260)
        baseline_vis.addWidget(self.canvas_baseline, 3)
        baseline_vis.addWidget(self.text_baseline, 2)

        baseline_layout.addLayout(baseline_control)
        baseline_layout.addWidget(self.table_baseline)
        baseline_layout.addLayout(baseline_vis)

        model_compare_group = QGroupBox("多模型对比")
        model_compare_layout = QVBoxLayout(model_compare_group)

        model_control = QHBoxLayout()
        self.spin_model_pool = QSpinBox()
        self.spin_model_pool.setRange(3, 50)
        self.spin_model_pool.setValue(8)
        self.spin_model_pool.setMaximumWidth(120)

        self.combo_model_metric = QComboBox()
        self.combo_model_metric.addItems(["rmse", "mae", "mape"])
        self.combo_model_metric.setMaximumWidth(160)

        self.btn_fill_pool = QPushButton("刷新候选集")
        self.btn_select_best = QPushButton("选择Top 3")
        self.btn_compare_selected = QPushButton("对比已选模型")
        self.btn_export_selected_csv = QPushButton("导出已选CSV")
        self.btn_generate_report = QPushButton("生成报告包")

        self.btn_fill_pool.clicked.connect(self.refresh_model_pool)
        self.btn_select_best.clicked.connect(self.select_top_models)
        self.btn_compare_selected.clicked.connect(self.compare_selected_models)
        self.btn_export_selected_csv.clicked.connect(self.export_selected_models_csv)
        self.btn_generate_report.clicked.connect(self.generate_report_bundle)

        model_control.addWidget(QLabel("候选数量:"))
        model_control.addWidget(self.spin_model_pool)
        model_control.addSpacing(10)
        model_control.addWidget(QLabel("对比指标:"))
        model_control.addWidget(self.combo_model_metric)
        model_control.addSpacing(10)
        model_control.addWidget(self.btn_fill_pool)
        model_control.addWidget(self.btn_select_best)
        model_control.addWidget(self.btn_compare_selected)
        model_control.addWidget(self.btn_export_selected_csv)
        model_control.addWidget(self.btn_generate_report)
        model_control.addStretch()

        horizon_control = QHBoxLayout()
        self.combo_horizon_metric = QComboBox()
        self.combo_horizon_metric.addItems(["rmse", "mae", "mape"])
        self.combo_horizon_metric.setMaximumWidth(160)
        self.spin_horizon_index = QSpinBox()
        self.spin_horizon_index.setRange(1, 1)
        self.spin_horizon_index.setMaximumWidth(120)
        self.btn_compare_horizon = QPushButton("按步长对比")
        self.btn_export_horizon_csv = QPushButton("导出步长CSV")

        self.btn_compare_horizon.clicked.connect(self.compare_selected_models_by_horizon)
        self.btn_export_horizon_csv.clicked.connect(self.export_horizon_compare_csv)

        horizon_control.addWidget(QLabel("步长指标:"))
        horizon_control.addWidget(self.combo_horizon_metric)
        horizon_control.addSpacing(10)
        horizon_control.addWidget(QLabel("预测步:"))
        horizon_control.addWidget(self.spin_horizon_index)
        horizon_control.addSpacing(10)
        horizon_control.addWidget(self.btn_compare_horizon)
        horizon_control.addWidget(self.btn_export_horizon_csv)
        horizon_control.addStretch()

        self.table_models_pick = QTableWidget()
        self.table_models_pick.setColumnCount(9)
        self.table_models_pick.setHorizontalHeaderLabels([
            "选择", "模型名", "图类型", "空间模块", "时间模块", "MAE", "MAPE", "RMSE", "时间"
        ])
        self.table_models_pick.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_models_pick.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_models_pick.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_models_pick.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_models_pick.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for idx in [2, 3, 4, 5, 6, 7]:
            self.table_models_pick.horizontalHeader().setSectionResizeMode(idx, QHeaderView.ResizeToContents)
        self.table_models_pick.horizontalHeader().setSectionResizeMode(8, QHeaderView.Stretch)
        self.table_models_pick.setMinimumHeight(220)

        model_chart_layout = QHBoxLayout()
        self.canvas_models = MplCanvas(self, width=8, height=4, dpi=100)
        self.text_models_summary = QTextEdit()
        self.text_models_summary.setReadOnly(True)
        self.text_models_summary.setMinimumHeight(280)
        model_chart_layout.addWidget(self.canvas_models, 3)
        model_chart_layout.addWidget(self.text_models_summary, 2)

        horizon_chart_layout = QHBoxLayout()
        self.canvas_horizon = MplCanvas(self, width=8, height=4, dpi=100)
        self.text_horizon_summary = QTextEdit()
        self.text_horizon_summary.setReadOnly(True)
        self.text_horizon_summary.setMinimumHeight(280)
        horizon_chart_layout.addWidget(self.canvas_horizon, 3)
        horizon_chart_layout.addWidget(self.text_horizon_summary, 2)

        model_compare_layout.addLayout(model_control)
        model_compare_layout.addLayout(horizon_control)
        model_compare_layout.addWidget(self.table_models_pick)
        model_compare_layout.addLayout(model_chart_layout)
        model_compare_layout.addLayout(horizon_chart_layout)

        fig_group = QGroupBox("当前模型图像")
        fig_layout = QHBoxLayout(fig_group)
        fig_layout.setSpacing(16)

        self.label_pred_fig = QLabel("预测图")
        self.label_pred_fig.setAlignment(Qt.AlignCenter)
        self.label_pred_fig.setMinimumHeight(300)
        self.label_pred_fig.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_pred_fig.setStyleSheet("border: 1px solid #d1d5db; border-radius: 8px; background: #ffffff;")

        self.label_loss_fig = QLabel("损失曲线")
        self.label_loss_fig.setAlignment(Qt.AlignCenter)
        self.label_loss_fig.setMinimumHeight(300)
        self.label_loss_fig.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_loss_fig.setStyleSheet("border: 1px solid #d1d5db; border-radius: 8px; background: #ffffff;")

        fig_layout.addWidget(self.label_pred_fig, 1)
        fig_layout.addWidget(self.label_loss_fig, 1)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addLayout(top_layout)
        layout.addWidget(compare_group)
        layout.addLayout(compare_vis_layout)
        layout.addWidget(baseline_group)
        layout.addWidget(model_compare_group)
        layout.addWidget(fig_group)
        layout.addStretch(1)

        scroll_root.addWidget(panel)
        scroll.setWidget(scroll_content)
        root.addWidget(scroll)

    def _make_metric_title(self, text: str):
        label = QLabel(text)
        label.setStyleSheet("color: #9ac2e8; font-size: 12px;")
        label.setAlignment(Qt.AlignCenter)
        return label

    def _make_metric_value(self, text: str):
        label = QLabel(text)
        label.setStyleSheet("font-size: 24px; font-weight: bold; color: #f2fbff;")
        label.setAlignment(Qt.AlignCenter)
        return label

    def set_model_row(self, row):
        self.current_model_row = row
        self.update_view()

    @staticmethod
    def _safe_float_text(value, digits=4):
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return "-"

    def _load_run_config(self, row):
        cfg_path = str(row.get("run_config_path", "")).strip()
        if not cfg_path or not os.path.exists(cfg_path):
            return {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _build_struct_text(self, row):
        cfg = self._load_run_config(row)
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}

        spatial = model_cfg.get("spatial", {})
        temporal = model_cfg.get("temporal", {})
        output = model_cfg.get("output", {})
        regular = model_cfg.get("regularization", {})

        lines = [
            f"空间模块: {spatial.get('type', row.get('spatial_type', '-'))}",
            f"空间隐藏维度: {spatial.get('hidden_dim', '-')}",
            f"Cheb K: {spatial.get('cheb_k', '-')}",
            f"GAT heads: {spatial.get('heads', '-')}",
            "",
            f"时间模块: {temporal.get('type', row.get('temporal_type', '-'))}",
            f"时间隐藏维度: {temporal.get('hidden_dim', '-')}",
            f"时间层数: {temporal.get('num_layers', '-')}",
            "",
            f"输出头类型: {output.get('head_type', '-')}",
            f"预测步数: {output.get('predict_steps', row.get('predict_steps', '-'))}",
            f"预测头维度: {output.get('pred_hidden_dim', '-')}",
            f"步长嵌入维度: {output.get('horizon_emb_dim', '-')}",
            f"末值残差: {output.get('use_last_value_residual', '-')}",
            f"模型dropout: {self._safe_float_text(output.get('dropout', regular.get('dropout', '-')), 3)}",
            "",
            f"优化器: {row.get('optimizer', train_cfg.get('optimizer', '-'))}",
            f"调度器: {row.get('lr_scheduler', train_cfg.get('lr_scheduler', '-'))}",
            f"权重衰减: {self._safe_float_text(row.get('weight_decay', train_cfg.get('weight_decay', '-')), 6)}",
            f"随机种子: {train_cfg.get('seed', '-')}",
        ]
        return "\n".join(lines)

    def update_view(self):
        row = self.current_model_row
        if row is None:
            self.text_results_info.setPlainText("未加载模型。")
            self.label_mae.setText("-")
            self.label_mape.setText("-")
            self.label_rmse.setText("-")
            self.label_params.setText("-")
            self.text_model_struct.setPlainText("未加载模型。")
            self._pred_fig_path = ""
            self._loss_fig_path = ""
            self._set_image_to_label(self.label_pred_fig, "")
            self._set_image_to_label(self.label_loss_fig, "")
            return

        result_text = [
            f"模型名: {row.get('model_name', '')}",
            f"图类型: {row.get('graph_type', '')}",
            f"空间模块: {row.get('spatial_type', '')}",
            f"时间模块: {row.get('temporal_type', '')}",
            f"损失函数: {row.get('loss_fn', '-')}",
            f"多步权重模式: {row.get('horizon_weight_mode', '-')}",
            f"多步权重Gamma: {row.get('horizon_weight_gamma', '-')}",
            f"多步权重: {row.get('horizon_weights', '-')}",
            f"历史长度: {row.get('history_length', '')}",
            f"预测步数: {row.get('predict_steps', '')}",
            f"批大小: {row.get('batch_size', '')}",
            f"学习率: {row.get('learning_rate', '')}",
            f"训练轮数: {row.get('epochs', '')}",
            f"图像步索引: {row.get('figure_horizon_step', '')}",
            f"分步指标路径: {row.get('horizon_metrics_path', '')}",
            f"参数量: {row.get('num_params', '')}",
            f"峰值显存(MB): {row.get('peak_gpu_mb', '')}",
            f"MAE: {row.get('mae', 0.0):.4f}",
            f"MAPE(%): {row.get('mape', 0.0):.4f}",
            f"RMSE: {row.get('rmse', 0.0):.4f}",
            f"Checkpoint: {row.get('ckpt_path', '')}",
            f"运行配置: {row.get('run_config_path', '')}",
            f"实验时间: {row.get('time', '')}",
        ]
        self.text_results_info.setPlainText("\n".join(result_text))

        self.label_mae.setText(f"{row.get('mae', 0.0):.4f}")
        self.label_mape.setText(f"{row.get('mape', 0.0):.4f}")
        self.label_rmse.setText(f"{row.get('rmse', 0.0):.4f}")
        self.label_params.setText(str(row.get("num_params", "-")))
        self.text_model_struct.setPlainText(self._build_struct_text(row))

        self._pred_fig_path = os.path.normpath(row.get("fig_path", ""))
        self._loss_fig_path = os.path.normpath(os.path.join("results", "figures", f"{row.get('model_name', '')}_loss_curve.png"))

        self._set_image_to_label(self.label_pred_fig, self._pred_fig_path)
        self._set_image_to_label(self.label_loss_fig, self._loss_fig_path)

        self.refresh_compare_view()

    def refresh_compare_view(self):
        rows = self.registry.list_models(sort_by="rmse")
        self._all_rows = rows

        ax = self.canvas_compare.ax
        ax.clear()

        if not rows:
            ax.set_title("No experiment records")
            self.canvas_compare.draw()
            self.text_ranking.setPlainText("No experiment records.")
            self._last_compare_rows = []
            self.refresh_model_pool()
            return

        metric_key = self.combo_metric.currentText()
        group_key = self.combo_group_by.currentText()
        agg_type = self.combo_agg.currentText()
        topk = self.spin_topk.value()

        grouped = defaultdict(list)
        for row in rows:
            group_name = str(row.get(group_key, "unknown"))
            value = row.get(metric_key, None)
            if value is None:
                continue
            try:
                grouped[group_name].append(float(value))
            except Exception:
                continue

        full_stats = []
        for name, values in grouped.items():
            if not values:
                continue
            score = min(values) if agg_type == "best" else float(sum(values) / len(values))
            full_stats.append({"name": name, "score": score, "count": len(values)})

        full_stats.sort(key=lambda x: x["score"])
        stats = full_stats[:topk]

        self._last_compare_rows = full_stats
        self._last_compare_metric = metric_key
        self._last_compare_group = group_key
        self._last_compare_agg = agg_type

        if not stats:
            ax.set_title("No valid grouped comparison result")
            self.canvas_compare.draw()
            self.text_ranking.setPlainText("No valid grouped comparison result.")
            self.refresh_model_pool()
            return

        names = [item["name"] for item in stats]
        scores = [item["score"] for item in stats]

        ax.bar(range(len(names)), scores, color="#3b82f6")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel(metric_key.upper())
        ax.set_title(f"Top-{len(stats)} by {metric_key.upper()} | group={group_key} | agg={agg_type}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        self.canvas_compare.figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.28)
        self.canvas_compare.draw()

        lines = [f"[分组对比] metric={metric_key}, group_by={group_key}, agg={agg_type}", ""]

        current_group = None
        if self.current_model_row is not None:
            current_group = str(self.current_model_row.get(group_key, "unknown"))

        if current_group is not None:
            rank = None
            score = None
            for idx, item in enumerate(full_stats, start=1):
                if item["name"] == current_group:
                    rank = idx
                    score = item["score"]
                    break
            lines.append(f"当前模型分组: {current_group}")
            if rank is not None:
                lines.append(f"当前分组排名: {rank}/{len(full_stats)} | {metric_key.upper()}={score:.4f}")
            lines.append("")

        lines.append("[Top 排名]")
        for idx, item in enumerate(stats, start=1):
            lines.append(f"{idx}. {item['name']} | {metric_key.upper()}={item['score']:.4f} | count={item['count']}")

        self.text_ranking.setPlainText("\n".join(lines))
        self.refresh_model_pool()
        self.refresh_baseline_summary()

    @staticmethod
    def _fmt_mean_std(mean_v, std_v):
        try:
            return f"{float(mean_v):.4f} ± {float(std_v):.4f}"
        except Exception:
            return "-"

    def refresh_baseline_summary(self):
        baseline_csv = os.path.join("results", "baseline_summary.csv")
        rows = []

        if os.path.exists(baseline_csv):
            try:
                with open(baseline_csv, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row["mae_mean"] = float(row.get("mae_mean", "nan"))
                        row["mae_std"] = float(row.get("mae_std", "nan"))
                        row["mape_mean"] = float(row.get("mape_mean", "nan"))
                        row["mape_std"] = float(row.get("mape_std", "nan"))
                        row["rmse_mean"] = float(row.get("rmse_mean", "nan"))
                        row["rmse_std"] = float(row.get("rmse_std", "nan"))
                        row["runs_total"] = int(float(row.get("runs_total", 0)))
                        row["runs_success"] = int(float(row.get("runs_success", 0)))
                        rows.append(row)
            except Exception:
                rows = []

        self._baseline_rows = rows
        ax = self.canvas_baseline.ax
        ax.clear()

        if not rows:
            self.table_baseline.setRowCount(0)
            self.text_baseline.setPlainText("未找到 baseline_summary.csv，请先运行 `python run_all.py --seeds 42,2026,3407`。")
            ax.set_title("No baseline summary")
            self.canvas_baseline.draw()
            return

        metric = self.combo_baseline_metric.currentText().strip()
        topk = self.spin_baseline_topk.value()
        rows = sorted(rows, key=lambda r: float(r.get(metric, float("inf"))))
        show_rows = rows[:topk]

        self.table_baseline.setRowCount(len(show_rows))
        for r, row in enumerate(show_rows):
            values = [
                str(r + 1),
                str(row.get("base_model", "")),
                str(row.get("runs_total", "")),
                str(row.get("runs_success", "")),
                self._fmt_mean_std(row.get("mae_mean"), row.get("mae_std")),
                self._fmt_mean_std(row.get("mape_mean"), row.get("mape_std")),
                self._fmt_mean_std(row.get("rmse_mean"), row.get("rmse_std")),
                str(row.get("best_model_name", "")),
            ]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setToolTip(v)
                self.table_baseline.setItem(r, c, item)

        names = [str(r.get("base_model", "")) for r in show_rows]
        vals = [float(r.get(metric, 0.0)) for r in show_rows]
        x = np.arange(len(names))
        ax.bar(x, vals, color="#6366f1")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=18, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Baseline Top-{len(show_rows)} by {metric.upper()}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        self.canvas_baseline.figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.30)
        self.canvas_baseline.draw()

        lines = [f"[基线汇总] metric={metric}", ""]
        for idx, row in enumerate(show_rows, start=1):
            lines.append(
                f"{idx}. {row.get('base_model', '')} | "
                f"RMSE={self._fmt_mean_std(row.get('rmse_mean'), row.get('rmse_std'))} | "
                f"MAE={self._fmt_mean_std(row.get('mae_mean'), row.get('mae_std'))} | "
                f"MAPE={self._fmt_mean_std(row.get('mape_mean'), row.get('mape_std'))}"
            )
        self.text_baseline.setPlainText("\n".join(lines))

    def refresh_model_pool(self):
        rows = self._all_rows
        pool = min(self.spin_model_pool.value(), len(rows))
        rows = rows[:pool]

        self.table_models_pick.setRowCount(len(rows))
        for r, row in enumerate(rows):
            check_item = QTableWidgetItem("")
            check_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check_item.setCheckState(Qt.Unchecked)
            self.table_models_pick.setItem(r, 0, check_item)

            vals = [
                row.get("model_name", ""),
                row.get("graph_type", ""),
                row.get("spatial_type", ""),
                row.get("temporal_type", ""),
                f"{float(row.get('mae', 0.0)):.4f}",
                f"{float(row.get('mape', 0.0)):.4f}",
                f"{float(row.get('rmse', 0.0)):.4f}",
                row.get("time", ""),
            ]
            for c, v in enumerate(vals, start=1):
                item = QTableWidgetItem(str(v))
                item.setToolTip(str(v))
                self.table_models_pick.setItem(r, c, item)

        self._update_horizon_index_range(rows)

    def _update_horizon_index_range(self, rows):
        max_steps = 1
        for row in rows:
            try:
                max_steps = max(max_steps, int(row.get("predict_steps", 1)))
            except Exception:
                continue
        self.spin_horizon_index.setRange(1, max_steps)
        if self.spin_horizon_index.value() > max_steps:
            self.spin_horizon_index.setValue(max_steps)

    def _load_horizon_metric_for_row(self, row, horizon_index, metric_key):
        path = str(row.get("horizon_metrics_path", "")).strip()
        if not path or not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None

        horizons = payload.get("horizons", [])
        if not isinstance(horizons, list) or horizon_index < 0 or horizon_index >= len(horizons):
            return None

        try:
            return float(horizons[horizon_index].get(metric_key, None))
        except Exception:
            return None

    def select_top_models(self):
        rows = self.table_models_pick.rowCount()
        for r in range(rows):
            item = self.table_models_pick.item(r, 0)
            if item is None:
                continue
            item.setCheckState(Qt.Checked if r < 3 else Qt.Unchecked)
        self.compare_selected_models()

    def _get_selected_model_rows(self):
        selected = []
        for r in range(self.table_models_pick.rowCount()):
            check_item = self.table_models_pick.item(r, 0)
            if check_item is None or check_item.checkState() != Qt.Checked:
                continue
            if r < len(self._all_rows):
                selected.append(self._all_rows[r])
        return selected

    def compare_selected_models(self):
        rows = self._get_selected_model_rows()
        ax = self.canvas_models.ax
        ax.clear()

        if not rows:
            ax.set_title("Please select models in table")
            self.canvas_models.draw()
            self.text_models_summary.setPlainText("未选择模型。")
            return

        metric = self.combo_model_metric.currentText()
        names = [str(r.get("model_name", "")) for r in rows]
        values = [float(r.get(metric, 0.0)) for r in rows]

        x = np.arange(len(names))
        ax.bar(x, values, color="#10b981")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=18, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Selected Models by {metric.upper()}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        self.canvas_models.figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.30)
        self.canvas_models.draw()

        lines = [f"[已选模型对比] metric={metric}", ""]
        ranking = sorted(rows, key=lambda r: float(r.get(metric, 0.0)))
        for idx, row in enumerate(ranking, start=1):
            lines.append(
                f"{idx}. {row.get('model_name', '')} | "
                f"MAE={float(row.get('mae', 0.0)):.4f} | "
                f"MAPE={float(row.get('mape', 0.0)):.4f} | "
                f"RMSE={float(row.get('rmse', 0.0)):.4f}"
            )
        self.text_models_summary.setPlainText("\n".join(lines))
        self.compare_selected_models_by_horizon()

    def compare_selected_models_by_horizon(self):
        rows = self._get_selected_model_rows()
        ax = self.canvas_horizon.ax
        ax.clear()

        if not rows:
            ax.set_title("Please select models in table")
            self.canvas_horizon.draw()
            self.text_horizon_summary.setPlainText("未选择模型。")
            self._last_horizon_rows = []
            return

        self._update_horizon_index_range(rows)

        metric_key = self.combo_horizon_metric.currentText()
        horizon_index = self.spin_horizon_index.value() - 1
        valid = []

        for row in rows:
            value = self._load_horizon_metric_for_row(row, horizon_index, metric_key)
            if value is None:
                continue
            valid.append(
                {
                    "model_name": str(row.get("model_name", "")),
                    "metric": metric_key,
                    "horizon_index": int(horizon_index),
                    "horizon_step": int(horizon_index + 1),
                    "value": float(value),
                }
            )

        self._last_horizon_rows = sorted(valid, key=lambda x: x["value"])
        self._last_horizon_metric = metric_key
        self._last_horizon_index = horizon_index

        if not self._last_horizon_rows:
            ax.set_title("No horizon metrics file or missing horizon data")
            self.canvas_horizon.draw()
            self.text_horizon_summary.setPlainText(
                "没有可用的分步指标数据，请在开启 horizon metrics 后重新训练模型。"
            )
            return

        names = [item["model_name"] for item in self._last_horizon_rows]
        values = [item["value"] for item in self._last_horizon_rows]
        x = np.arange(len(names))
        ax.bar(x, values, color="#f59e0b")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=18, ha="right")
        ax.set_ylabel(metric_key.upper())
        ax.set_title(f"Horizon Step {horizon_index + 1} | {metric_key.upper()} Comparison")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        self.canvas_horizon.figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.30)
        self.canvas_horizon.draw()

        lines = [f"[Horizon Comparison] metric={metric_key}, horizon_step={horizon_index + 1}", ""]
        for idx, item in enumerate(self._last_horizon_rows, start=1):
            lines.append(f"{idx}. {item['model_name']} | {metric_key.upper()}={item['value']:.4f}")
        self.text_horizon_summary.setPlainText("\n".join(lines))

    def export_selected_models_csv(self):
        rows = self._get_selected_model_rows()
        if not rows:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出已选模型",
            "selected_models_compare.csv",
            "CSV Files (*.csv)",
        )
        if not save_path:
            return

        headers = [
            "model_name", "graph_type", "spatial_type", "temporal_type", "loss_fn",
            "predict_steps", "mae", "mape", "rmse", "batch_size", "learning_rate", "epochs", "time"
        ]
        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in headers})

    def export_horizon_compare_csv(self):
        if not self._last_horizon_rows:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出步长对比",
            f"horizon_compare_h{self._last_horizon_index + 1}.csv",
            "CSV Files (*.csv)",
        )
        if not save_path:
            return

        headers = ["rank", "model_name", "metric", "horizon_step", "value"]
        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for idx, row in enumerate(self._last_horizon_rows, start=1):
                writer.writerow(
                    {
                        "rank": idx,
                        "model_name": row.get("model_name", ""),
                        "metric": row.get("metric", ""),
                        "horizon_step": row.get("horizon_step", ""),
                        "value": f"{float(row.get('value', 0.0)):.6f}",
                    }
                )

    def export_baseline_summary_csv(self):
        if not self._baseline_rows:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出基线汇总",
            "baseline_summary_export.csv",
            "CSV Files (*.csv)",
        )
        if not save_path:
            return

        headers = [
            "rank", "base_model", "runs_total", "runs_success", "seeds",
            "mae_mean", "mae_std", "mape_mean", "mape_std", "rmse_mean", "rmse_std", "best_model_name",
        ]
        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for idx, row in enumerate(sorted(self._baseline_rows, key=lambda r: float(r.get("rmse_mean", float("inf")))), start=1):
                out = {k: row.get(k, "") for k in headers}
                out["rank"] = idx
                writer.writerow(out)

    def generate_report_bundle(self):
        report_dir = self.report_service.create_report_dir(prefix="experiment_report")

        ranking_chart_path = (report_dir / "group_ranking_chart.png").resolve()
        self.canvas_compare.figure.savefig(str(ranking_chart_path), dpi=180, bbox_inches="tight")

        selected_chart_path = (report_dir / "selected_models_chart.png").resolve()
        self.canvas_models.figure.savefig(str(selected_chart_path), dpi=180, bbox_inches="tight")

        horizon_chart_path = (report_dir / "horizon_compare_chart.png").resolve()
        self.canvas_horizon.figure.savefig(str(horizon_chart_path), dpi=180, bbox_inches="tight")

        baseline_chart_path = (report_dir / "baseline_summary_chart.png").resolve()
        self.canvas_baseline.figure.savefig(str(baseline_chart_path), dpi=180, bbox_inches="tight")

        pred_fig_local = self.report_service.copy_file_if_exists(self._pred_fig_path, report_dir)
        loss_fig_local = self.report_service.copy_file_if_exists(self._loss_fig_path, report_dir)

        selected_rows = self._get_selected_model_rows()
        ranking_rows = self._last_compare_rows[: min(20, len(self._last_compare_rows))]

        ranking_csv = (report_dir / "group_ranking.csv").resolve()
        self.report_service.save_table_csv(
            rows=[
                {
                    "rank": idx,
                    "name": row.get("name", ""),
                    "score": f"{float(row.get('score', 0.0)):.6f}",
                    "count": row.get("count", 0),
                }
                for idx, row in enumerate(ranking_rows, start=1)
            ],
            headers=["rank", "name", "score", "count"],
            csv_path=ranking_csv,
        )

        selected_csv = (report_dir / "selected_models.csv").resolve()
        self.report_service.save_table_csv(
            rows=[
                {
                    "model_name": row.get("model_name", ""),
                    "graph_type": row.get("graph_type", ""),
                    "spatial_type": row.get("spatial_type", ""),
                    "temporal_type": row.get("temporal_type", ""),
                    "loss_fn": row.get("loss_fn", ""),
                    "predict_steps": row.get("predict_steps", ""),
                    "mae": f"{float(row.get('mae', 0.0)):.6f}",
                    "mape": f"{float(row.get('mape', 0.0)):.6f}",
                    "rmse": f"{float(row.get('rmse', 0.0)):.6f}",
                    "time": row.get("time", ""),
                }
                for row in selected_rows
            ],
            headers=["model_name", "graph_type", "spatial_type", "temporal_type", "loss_fn", "predict_steps", "mae", "mape", "rmse", "time"],
            csv_path=selected_csv,
        )

        horizon_csv = ""
        if self._last_horizon_rows:
            horizon_csv_path = (report_dir / "horizon_compare.csv").resolve()
            self.report_service.save_table_csv(
                rows=[
                    {
                        "rank": idx,
                        "model_name": row.get("model_name", ""),
                        "metric": row.get("metric", ""),
                        "horizon_step": row.get("horizon_step", ""),
                        "value": f"{float(row.get('value', 0.0)):.6f}",
                    }
                    for idx, row in enumerate(self._last_horizon_rows, start=1)
                ],
                headers=["rank", "model_name", "metric", "horizon_step", "value"],
                csv_path=horizon_csv_path,
            )
            horizon_csv = str(horizon_csv_path)

        baseline_csv = ""
        if self._baseline_rows:
            baseline_csv_path = (report_dir / "baseline_summary.csv").resolve()
            self.report_service.save_table_csv(
                rows=[
                    {
                        "rank": idx,
                        "base_model": row.get("base_model", ""),
                        "runs_total": row.get("runs_total", ""),
                        "runs_success": row.get("runs_success", ""),
                        "mae_mean": row.get("mae_mean", ""),
                        "mae_std": row.get("mae_std", ""),
                        "mape_mean": row.get("mape_mean", ""),
                        "mape_std": row.get("mape_std", ""),
                        "rmse_mean": row.get("rmse_mean", ""),
                        "rmse_std": row.get("rmse_std", ""),
                        "best_model_name": row.get("best_model_name", ""),
                    }
                    for idx, row in enumerate(sorted(self._baseline_rows, key=lambda x: float(x.get("rmse_mean", float("inf")))), start=1)
                ],
                headers=[
                    "rank", "base_model", "runs_total", "runs_success",
                    "mae_mean", "mae_std", "mape_mean", "mape_std", "rmse_mean", "rmse_std", "best_model_name",
                ],
                csv_path=baseline_csv_path,
            )
            baseline_csv = str(baseline_csv_path)

        md_path = self.report_service.generate_markdown_report(
            report_dir=report_dir,
            title="Traffic Flow Prediction Experiment Report",
            current_model_row=self.current_model_row,
            ranking_rows=ranking_rows,
            ranking_meta={
                "metric": self._last_compare_metric,
                "group_by": self._last_compare_group,
                "agg": self._last_compare_agg,
            },
            selected_rows=selected_rows,
            ranking_chart_file=str(ranking_chart_path),
            selected_chart_file=str(selected_chart_path),
            current_pred_fig_file=pred_fig_local,
            current_loss_fig_file=loss_fig_local,
            baseline_rows=self._baseline_rows,
            baseline_metric=self.combo_baseline_metric.currentText().strip(),
            baseline_chart_file=str(baseline_chart_path),
        )

        self.text_models_summary.append("")
        self.text_models_summary.append(f"报告已生成: {md_path}")
        if horizon_csv:
            self.text_models_summary.append(f"步长对比CSV: {horizon_csv}")
        if baseline_csv:
            self.text_models_summary.append(f"基线汇总CSV: {baseline_csv}")

    def export_compare_csv(self):
        if not self._last_compare_rows:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出分组排名",
            "results_compare_ranking.csv",
            "CSV Files (*.csv)",
        )
        if not save_path:
            return

        with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", self._last_compare_group, self._last_compare_metric, "count", "agg"])
            for idx, item in enumerate(self._last_compare_rows, start=1):
                writer.writerow([idx, item["name"], f"{item['score']:.6f}", item["count"], self._last_compare_agg])

    def export_compare_chart(self):
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出分组图",
            "results_compare_chart.png",
            "PNG Files (*.png)",
        )
        if not save_path:
            return

        self.canvas_compare.figure.savefig(save_path, dpi=180, bbox_inches="tight")

    def _set_image_to_label(self, label: QLabel, image_path: str):
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                target_w = max(label.width() - 12, 100)
                target_h = max(label.height() - 12, 100)
                scaled = pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled)
                label.setText("")
                return

        label.setPixmap(QPixmap())
        label.setText(f"未找到图像\n{image_path}" if image_path else "无图像")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "label_pred_fig"):
            self._set_image_to_label(self.label_pred_fig, self._pred_fig_path)
        if hasattr(self, "label_loss_fig"):
            self._set_image_to_label(self.label_loss_fig, self._loss_fig_path)
