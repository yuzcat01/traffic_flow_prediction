import csv
import json
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.services.model_registry import ModelRegistry


class ModelManagePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.registry = ModelRegistry(results_dir="results")
        self.model_rows = []
        self.filtered_rows = []
        self.current_model_row = None
        self.load_callback = None

        self._init_ui()
        self.refresh_model_table()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        panel = QFrame()
        panel.setObjectName("PagePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("模型管理")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        # ---------------- 筛选区 ----------------
        filter_group = QGroupBox("筛选与检索")
        filter_layout = QGridLayout(filter_group)
        filter_layout.setHorizontalSpacing(12)
        filter_layout.setVerticalSpacing(10)

        self.combo_graph = QComboBox()
        self.combo_spatial = QComboBox()
        self.combo_temporal = QComboBox()
        self.combo_graph.setMaximumWidth(180)
        self.combo_spatial.setMaximumWidth(180)
        self.combo_temporal.setMaximumWidth(180)
        self.edit_keyword = QLineEdit()
        self.edit_keyword.setMaximumWidth(300)
        self.edit_keyword.setPlaceholderText("按 model_name / graph / spatial / temporal 搜索")
        self.check_best_only = QCheckBox("仅显示当前筛选下最佳项")

        self.combo_graph.currentIndexChanged.connect(self.apply_filters)
        self.combo_spatial.currentIndexChanged.connect(self.apply_filters)
        self.combo_temporal.currentIndexChanged.connect(self.apply_filters)
        self.edit_keyword.textChanged.connect(self.apply_filters)
        self.check_best_only.stateChanged.connect(self.apply_filters)

        filter_layout.addWidget(QLabel("graph:"), 0, 0)
        filter_layout.addWidget(self.combo_graph, 0, 1)
        filter_layout.addWidget(QLabel("spatial:"), 0, 2)
        filter_layout.addWidget(self.combo_spatial, 0, 3)

        filter_layout.addWidget(QLabel("temporal:"), 1, 0)
        filter_layout.addWidget(self.combo_temporal, 1, 1)
        filter_layout.addWidget(QLabel("关键词:"), 1, 2)
        filter_layout.addWidget(self.edit_keyword, 1, 3)

        filter_layout.addWidget(self.check_best_only, 2, 0, 1, 2)

        # ---------------- 操作区 ----------------
        btn_layout = QHBoxLayout()
        self.btn_refresh_models = QPushButton("刷新实验记录")
        self.btn_reset_filters = QPushButton("重置筛选")
        self.btn_show_config = QPushButton("查看完整配置")
        self.btn_export_csv = QPushButton("导出当前表格")
        self.btn_load_selected = QPushButton("加载选中模型")
        self.label_current_model = QLabel("当前模型：未加载")
        self.label_current_model.setStyleSheet("font-weight: bold; color: #374151;")

        self.btn_refresh_models.clicked.connect(self.refresh_model_table)
        self.btn_reset_filters.clicked.connect(self.reset_filters)
        self.btn_show_config.clicked.connect(self.show_current_config)
        self.btn_export_csv.clicked.connect(self.export_current_table)
        self.btn_load_selected.clicked.connect(self.load_selected_model)

        btn_layout.addWidget(self.btn_refresh_models)
        btn_layout.addWidget(self.btn_reset_filters)
        btn_layout.addWidget(self.btn_show_config)
        btn_layout.addWidget(self.btn_export_csv)
        btn_layout.addWidget(self.btn_load_selected)
        btn_layout.addStretch()
        btn_layout.addWidget(self.label_current_model)

        # ---------------- 表格 ----------------
        self.table_models = QTableWidget()
        self.table_models.setColumnCount(9)
        self.table_models.setHorizontalHeaderLabels([
            "model_name", "graph", "spatial", "temporal",
            "MAE", "MAPE(%)", "RMSE", "Params", "Run Time"
        ])

        header = self.table_models.horizontalHeader()
        header.setMinimumSectionSize(70)
        header.setStretchLastSection(False)

        header.setSectionResizeMode(0, QHeaderView.Stretch)           # model_name
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # graph
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # spatial
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # temporal
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # MAE
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # MAPE
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # RMSE
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # Params
        header.setSectionResizeMode(8, QHeaderView.Stretch)           # Run Time

        self.table_models.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_models.setSelectionMode(QTableWidget.SingleSelection)
        self.table_models.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_models.setWordWrap(False)
        self.table_models.setTextElideMode(Qt.ElideRight)
        self.table_models.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table_models.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.table_models.setAlternatingRowColors(True)
        self.table_models.itemSelectionChanged.connect(self._on_table_selection_changed)

        self.table_models.setColumnWidth(0, 220)
        self.table_models.setColumnWidth(8, 240)

        # ---------------- 详情区 ----------------
        detail_group = QGroupBox("实验详情")
        detail_layout = QVBoxLayout(detail_group)

        self.text_detail = QTextEdit()
        self.text_detail.setReadOnly(True)
        self.text_detail.setMinimumHeight(260)
        self.text_detail.setPlaceholderText("选择一条实验记录后，将在这里显示详细信息。")
        detail_layout.addWidget(self.text_detail)

        layout.addWidget(title)
        layout.addWidget(filter_group)
        layout.addLayout(btn_layout)
        layout.addWidget(self.table_models)
        layout.addWidget(detail_group)

        root.addWidget(panel)

    def refresh_model_table(self):
        self.model_rows = self.registry.list_models(sort_by="rmse")
        self._populate_filter_options()
        self.apply_filters()

    def _populate_filter_options(self):
        current_graph = self.combo_graph.currentText()
        current_spatial = self.combo_spatial.currentText()
        current_temporal = self.combo_temporal.currentText()

        graph_values = sorted({str(row.get("graph_type", "")) for row in self.model_rows if row.get("graph_type", "")})
        spatial_values = sorted({str(row.get("spatial_type", "")) for row in self.model_rows if row.get("spatial_type", "")})
        temporal_values = sorted({str(row.get("temporal_type", "")) for row in self.model_rows if row.get("temporal_type", "")})

        self.combo_graph.blockSignals(True)
        self.combo_spatial.blockSignals(True)
        self.combo_temporal.blockSignals(True)

        self.combo_graph.clear()
        self.combo_spatial.clear()
        self.combo_temporal.clear()

        self.combo_graph.addItem("全部")
        self.combo_spatial.addItem("全部")
        self.combo_temporal.addItem("全部")

        self.combo_graph.addItems(graph_values)
        self.combo_spatial.addItems(spatial_values)
        self.combo_temporal.addItems(temporal_values)

        self._restore_combo_value(self.combo_graph, current_graph)
        self._restore_combo_value(self.combo_spatial, current_spatial)
        self._restore_combo_value(self.combo_temporal, current_temporal)

        self.combo_graph.blockSignals(False)
        self.combo_spatial.blockSignals(False)
        self.combo_temporal.blockSignals(False)

    def _restore_combo_value(self, combo: QComboBox, value: str):
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentIndex(0)

    def apply_filters(self):
        graph_filter = self.combo_graph.currentText().strip()
        spatial_filter = self.combo_spatial.currentText().strip()
        temporal_filter = self.combo_temporal.currentText().strip()
        keyword = self.edit_keyword.text().strip().lower()
        best_only = self.check_best_only.isChecked()

        rows = list(self.model_rows)

        if graph_filter and graph_filter != "全部":
            rows = [r for r in rows if str(r.get("graph_type", "")) == graph_filter]

        if spatial_filter and spatial_filter != "全部":
            rows = [r for r in rows if str(r.get("spatial_type", "")) == spatial_filter]

        if temporal_filter and temporal_filter != "全部":
            rows = [r for r in rows if str(r.get("temporal_type", "")) == temporal_filter]

        if keyword:
            rows = [
                r for r in rows
                if keyword in str(r.get("model_name", "")).lower()
                or keyword in str(r.get("graph_type", "")).lower()
                or keyword in str(r.get("spatial_type", "")).lower()
                or keyword in str(r.get("temporal_type", "")).lower()
            ]

        if best_only and rows:
            rows = [rows[0]]

        self.filtered_rows = rows
        self._render_table()

    def _render_table(self):
        self.table_models.setRowCount(len(self.filtered_rows))

        for row_idx, row in enumerate(self.filtered_rows):
            values = [
                row.get("model_name", ""),
                row.get("graph_type", ""),
                row.get("spatial_type", ""),
                row.get("temporal_type", ""),
                f"{row.get('mae', 0.0):.4f}",
                f"{row.get('mape', 0.0):.4f}",
                f"{row.get('rmse', 0.0):.4f}",
                str(row.get("num_params", "")),
                row.get("time", ""),
            ]

            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))

                if col_idx in [4, 5, 6, 7]:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

                self.table_models.setItem(row_idx, col_idx, item)

        if self.filtered_rows:
            self.table_models.selectRow(0)
            self._update_detail_panel(self.filtered_rows[0], show_full_config=False)
        else:
            self.text_detail.setPlainText("当前筛选条件下没有匹配结果。")

    def _get_selected_filtered_row(self):
        row_idx = self.table_models.currentRow()
        if row_idx < 0 or row_idx >= len(self.filtered_rows):
            return None
        return self.filtered_rows[row_idx]

    def _on_table_selection_changed(self):
        row = self._get_selected_filtered_row()
        if row is not None:
            self._update_detail_panel(row, show_full_config=False)

    def _update_detail_panel(self, row, show_full_config=False):
        summary_lines = [
            f"model_name: {row.get('model_name', '')}",
            f"graph_type: {row.get('graph_type', '')}",
            f"spatial_type: {row.get('spatial_type', '')}",
            f"temporal_type: {row.get('temporal_type', '')}",
            f"loss_fn: {row.get('loss_fn', '')}",
            f"val_ratio: {row.get('val_ratio', '')}",
            f"early_stop_patience: {row.get('early_stop_patience', '')}",
            f"early_stop_min_delta: {row.get('early_stop_min_delta', '')}",
            f"correlation_topk: {row.get('correlation_topk', '')}",
            f"correlation_threshold: {row.get('correlation_threshold', '')}",
            f"use_abs_corr: {row.get('use_abs_corr', '')}",
            f"fusion_alpha: {row.get('fusion_alpha', '')}",
            f"predict_steps: {row.get('predict_steps', '')}",
            f"history_length: {row.get('history_length', '')}",
            f"batch_size: {row.get('batch_size', '')}",
            f"learning_rate: {row.get('learning_rate', '')}",
            f"epochs: {row.get('epochs', '')}",
            f"figure_horizon_step: {row.get('figure_horizon_step', '')}",
            f"num_params: {row.get('num_params', '')}",
            f"peak_gpu_mb: {row.get('peak_gpu_mb', '')}",
            f"mae: {row.get('mae', '')}",
            f"mape: {row.get('mape', '')}",
            f"rmse: {row.get('rmse', '')}",
            f"ckpt_path: {row.get('ckpt_path', '')}",
            f"fig_path: {row.get('fig_path', '')}",
            f"run_config_path: {row.get('run_config_path', '')}",
            f"time: {row.get('time', '')}",
        ]

        text = ["【实验摘要】", *summary_lines]

        if show_full_config:
            config_path = row.get("run_config_path", "")
            text.append("")
            text.append("【完整配置】")
            text.append(self._load_config_text(config_path))

        self.text_detail.setPlainText("\n".join(text))

    def _load_config_text(self, config_path: str):
        if not config_path:
            return "未找到 run_config_path。"

        if not os.path.exists(config_path):
            return f"配置文件不存在：{config_path}"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            return json.dumps(content, ensure_ascii=False, indent=2)
        except Exception:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"读取配置文件失败：{e}"

    def show_current_config(self):
        row = self._get_selected_filtered_row()
        if row is None:
            QMessageBox.warning(self, "提示", "请先选择一条实验记录。")
            return
        self._update_detail_panel(row, show_full_config=True)

    def export_current_table(self):
        if not self.filtered_rows:
            QMessageBox.information(self, "提示", "当前没有可导出的实验记录。")
            return

        default_name = "experiment_records.csv"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出实验表",
            default_name,
            "CSV Files (*.csv)"
        )

        if not save_path:
            return

        fieldnames = [
            "model_name", "graph_type", "spatial_type", "temporal_type",
            "loss_fn", "val_ratio", "early_stop_patience", "early_stop_min_delta",
            "correlation_topk", "correlation_threshold", "use_abs_corr", "fusion_alpha",
            "predict_steps", "history_length", "batch_size", "learning_rate", "epochs",
            "figure_horizon_step",
            "num_params", "peak_gpu_mb", "mae", "mape", "rmse",
            "ckpt_path", "fig_path", "run_config_path", "time"
        ]

        try:
            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.filtered_rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

            QMessageBox.information(self, "导出成功", f"已导出到：\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def reset_filters(self):
        self.combo_graph.setCurrentIndex(0)
        self.combo_spatial.setCurrentIndex(0)
        self.combo_temporal.setCurrentIndex(0)
        self.edit_keyword.clear()
        self.check_best_only.setChecked(False)
        self.apply_filters()

    def get_best_model_row(self):
        return self.model_rows[0] if self.model_rows else None

    def load_selected_model(self):
        row = self._get_selected_filtered_row()
        if row is None:
            QMessageBox.warning(self, "提示", "请先选择一个模型。")
            return

        self.current_model_row = row
        self.label_current_model.setText(
            f"当前模型：{row.get('model_name', '')} | RMSE={row.get('rmse', 0.0):.4f}"
        )

        if self.load_callback is not None:
            self.load_callback(row)

    def set_load_callback(self, callback):
        self.load_callback = callback
