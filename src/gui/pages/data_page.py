from pathlib import Path

from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.services.config_service import ConfigService
from src.services.data_service import DataService


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(320)

    def reset_axes(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)


class DataPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.config_service = ConfigService()
        self.data_service = DataService()
        self.preview = None

        self.imported_graph_path = ""
        self.imported_flow_path = ""

        self._init_ui()
        self.refresh_config_options()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        panel = QFrame()
        panel.setObjectName("PagePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("数据管理")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        desc = QLabel(
            "用于数据预览、数据导入导出与数据配置生成。"
        )
        desc.setStyleSheet("color: #6b7280; line-height: 1.6;")
        desc.setWordWrap(True)

        config_group = QGroupBox("数据预览配置")
        config_layout = QGridLayout(config_group)
        config_layout.setHorizontalSpacing(12)
        config_layout.setVerticalSpacing(10)

        self.combo_data_cfg = QComboBox()
        self.combo_data_cfg.setMaximumWidth(320)
        self.btn_refresh_cfg = QPushButton("刷新配置列表")
        self.btn_load_preview = QPushButton("加载数据预览")

        self.btn_refresh_cfg.clicked.connect(self.refresh_config_options)
        self.btn_load_preview.clicked.connect(self.load_preview)

        config_layout.addWidget(QLabel("数据配置:"), 0, 0)
        config_layout.addWidget(self.combo_data_cfg, 0, 1)
        config_layout.addWidget(self.btn_refresh_cfg, 0, 2)
        config_layout.addWidget(self.btn_load_preview, 0, 3)

        io_group = QGroupBox("数据导入与配置生成")
        io_layout = QGridLayout(io_group)
        io_layout.setHorizontalSpacing(10)
        io_layout.setVerticalSpacing(8)

        self.edit_graph_path = QLineEdit()
        self.edit_graph_path.setReadOnly(True)
        self.edit_graph_path.setPlaceholderText("未导入图结构文件")

        self.edit_flow_path = QLineEdit()
        self.edit_flow_path.setReadOnly(True)
        self.edit_flow_path.setPlaceholderText("未导入流量文件")

        self.btn_import_graph = QPushButton("导入图文件")
        self.btn_import_flow = QPushButton("导入流量文件")
        self.btn_import_graph.clicked.connect(self.import_graph_file)
        self.btn_import_flow.clicked.connect(self.import_flow_file)

        self.edit_dataset_name = QLineEdit("ImportedDataset")
        self.edit_data_cfg_name = QLineEdit("imported_dataset")
        self.edit_dataset_name.setMaximumWidth(220)
        self.edit_data_cfg_name.setMaximumWidth(220)

        self.spin_num_nodes = QSpinBox()
        self.spin_num_nodes.setRange(1, 100000)
        self.spin_num_nodes.setValue(307)
        self.spin_num_nodes.setMaximumWidth(160)

        self.spin_train_days = QSpinBox()
        self.spin_train_days.setRange(1, 3650)
        self.spin_train_days.setValue(45)
        self.spin_train_days.setMaximumWidth(160)

        self.spin_test_days = QSpinBox()
        self.spin_test_days.setRange(1, 3650)
        self.spin_test_days.setValue(14)
        self.spin_test_days.setMaximumWidth(160)

        self.spin_time_interval = QSpinBox()
        self.spin_time_interval.setRange(1, 1440)
        self.spin_time_interval.setValue(5)
        self.spin_time_interval.setMaximumWidth(160)

        self.combo_missing_strategy = QComboBox()
        self.combo_missing_strategy.addItems(["none", "linear_interpolate", "forward_fill", "mean_fill"])
        self.combo_missing_strategy.setMaximumWidth(220)

        self.spin_clip_min = QDoubleSpinBox()
        self.spin_clip_min.setDecimals(3)
        self.spin_clip_min.setRange(-1000000.0, 1000000.0)
        self.spin_clip_min.setValue(0.0)
        self.spin_clip_min.setMaximumWidth(160)

        self.spin_clip_quantile = QDoubleSpinBox()
        self.spin_clip_quantile.setDecimals(3)
        self.spin_clip_quantile.setRange(0.0, 0.999)
        self.spin_clip_quantile.setSingleStep(0.01)
        self.spin_clip_quantile.setValue(0.0)
        self.spin_clip_quantile.setToolTip("0 表示不启用高分位裁剪")
        self.spin_clip_quantile.setMaximumWidth(160)

        self.btn_create_data_cfg = QPushButton("生成数据配置")
        self.btn_create_data_cfg.clicked.connect(self.create_data_config)

        io_layout.addWidget(QLabel("图文件:"), 0, 0)
        io_layout.addWidget(self.edit_graph_path, 0, 1, 1, 3)
        io_layout.addWidget(self.btn_import_graph, 0, 4)

        io_layout.addWidget(QLabel("流量文件:"), 1, 0)
        io_layout.addWidget(self.edit_flow_path, 1, 1, 1, 3)
        io_layout.addWidget(self.btn_import_flow, 1, 4)

        io_layout.addWidget(QLabel("dataset_name:"), 2, 0)
        io_layout.addWidget(self.edit_dataset_name, 2, 1)
        io_layout.addWidget(QLabel("config_name:"), 2, 2)
        io_layout.addWidget(self.edit_data_cfg_name, 2, 3)

        io_layout.addWidget(QLabel("num_nodes:"), 3, 0)
        io_layout.addWidget(self.spin_num_nodes, 3, 1)
        io_layout.addWidget(QLabel("time_interval(min):"), 3, 2)
        io_layout.addWidget(self.spin_time_interval, 3, 3)

        io_layout.addWidget(QLabel("train_days:"), 4, 0)
        io_layout.addWidget(self.spin_train_days, 4, 1)
        io_layout.addWidget(QLabel("test_days:"), 4, 2)
        io_layout.addWidget(self.spin_test_days, 4, 3)

        io_layout.addWidget(QLabel("missing_strategy:"), 5, 0)
        io_layout.addWidget(self.combo_missing_strategy, 5, 1)
        io_layout.addWidget(QLabel("clip_min:"), 5, 2)
        io_layout.addWidget(self.spin_clip_min, 5, 3)

        io_layout.addWidget(QLabel("clip_max_quantile:"), 6, 0)
        io_layout.addWidget(self.spin_clip_quantile, 6, 1)

        io_layout.addWidget(self.btn_create_data_cfg, 3, 4, 4, 1)

        info_layout = QHBoxLayout()
        info_layout.setSpacing(16)

        summary_group = QGroupBox("数据集摘要")
        summary_box_layout = QVBoxLayout(summary_group)
        self.text_summary = QTextEdit()
        self.text_summary.setReadOnly(True)
        self.text_summary.setMinimumHeight(260)
        summary_box_layout.addWidget(self.text_summary)

        graph_group = QGroupBox("图结构统计")
        graph_box_layout = QVBoxLayout(graph_group)
        self.text_graph = QTextEdit()
        self.text_graph.setReadOnly(True)
        self.text_graph.setMinimumHeight(260)
        graph_box_layout.addWidget(self.text_graph)

        info_layout.addWidget(summary_group, 2)
        info_layout.addWidget(graph_group, 1)

        preview_ctrl_group = QGroupBox("节点曲线预览与导出")
        preview_ctrl_layout = QHBoxLayout(preview_ctrl_group)

        self.spin_node_id = QSpinBox()
        self.spin_node_id.setRange(0, 0)
        self.spin_node_id.setMaximumWidth(120)
        self.spin_node_id.valueChanged.connect(self.update_plot)

        self.spin_start_index = QSpinBox()
        self.spin_start_index.setRange(0, 0)
        self.spin_start_index.setMaximumWidth(140)
        self.spin_start_index.valueChanged.connect(self.update_plot)

        self.spin_points = QSpinBox()
        self.spin_points.setRange(50, 50000)
        self.spin_points.setValue(300)
        self.spin_points.setMaximumWidth(120)
        self.spin_points.valueChanged.connect(self.update_plot)

        self.spin_heatmap_nodes = QSpinBox()
        self.spin_heatmap_nodes.setRange(8, 256)
        self.spin_heatmap_nodes.setValue(32)
        self.spin_heatmap_nodes.setMaximumWidth(120)
        self.spin_heatmap_nodes.valueChanged.connect(self.update_plot)

        self.spin_heatmap_start_node = QSpinBox()
        self.spin_heatmap_start_node.setRange(0, 0)
        self.spin_heatmap_start_node.setMaximumWidth(140)
        self.spin_heatmap_start_node.valueChanged.connect(self.update_plot)

        self.btn_refresh_plot = QPushButton("刷新曲线")
        self.btn_refresh_plot.clicked.connect(self.update_plot)

        self.btn_export_summary = QPushButton("导出预览摘要")
        self.btn_export_summary.clicked.connect(self.export_preview_summary)

        self.btn_export_series = QPushButton("导出节点序列")
        self.btn_export_series.clicked.connect(self.export_node_series)

        preview_ctrl_layout.addWidget(QLabel("node_id:"))
        preview_ctrl_layout.addWidget(self.spin_node_id)
        preview_ctrl_layout.addSpacing(10)
        preview_ctrl_layout.addWidget(QLabel("起始点:"))
        preview_ctrl_layout.addWidget(self.spin_start_index)
        preview_ctrl_layout.addSpacing(10)
        preview_ctrl_layout.addWidget(QLabel("显示点数:"))
        preview_ctrl_layout.addWidget(self.spin_points)
        preview_ctrl_layout.addSpacing(10)
        preview_ctrl_layout.addWidget(QLabel("热力图开始节点:"))
        preview_ctrl_layout.addWidget(self.spin_heatmap_start_node)
        preview_ctrl_layout.addSpacing(10)
        preview_ctrl_layout.addWidget(QLabel("热力图节点数:"))
        preview_ctrl_layout.addWidget(self.spin_heatmap_nodes)
        preview_ctrl_layout.addSpacing(10)
        preview_ctrl_layout.addWidget(self.btn_refresh_plot)
        preview_ctrl_layout.addSpacing(10)
        preview_ctrl_layout.addWidget(self.btn_export_summary)
        preview_ctrl_layout.addWidget(self.btn_export_series)
        preview_ctrl_layout.addStretch()

        figure_row = QHBoxLayout()
        figure_row.setSpacing(16)

        figure_group = QGroupBox("节点流量曲线")
        figure_layout = QVBoxLayout(figure_group)
        self.canvas_series = MplCanvas(self, width=9, height=4, dpi=100)
        figure_layout.addWidget(self.canvas_series)

        heatmap_group = QGroupBox("时空热力图预览")
        heatmap_layout = QVBoxLayout(heatmap_group)
        self.canvas_heatmap = MplCanvas(self, width=9, height=4, dpi=100)
        heatmap_layout.addWidget(self.canvas_heatmap)

        figure_row.addWidget(figure_group, 1)
        figure_row.addWidget(heatmap_group, 1)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addWidget(config_group)
        layout.addWidget(io_group)
        layout.addLayout(info_layout)
        layout.addWidget(preview_ctrl_group)
        layout.addLayout(figure_row)

        root.addWidget(panel)

    def refresh_config_options(self):
        data_items = self.config_service.list_data_configs()
        self._fill_combo(self.combo_data_cfg, data_items)

    def _fill_combo(self, combo, items):
        combo.blockSignals(True)
        combo.clear()
        for item in items:
            combo.addItem(item["name"], item["path"])
        combo.blockSignals(False)

    def import_graph_file(self):
        src_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图结构文件",
            "",
            "Graph Files (*.csv *.txt);;All Files (*)",
        )
        if not src_path:
            return

        try:
            result = self.data_service.import_data_file(src_path=src_path, data_kind="graph")
            self.imported_graph_path = result["target_relative_path"]
            self.edit_graph_path.setText(self.imported_graph_path)
            QMessageBox.information(self, "导入成功", f"图文件已导入:\n{result['target_path']}")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    def import_flow_file(self):
        src_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择流量文件",
            "",
            "Flow Files (*.npz);;All Files (*)",
        )
        if not src_path:
            return

        try:
            result = self.data_service.import_data_file(src_path=src_path, data_kind="flow")
            self.imported_flow_path = result["target_relative_path"]
            self.edit_flow_path.setText(self.imported_flow_path)
            QMessageBox.information(self, "导入成功", f"流量文件已导入:\n{result['target_path']}")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    def create_data_config(self):
        graph_path = self.edit_graph_path.text().strip()
        flow_path = self.edit_flow_path.text().strip()

        if not graph_path or not flow_path:
            QMessageBox.warning(self, "提示", "请先导入图文件和流量文件。")
            return

        try:
            result = self.data_service.create_data_config(
                config_name=self.edit_data_cfg_name.text().strip(),
                dataset_name=self.edit_dataset_name.text().strip(),
                graph_path=graph_path,
                flow_path=flow_path,
                num_nodes=self.spin_num_nodes.value(),
                train_days=self.spin_train_days.value(),
                test_days=self.spin_test_days.value(),
                time_interval=self.spin_time_interval.value(),
                preprocess_cfg={
                    "missing_strategy": self.combo_missing_strategy.currentText(),
                    "clip_min": self.spin_clip_min.value(),
                    "clip_max_quantile": self.spin_clip_quantile.value() or None,
                },
            )
            self.refresh_config_options()

            cfg_file_name = Path(result["config_path"]).name
            idx = self.combo_data_cfg.findText(cfg_file_name)
            if idx >= 0:
                self.combo_data_cfg.setCurrentIndex(idx)

            QMessageBox.information(self, "生成成功", f"数据配置已生成:\n{result['config_path']}")
        except Exception as e:
            QMessageBox.critical(self, "生成失败", str(e))

    def load_preview(self):
        data_cfg_path = self.combo_data_cfg.currentData()

        if not data_cfg_path:
            QMessageBox.warning(self, "提示", "请先选择数据配置文件。")
            return

        try:
            self.preview = self.data_service.load_preview(
                data_cfg_path=data_cfg_path,
            )

            max_node = max(0, int(self.preview["num_nodes_actual"]) - 1)
            max_step = max(0, int(self.preview["total_steps"]) - 1)
            self.spin_node_id.setRange(0, max_node)
            self.spin_start_index.setRange(0, max_step)
            self.spin_heatmap_start_node.setRange(0, max_node)
            self.spin_heatmap_nodes.setRange(1, max(1, int(self.preview["num_nodes_actual"])))
            self.spin_start_index.setValue(0)
            self.spin_heatmap_start_node.setValue(0)

            self.edit_dataset_name.setText(str(self.preview.get("dataset_name", "ImportedDataset")))
            self.spin_num_nodes.setValue(int(self.preview.get("num_nodes_actual", 307)))
            self.spin_train_days.setValue(int(self.preview.get("train_days", 45)))
            self.spin_test_days.setValue(int(self.preview.get("test_days", 14)))
            self.spin_time_interval.setValue(int(self.preview.get("time_interval", 5)))
            preprocess_cfg = self.preview.get("preprocess_cfg", {})
            self.combo_missing_strategy.setCurrentText(str(preprocess_cfg.get("missing_strategy", "none")))
            self.spin_clip_min.setValue(float(preprocess_cfg.get("clip_min", 0.0) or 0.0))
            self.spin_clip_quantile.setValue(float(preprocess_cfg.get("clip_max_quantile") or 0.0))

            self._update_summary_text()
            self._update_graph_text()
            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "加载数据预览失败", str(e))

    def _update_summary_text(self):
        if self.preview is None:
            return

        p = self.preview
        lines = [
            f"dataset_name: {p['dataset_name']}",
            f"graph_path: {p['graph_path']}",
            f"flow_path: {p['flow_path']}",
            f"num_nodes(config): {p['num_nodes_cfg']}",
            f"num_nodes(actual): {p['num_nodes_actual']}",
            f"total_steps(T): {p['total_steps']}",
            f"input_dim(D): {p['input_dim']}",
            f"divide_days: {p['divide_days']}",
            f"time_interval(min): {p['time_interval']}",
            f"one_day_length: {p['one_day_length']}",
            f"train_days: {p['train_days']}",
            f"test_days: {p['test_days']}",
            f"train_steps: {p['train_steps']}",
            f"test_steps(config): {p['test_steps_cfg']}",
            f"test_steps(actual): {p['test_steps_actual']}",
            f"history_length: {p['history_length']}",
            f"predict_steps: {p['predict_steps']}",
            f"graph_type: {p['graph_type']}",
            f"train_samples: {p['train_samples']}",
            f"test_samples: {p['test_samples']}",
            f"preprocess_cfg: {p.get('preprocess_cfg', {})}",
        ]
        self.text_summary.setPlainText("\n".join(lines))

    def _update_graph_text(self):
        if self.preview is None:
            return

        p = self.preview
        lines = [
            f"adjacency_shape: {p['adjacency_shape']}",
            f"nonzero_edges: {p['nonzero_edges']}",
            f"density: {p['density']:.6f}",
            f"graph_cfg: {p['graph_cfg']}",
            f"preprocess_stats: {p.get('preprocess_stats', {})}",
        ]
        self.text_graph.setPlainText("\n".join(lines))

    def update_plot(self):
        if self.preview is None:
            self.canvas_series.ax.clear()
            self.canvas_series.ax.set_title("暂无数据")
            self.canvas_series.draw()
            self.canvas_heatmap.reset_axes()
            self.canvas_heatmap.ax.set_title("暂无数据")
            self.canvas_heatmap.draw()
            return

        node_id = self.spin_node_id.value()
        start_index = self.spin_start_index.value()
        max_points = self.spin_points.value()

        series = self.data_service.get_node_series(
            preview=self.preview,
            node_id=node_id,
            start_index=start_index,
            max_points=max_points,
        )
        x_values = list(range(start_index, start_index + len(series)))

        ax = self.canvas_series.ax
        ax.clear()
        ax.plot(x_values, series)
        ax.set_title(
            f"Node {node_id} Traffic Flow Preview ({start_index} - {start_index + max(0, len(series) - 1)})"
        )
        if x_values:
            ax.set_xlim(x_values[0], x_values[-1])
        ax.set_xlabel("Time Step Index")
        ax.set_ylabel("Traffic Flow")
        ax.grid(True, linestyle="--", alpha=0.3)
        self.canvas_series.figure.tight_layout()
        self.canvas_series.draw()
        self._update_heatmap(start_index=start_index, max_points=max_points)

    def _update_heatmap(self, start_index: int, max_points: int):
        if self.preview is None:
            return

        start_node = self.spin_heatmap_start_node.value()
        heatmap = self.data_service.get_flow_heatmap(
            preview=self.preview,
            start_index=start_index,
            start_node=start_node,
            max_nodes=self.spin_heatmap_nodes.value(),
            max_points=max_points,
        )

        self.canvas_heatmap.reset_axes()
        ax = self.canvas_heatmap.ax
        end_time = start_index + max(0, heatmap.shape[1] - 1)
        end_node = start_node + max(0, heatmap.shape[0] - 1)
        image = ax.imshow(
            heatmap,
            aspect="auto",
            origin="lower",
            cmap="YlOrRd",
            extent=(
                start_index - 0.5,
                end_time + 0.5,
                start_node - 0.5,
                end_node + 0.5,
            ),
        )
        ax.set_title(
            f"Traffic Flow Heatmap ({heatmap.shape[0]} nodes x {heatmap.shape[1]} steps, "
            f"time_start={start_index}, node_start={start_node})"
        )
        ax.set_xlabel("Time Step Index")
        ax.set_ylabel("Node ID")

        self.canvas_heatmap.figure.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        ).set_label("Traffic Flow")

        self.canvas_heatmap.figure.tight_layout()
        self.canvas_heatmap.draw()

    def export_preview_summary(self):
        if self.preview is None:
            QMessageBox.warning(self, "提示", "请先加载数据预览。")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出预览摘要",
            "data_preview_summary.json",
            "JSON Files (*.json)",
        )
        if not save_path:
            return

        try:
            self.data_service.export_preview_summary(self.preview, save_path)
            QMessageBox.information(self, "导出成功", f"已导出:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def export_node_series(self):
        if self.preview is None:
            QMessageBox.warning(self, "提示", "请先加载数据预览。")
            return

        node_id = self.spin_node_id.value()
        start_index = self.spin_start_index.value()
        max_points = self.spin_points.value()

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出节点序列",
            f"node_{node_id}_series.csv",
            "CSV Files (*.csv)",
        )
        if not save_path:
            return

        try:
            self.data_service.export_node_series(
                preview=self.preview,
                node_id=node_id,
                start_index=start_index,
                max_points=max_points,
                save_path=save_path,
            )
            QMessageBox.information(self, "导出成功", f"已导出:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
