from collections import Counter

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.gui.widgets.metric_card import MetricCard


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=3.2, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.ax_left = self.figure.add_subplot(121)
        self.ax_right = self.figure.add_subplot(122)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(280)


class HomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        panel = QFrame()
        panel.setObjectName("PagePanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(16)

        # ---------------- 标题区 ----------------
        title = QLabel("交通流量预测系统")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")

        subtitle = QLabel(
            "基于图神经网络的交通流量预测实验与分析平台\n"
            "支持数据预览、模型训练、实验管理、在线推理与结果对比分析。"
        )
        subtitle.setStyleSheet("color: #6b7280; line-height: 1.6;")
        subtitle.setWordWrap(True)

        panel_layout.addWidget(title)
        panel_layout.addWidget(subtitle)

        # ---------------- 指标卡片 ----------------
        card_layout = QGridLayout()
        card_layout.setSpacing(16)

        self.card_total_runs = MetricCard("实验总数", "0")
        self.card_best_model = MetricCard("当前最佳模型", "-")
        self.card_best_rmse = MetricCard("最佳 RMSE", "-")
        self.card_best_mae = MetricCard("最佳 MAE", "-")
        self.card_spatial_count = MetricCard("空间模块种类", "0")
        self.card_graph_count = MetricCard("建图方式种类", "0")

        card_layout.addWidget(self.card_total_runs, 0, 0)
        card_layout.addWidget(self.card_best_model, 0, 1)
        card_layout.addWidget(self.card_best_rmse, 0, 2)

        card_layout.addWidget(self.card_best_mae, 1, 0)
        card_layout.addWidget(self.card_spatial_count, 1, 1)
        card_layout.addWidget(self.card_graph_count, 1, 2)

        panel_layout.addLayout(card_layout)

        chart_group = QGroupBox("实验概览图")
        chart_layout = QVBoxLayout(chart_group)
        self.canvas_overview = MplCanvas(self, width=10, height=3.2, dpi=100)
        chart_layout.addWidget(self.canvas_overview)
        panel_layout.addWidget(chart_group)

        # ---------------- 中间区域：系统状态 + 最近实验 ----------------
        middle_layout = QGridLayout()
        middle_layout.setHorizontalSpacing(16)
        middle_layout.setVerticalSpacing(16)

        # 系统状态
        status_group = QGroupBox("当前系统状态")
        status_layout = QVBoxLayout(status_group)

        self.text_status = QTextEdit()
        self.text_status.setReadOnly(True)
        self.text_status.setMinimumHeight(180)
        status_layout.addWidget(self.text_status)

        # 最近实验
        recent_group = QGroupBox("最近实验记录（按当前排序显示前 5 项）")
        recent_layout = QVBoxLayout(recent_group)

        self.table_recent = QTableWidget()
        self.table_recent.setColumnCount(6)
        self.table_recent.setHorizontalHeaderLabels([
            "模型名", "图类型", "空间模块", "时间模块", "RMSE", "实验时间"
        ])
        self.table_recent.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_recent.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_recent.setSelectionMode(QTableWidget.SingleSelection)
        self.table_recent.setAlternatingRowColors(True)
        self.table_recent.setWordWrap(False)
        self.table_recent.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_recent.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_recent.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_recent.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_recent.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_recent.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table_recent.setMinimumHeight(220)
        recent_layout.addWidget(self.table_recent)

        middle_layout.addWidget(status_group, 0, 0)
        middle_layout.addWidget(recent_group, 0, 1)

        panel_layout.addLayout(middle_layout)

        # ---------------- 底部区域：系统说明 ----------------
        bottom_layout = QGridLayout()
        bottom_layout.setHorizontalSpacing(16)
        bottom_layout.setVerticalSpacing(16)

        module_group = QGroupBox("系统模块概览")
        module_layout = QVBoxLayout(module_group)

        self.text_modules = QTextEdit()
        self.text_modules.setReadOnly(True)
        self.text_modules.setMinimumHeight(180)
        self.text_modules.setPlainText(
            "1. 数据管理：展示数据集信息、样本规模、图结构统计和节点曲线预览。\n"
            "2. 实验训练：基于配置模板和页面参数覆盖启动训练任务。\n"
            "3. 模型管理：筛选、检索、导出实验记录并加载指定模型。\n"
            "4. 在线推理：对测试样本执行推理，分析单节点与全节点误差。\n"
            "5. 结果分析：展示当前模型指标，并进行多实验对比分析。"
        )
        module_layout.addWidget(self.text_modules)

        workflow_group = QGroupBox("推荐使用流程")
        workflow_layout = QVBoxLayout(workflow_group)

        self.text_workflow = QTextEdit()
        self.text_workflow.setReadOnly(True)
        self.text_workflow.setMinimumHeight(180)
        self.text_workflow.setPlainText(
            "① 在“数据管理”页确认数据集信息、图结构和节点曲线。\n"
            "② 在“实验训练”页选择默认配置，并按需修改参数启动训练。\n"
            "③ 在“模型管理”页筛选与加载实验结果。\n"
            "④ 在“在线推理”页分析样本级预测效果与误差分布。\n"
            "⑤ 在“结果分析”页查看指标、曲线与多实验对比结果。"
        )
        workflow_layout.addWidget(self.text_workflow)

        bottom_layout.addWidget(module_group, 0, 0)
        bottom_layout.addWidget(workflow_group, 0, 1)

        panel_layout.addLayout(bottom_layout)

        root.addWidget(panel)

    def update_summary(
        self,
        best_row,
        total_runs: int,
        rows=None,
        current_model_text: str = "",
        device_text: str = "",
        status_text: str = "",
    ):
        rows = rows or []

        if not best_row:
            self.card_best_model.set_value("-")
            self.card_best_rmse.set_value("-")
            self.card_best_mae.set_value("-")
        else:
            self.card_best_model.set_value(best_row.get("model_name", "-"))
            self.card_best_rmse.set_value(f"{best_row.get('rmse', 0.0):.4f}")
            self.card_best_mae.set_value(f"{best_row.get('mae', 0.0):.4f}")

        self.card_total_runs.set_value(str(total_runs))

        spatial_types = {str(r.get("spatial_type", "")) for r in rows if r.get("spatial_type", "")}
        graph_types = {str(r.get("graph_type", "")) for r in rows if r.get("graph_type", "")}
        self.card_spatial_count.set_value(str(len(spatial_types)))
        self.card_graph_count.set_value(str(len(graph_types)))
        self._update_overview_chart(rows)

        # 状态文本
        status_lines = [
            f"系统状态：{status_text or '未知'}",
            f"{current_model_text or '当前模型：未加载'}",
            f"{device_text or '设备：-'}",
            "",
        ]

        if best_row:
            status_lines.extend([
                "当前最佳实验：",
                f"模型名: {best_row.get('model_name', '-')}",
                f"图类型: {best_row.get('graph_type', '-')}",
                f"空间模块: {best_row.get('spatial_type', '-')}",
                f"时间模块: {best_row.get('temporal_type', '-')}",
                f"RMSE: {best_row.get('rmse', 0.0):.4f}",
                f"MAE: {best_row.get('mae', 0.0):.4f}",
                f"实验时间: {best_row.get('time', '-')}",
            ])
        else:
            status_lines.append("当前暂无实验记录。")

        self.text_status.setPlainText("\n".join(status_lines))

        # 最近实验表
        recent_rows = rows[:5]
        self.table_recent.setRowCount(len(recent_rows))

        for row_idx, row in enumerate(recent_rows):
            values = [
                row.get("model_name", ""),
                row.get("graph_type", ""),
                row.get("spatial_type", ""),
                row.get("temporal_type", ""),
                f"{row.get('rmse', 0.0):.4f}",
                row.get("time", ""),
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.table_recent.setItem(row_idx, col_idx, item)

    def _update_overview_chart(self, rows):
        ax_l = self.canvas_overview.ax_left
        ax_r = self.canvas_overview.ax_right
        ax_l.clear()
        ax_r.clear()

        if not rows:
            ax_l.set_title("RMSE Trend")
            ax_l.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax_l.set_xticks([])
            ax_l.set_yticks([])

            ax_r.set_title("Graph Type Distribution")
            ax_r.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax_r.set_xticks([])
            ax_r.set_yticks([])

            self.canvas_overview.figure.tight_layout()
            self.canvas_overview.draw()
            return

        top_rows = rows[:10]
        rmse_values = []
        for row in top_rows:
            try:
                rmse_values.append(float(row.get("rmse", 0.0)))
            except Exception:
                rmse_values.append(0.0)

        x = list(range(1, len(rmse_values) + 1))
        ax_l.plot(x, rmse_values, marker="o", linewidth=1.8)
        ax_l.set_title("Top Runs RMSE")
        ax_l.set_xlabel("Rank")
        ax_l.set_ylabel("RMSE")
        ax_l.grid(True, linestyle="--", alpha=0.3)

        graph_counter = Counter(str(r.get("graph_type", "unknown")) for r in rows)
        graph_items = graph_counter.most_common(8)
        names = [k for k, _ in graph_items]
        counts = [v for _, v in graph_items]
        x2 = list(range(len(names)))
        ax_r.bar(x2, counts)
        ax_r.set_title("Graph Type Distribution")
        ax_r.set_xlabel("Graph Type")
        ax_r.set_ylabel("Count")
        ax_r.set_xticks(x2)
        ax_r.set_xticklabels(names, rotation=20, ha="right")
        ax_r.grid(True, axis="y", linestyle="--", alpha=0.3)

        self.canvas_overview.figure.tight_layout()
        self.canvas_overview.draw()
