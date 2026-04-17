from collections import Counter

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
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
        self.setMinimumHeight(300)


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
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("HeroPanel")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        hero_layout.setSpacing(20)

        hero_left = QVBoxLayout()
        hero_left.setSpacing(10)

        eyebrow = QLabel("System Overview")
        eyebrow.setObjectName("HeroEyebrow")

        title = QLabel("交通流量预测与可视化分析平台")
        title.setObjectName("HeroTitle")
        title.setWordWrap(True)

        subtitle = QLabel(
            "提供数据接入、模型训练、推理分析和结果管理的一体化工作流，"
            "用于交通流量预测任务的实验运行与可视化分析。"
        )
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(10)
        badge_row.addWidget(self._make_badge("GCN / ChebNet / GAT"))
        badge_row.addWidget(self._make_badge("多种建图策略"))
        badge_row.addWidget(self._make_badge("训练-推理-分析一体化"))
        badge_row.addStretch()

        hero_left.addWidget(eyebrow)
        hero_left.addWidget(title)
        hero_left.addWidget(subtitle)
        hero_left.addLayout(badge_row)

        hero_right = QFrame()
        hero_right.setObjectName("HeroSummary")
        hero_right.setMinimumWidth(300)
        hero_right_layout = QVBoxLayout(hero_right)
        hero_right_layout.setContentsMargins(18, 18, 18, 18)
        hero_right_layout.setSpacing(8)

        hero_summary_title = QLabel("当前概况")
        hero_summary_title.setObjectName("HeroSummaryTitle")

        self.label_hero_summary = QLabel("等待实验记录载入。")
        self.label_hero_summary.setObjectName("HeroSummaryText")
        self.label_hero_summary.setWordWrap(True)
        self.label_hero_summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        hero_right_layout.addWidget(hero_summary_title)
        hero_right_layout.addWidget(self.label_hero_summary, 1)

        hero_layout.addLayout(hero_left, 3)
        hero_layout.addWidget(hero_right, 2)
        panel_layout.addWidget(hero)

        card_layout = QGridLayout()
        card_layout.setSpacing(14)

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

        chart_group = QGroupBox("实验概览")
        chart_layout = QVBoxLayout(chart_group)
        self.canvas_overview = MplCanvas(self, width=10, height=3.4, dpi=100)
        chart_layout.addWidget(self.canvas_overview)
        panel_layout.addWidget(chart_group)

        middle_layout = QGridLayout()
        middle_layout.setHorizontalSpacing(16)
        middle_layout.setVerticalSpacing(16)

        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)
        self.text_status = QTextEdit()
        self.text_status.setReadOnly(True)
        self.text_status.setMinimumHeight(200)
        status_layout.addWidget(self.text_status)

        recent_group = QGroupBox("最近实验记录")
        recent_layout = QVBoxLayout(recent_group)
        self.table_recent = QTableWidget()
        self.table_recent.setColumnCount(6)
        self.table_recent.setHorizontalHeaderLabels(
            ["模型名称", "图类型", "空间模块", "时间模块", "RMSE", "实验时间"]
        )
        self.table_recent.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_recent.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_recent.setSelectionMode(QTableWidget.SingleSelection)
        self.table_recent.setAlternatingRowColors(True)
        self.table_recent.setWordWrap(False)
        self.table_recent.verticalHeader().setVisible(False)
        self.table_recent.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_recent.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_recent.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_recent.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_recent.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_recent.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table_recent.setMinimumHeight(240)
        recent_layout.addWidget(self.table_recent)

        middle_layout.addWidget(status_group, 0, 0)
        middle_layout.addWidget(recent_group, 0, 1)
        panel_layout.addLayout(middle_layout)

        bottom_layout = QGridLayout()
        bottom_layout.setHorizontalSpacing(16)
        bottom_layout.setVerticalSpacing(16)

        alignment_group = QGroupBox("系统能力概览")
        alignment_layout = QVBoxLayout(alignment_group)
        self.text_alignment = QTextEdit()
        self.text_alignment.setReadOnly(True)
        self.text_alignment.setMinimumHeight(220)
        alignment_layout.addWidget(self.text_alignment)

        planning_group = QGroupBox("推荐使用流程")
        planning_layout = QVBoxLayout(planning_group)
        self.text_next_steps = QTextEdit()
        self.text_next_steps.setReadOnly(True)
        self.text_next_steps.setMinimumHeight(220)
        self.text_next_steps.setPlainText(
            "1. 在“数据管理”页加载数据集，确认节点规模、样本数量和图结构统计信息。\n"
            "2. 在“实验训练”页选择默认配置或覆盖关键参数，启动训练任务并观察日志输出。\n"
            "3. 在“模型管理”页筛选实验记录，加载需要分析的模型版本。\n"
            "4. 在“在线推理”页查看单样本预测结果、误差分布和批量评估导出。\n"
            "5. 在“结果分析”页对比分组指标、基线汇总、多模型结果和分步长表现。"
        )
        planning_layout.addWidget(self.text_next_steps)

        bottom_layout.addWidget(alignment_group, 0, 0)
        bottom_layout.addWidget(planning_group, 0, 1)
        panel_layout.addLayout(bottom_layout)

        root.addWidget(panel)

    @staticmethod
    def _make_badge(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("HeroBadge")
        label.setAlignment(Qt.AlignCenter)
        return label

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

        spatial_counter = Counter(
            str(row.get("spatial_type", "")) for row in rows if row.get("spatial_type", "")
        )
        graph_counter = Counter(
            str(row.get("graph_type", "")) for row in rows if row.get("graph_type", "")
        )

        self.card_total_runs.set_value(str(total_runs))
        self.card_spatial_count.set_value(str(len(spatial_counter)))
        self.card_graph_count.set_value(str(len(graph_counter)))

        hero_lines = [
            f"系统状态：{status_text.replace('状态：', '') if status_text else '未知'}",
            f"累计实验记录：{total_runs} 条",
        ]
        if best_row:
            hero_lines.append(f"当前最佳：{best_row.get('model_name', '-')}")
            hero_lines.append(
                f"最佳结果：RMSE {best_row.get('rmse', 0.0):.4f} / MAE {best_row.get('mae', 0.0):.4f}"
            )
        else:
            hero_lines.append("当前还没有实验记录，建议先跑通一组默认配置。")
        self.label_hero_summary.setText("\n".join(hero_lines))

        status_lines = [
            status_text or "状态：未知",
            current_model_text or "当前模型：未加载",
            device_text or "设备：-",
            "",
        ]
        if best_row:
            status_lines.extend(
                [
                    "当前最佳实验：",
                    f"模型名称：{best_row.get('model_name', '-')}",
                    f"图类型：{best_row.get('graph_type', '-')}",
                    f"空间模块：{best_row.get('spatial_type', '-')}",
                    f"时间模块：{best_row.get('temporal_type', '-')}",
                    f"预测步数：{best_row.get('predict_steps', '-')}",
                    f"RMSE：{best_row.get('rmse', 0.0):.4f}",
                    f"MAE：{best_row.get('mae', 0.0):.4f}",
                    f"实验时间：{best_row.get('time', '-')}",
                ]
            )
        else:
            status_lines.append("当前暂无实验记录。")
        self.text_status.setPlainText("\n".join(status_lines))

        alignment_lines = [
            "1. 数据层：支持数据导入、配置生成、缺失值处理、异常值裁剪、节点曲线预览和时空热力图预览。",
            "2. 模型层：已实现 GCN / ChebNet / GAT 空间模块，并在统一框架下支持 GRU 时间建模。",
            "3. 图结构层：支持 connect、distance、correlation、distance_correlation 四种建图方式。",
            "4. 功能层：已形成数据管理、实验训练、模型管理、在线推理、结果分析五个核心页面。",
        ]
        if best_row:
            alignment_lines.append(
                f"5. 结果层：当前已有 {total_runs} 条实验记录，最佳模型为 {best_row.get('model_name', '-')}"
                f"（RMSE {best_row.get('rmse', 0.0):.4f}）。"
            )
        else:
            alignment_lines.append("5. 结果层：当前尚无实验记录，可先运行一组默认配置以生成基础分析结果。")

        if spatial_counter:
            alignment_lines.append(
                f"6. 当前已覆盖的空间模块：{', '.join(sorted(spatial_counter.keys()))}。"
            )
        if graph_counter:
            alignment_lines.append(
                f"7. 当前已覆盖的图构建方式：{', '.join(sorted(graph_counter.keys()))}。"
            )
        alignment_lines.append("8. 系统支持继续扩展新的时间模块、图构建策略和可视化视图。")
        self.text_alignment.setPlainText("\n".join(alignment_lines))

        self._update_recent_table(rows[:5])
        self._update_overview_chart(rows)

    def _update_recent_table(self, recent_rows):
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
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.table_recent.setItem(row_idx, col_idx, item)

    def _update_overview_chart(self, rows):
        ax_left = self.canvas_overview.ax_left
        ax_right = self.canvas_overview.ax_right
        fig = self.canvas_overview.figure

        ax_left.clear()
        ax_right.clear()
        fig.patch.set_facecolor("#ffffff")

        for ax in (ax_left, ax_right):
            ax.set_facecolor("#f8fafc")
            for spine in ax.spines.values():
                spine.set_color("#d9e2ef")

        if not rows:
            ax_left.set_title("Top Runs RMSE")
            ax_left.text(0.5, 0.5, "No Data", ha="center", va="center", color="#64748b")
            ax_left.set_xticks([])
            ax_left.set_yticks([])

            ax_right.set_title("Graph Type Distribution")
            ax_right.text(0.5, 0.5, "No Data", ha="center", va="center", color="#64748b")
            ax_right.set_xticks([])
            ax_right.set_yticks([])

            fig.tight_layout()
            self.canvas_overview.draw()
            return

        top_rows = rows[: min(10, len(rows))]
        rmse_values = []
        for row in top_rows:
            try:
                rmse_values.append(float(row.get("rmse", 0.0)))
            except Exception:
                rmse_values.append(0.0)

        x_values = list(range(1, len(rmse_values) + 1))
        ax_left.plot(x_values, rmse_values, marker="o", linewidth=2.2, color="#0f766e")
        ax_left.fill_between(x_values, rmse_values, color="#99f6e4", alpha=0.22)
        ax_left.set_title("Top Runs RMSE")
        ax_left.set_xlabel("Rank")
        ax_left.set_ylabel("RMSE")
        ax_left.grid(True, linestyle="--", alpha=0.28, color="#94a3b8")

        graph_counter = Counter(str(row.get("graph_type", "unknown")) for row in rows)
        graph_items = graph_counter.most_common(8)
        names = [name for name, _ in graph_items]
        counts = [count for _, count in graph_items]
        positions = list(range(len(names)))
        ax_right.bar(positions, counts, color="#f59e0b", edgecolor="#d97706", linewidth=1.0)
        ax_right.set_title("Graph Type Distribution")
        ax_right.set_xlabel("Graph Type")
        ax_right.set_ylabel("Count")
        ax_right.set_xticks(positions)
        ax_right.set_xticklabels(names, rotation=18, ha="right")
        ax_right.grid(True, axis="y", linestyle="--", alpha=0.28, color="#94a3b8")

        fig.tight_layout()
        self.canvas_overview.draw()
