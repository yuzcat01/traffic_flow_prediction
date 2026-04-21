import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.services.route_service import RouteRecommendationService


rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False


class WarningCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=3.4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(320)


class CongestionWarningPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = None
        self.route_service = None
        self.current_model_row = None
        self._syncing_controls = False
        self._init_ui()
        self._reset_view()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        panel = QFrame()
        panel.setObjectName("PagePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("HeroPanel")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        hero_layout.setSpacing(20)

        hero_left = QVBoxLayout()
        eyebrow = QLabel("Congestion Warning")
        eyebrow.setObjectName("HeroEyebrow")
        title = QLabel("未来拥堵预警")
        title.setObjectName("HeroTitle")
        subtitle = QLabel("面向全网节点识别未来高风险位置，辅助交通管理者提前发现可能拥堵的路段。")
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        hero_left.addWidget(eyebrow)
        hero_left.addWidget(title)
        hero_left.addWidget(subtitle)

        hero_right = QFrame()
        hero_right.setObjectName("HeroSummary")
        hero_right.setMinimumWidth(330)
        hero_right_layout = QVBoxLayout(hero_right)
        hero_right_layout.setContentsMargins(18, 18, 18, 18)
        self.label_summary = QLabel("等待模型加载。")
        self.label_summary.setObjectName("HeroSummaryText")
        self.label_summary.setWordWrap(True)
        self.label_summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        hero_right_layout.addWidget(QLabel("预警摘要"))
        hero_right_layout.addWidget(self.label_summary, 1)

        hero_layout.addLayout(hero_left, 3)
        hero_layout.addWidget(hero_right, 2)
        layout.addWidget(hero)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(16)

        setting_group = QGroupBox("预警参数")
        setting_layout = QGridLayout(setting_group)
        setting_layout.setHorizontalSpacing(12)
        setting_layout.setVerticalSpacing(10)

        self.spin_sample_index = QSpinBox()
        self.spin_sample_index.setRange(0, 0)
        self.spin_sample_index.valueChanged.connect(self._on_controls_changed)

        self.spin_horizon_idx = QSpinBox()
        self.spin_horizon_idx.setRange(0, 0)
        self.spin_horizon_idx.valueChanged.connect(self._on_controls_changed)

        self.btn_last_horizon = QPushButton("选择最远预测步长")
        self.btn_last_horizon.clicked.connect(self._select_last_horizon)

        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(3, 50)
        self.spin_topk.setValue(10)
        self.spin_topk.valueChanged.connect(self._on_controls_changed)

        self.btn_refresh = QPushButton("刷新预警")
        self.btn_refresh.clicked.connect(self.run_warning)

        setting_layout.addWidget(QLabel("样本索引:"), 0, 0)
        setting_layout.addWidget(self.spin_sample_index, 0, 1)
        setting_layout.addWidget(QLabel("预测步长索引:"), 1, 0)
        setting_layout.addWidget(self.spin_horizon_idx, 1, 1)
        setting_layout.addWidget(self.btn_last_horizon, 2, 0, 1, 2)
        setting_layout.addWidget(QLabel("风险 Top-K:"), 3, 0)
        setting_layout.addWidget(self.spin_topk, 3, 1)
        setting_layout.addWidget(self.btn_refresh, 4, 0, 1, 2)

        metric_group = QGroupBox("预警指标")
        metric_layout = QGridLayout(metric_group)
        metric_layout.setHorizontalSpacing(18)
        metric_layout.setVerticalSpacing(10)

        self.label_horizon = self._make_metric_value("-")
        self.label_max_score = self._make_metric_value("-")
        self.label_avg_topk = self._make_metric_value("-")
        self.label_severe_count = self._make_metric_value("-")

        metrics = [
            ("预测时间", self.label_horizon),
            ("最高拥堵指数", self.label_max_score),
            ("Top-K 平均指数", self.label_avg_topk),
            ("严重风险节点数", self.label_severe_count),
        ]
        for idx, (name, label) in enumerate(metrics):
            col = idx % 4
            metric_layout.addWidget(self._make_metric_title(name), 0, col)
            metric_layout.addWidget(label, 1, col)

        top_layout.addWidget(setting_group, 1)
        top_layout.addWidget(metric_group, 3)
        layout.addLayout(top_layout)

        table_group = QGroupBox("高风险节点 Top-K")
        table_layout = QVBoxLayout(table_group)
        self.table_risk = QTableWidget()
        self.table_risk.setColumnCount(6)
        self.table_risk.setHorizontalHeaderLabels(["排名", "节点", "预测流量", "参考流量", "拥堵指数", "风险等级"])
        self._setup_table(self.table_risk)
        table_layout.addWidget(self.table_risk)
        layout.addWidget(table_group)

        chart_group = QGroupBox("拥堵指数分布")
        chart_layout = QVBoxLayout(chart_group)
        self.canvas_warning = WarningCanvas(self, width=10, height=3.4, dpi=100)
        chart_layout.addWidget(self.canvas_warning)
        layout.addWidget(chart_group)

        text_group = QGroupBox("预警说明")
        text_layout = QVBoxLayout(text_group)
        self.text_warning = QTextEdit()
        self.text_warning.setReadOnly(True)
        self.text_warning.setMinimumHeight(150)
        text_layout.addWidget(self.text_warning)
        layout.addWidget(text_group)

        root.addWidget(panel)

    @staticmethod
    def _setup_table(table: QTableWidget):
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.setMinimumHeight(250)

    @staticmethod
    def _make_metric_title(text: str):
        label = QLabel(text)
        label.setStyleSheet("color: #64748b; font-size: 12px;")
        return label

    @staticmethod
    def _make_metric_value(text: str):
        label = QLabel(text)
        label.setStyleSheet("font-size: 22px; font-weight: bold; color: #111827;")
        return label

    def set_predictor(self, predictor, current_model_row=None):
        self.predictor = predictor
        self.current_model_row = current_model_row
        if self.predictor is None:
            self.route_service = None
            self._reset_view()
            return

        try:
            self.route_service = RouteRecommendationService(
                graph_path=self.predictor.graph_path,
                flow_path=self.predictor.flow_path,
                num_nodes=self.predictor.get_node_count(),
                preprocess_cfg=getattr(self.predictor, "preprocess_cfg", {}),
            )
            self._syncing_controls = True
            try:
                self.spin_sample_index.setMaximum(max(0, self.predictor.get_test_size() - 1))
                predict_steps = max(1, self.predictor.get_predict_steps())
                self.spin_horizon_idx.setMaximum(predict_steps - 1)
                self.btn_last_horizon.setText(f"选择最远预测步长 (h{predict_steps})")
            finally:
                self._syncing_controls = False
            self.run_warning(show_warnings=False)
        except Exception as e:
            self.route_service = None
            self._reset_view()
            self.text_warning.setPlainText(f"拥堵预警初始化失败：{e}")

    def _reset_view(self):
        self.label_summary.setText("等待模型加载。")
        self.label_horizon.setText("-")
        self.label_max_score.setText("-")
        self.label_avg_topk.setText("-")
        self.label_severe_count.setText("-")
        self.table_risk.setRowCount(0)
        self.text_warning.setPlainText("请先在“模型管理”页面加载模型。")
        self.canvas_warning.ax.clear()
        self.canvas_warning.ax.set_title("暂无数据")
        self.canvas_warning.draw()

    def _on_controls_changed(self):
        if self._syncing_controls:
            return
        if self.predictor is not None and self.route_service is not None:
            self.run_warning(show_warnings=False)

    def _select_last_horizon(self):
        max_horizon = self.spin_horizon_idx.maximum()
        if self.spin_horizon_idx.value() == max_horizon:
            self.run_warning(show_warnings=False)
            return
        self.spin_horizon_idx.setValue(max_horizon)

    def _format_horizon_text(self, horizon_idx: int) -> str:
        steps = int(horizon_idx) + 1
        interval = int(self.predictor.dataset_cfg.get("time_interval", 1))
        minutes = steps * max(1, interval)
        if minutes >= 60 and minutes % 60 == 0:
            time_text = f"未来 {minutes // 60} 小时"
        elif minutes >= 60:
            time_text = f"未来 {minutes / 60.0:.1f} 小时"
        else:
            time_text = f"未来 {minutes} 分钟"
        return f"h{steps} / {time_text}"

    @staticmethod
    def _risk_color(score: float) -> str:
        if score < 0.55:
            return "#0f766e"
        if score < 0.75:
            return "#ca8a04"
        if score < 1.0:
            return "#ea580c"
        return "#dc2626"

    def run_warning(self, show_warnings=True):
        if self.predictor is None or self.route_service is None:
            if show_warnings:
                QMessageBox.warning(self, "提示", "请先加载模型。")
            return

        try:
            sample_index = self.spin_sample_index.value()
            horizon_idx = self.spin_horizon_idx.value()
            topk = self.spin_topk.value()
            sample = self.predictor.get_test_sample_detail(sample_index, 0, horizon_idx=horizon_idx)
            prediction = np.asarray(sample["prediction_all_horizons"], dtype=float)
            risks = self.route_service.top_risk_nodes(prediction, horizon_idx=horizon_idx, topk=topk)
            _, scores = self.route_service.compute_congestion_scores(prediction, horizon_idx=horizon_idx)

            severe_count = int(np.count_nonzero(scores >= 1.0))
            max_score = float(np.max(scores)) if scores.size else 0.0
            avg_topk = float(np.mean([row["congestion_score"] for row in risks])) if risks else 0.0
            horizon_text = self._format_horizon_text(horizon_idx)

            self.label_horizon.setText(horizon_text)
            self.label_max_score.setText(f"{max_score:.3f}")
            self.label_avg_topk.setText(f"{avg_topk:.3f}")
            self.label_severe_count.setText(str(severe_count))

            self._fill_table(risks)
            self._draw_chart(risks, horizon_text)
            self._update_text(risks, severe_count, max_score, horizon_text)
        except Exception as e:
            if show_warnings:
                QMessageBox.critical(self, "拥堵预警失败", str(e))
            self.text_warning.setPlainText(f"拥堵预警失败：{e}")

    def _fill_table(self, rows):
        self.table_risk.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            values = [
                row["rank"],
                row["node_id"],
                f"{row['predicted_flow']:.4f}",
                f"{row['baseline_flow']:.4f}",
                f"{row['congestion_score']:.4f}",
                row["risk_level"],
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.table_risk.setItem(row_idx, col_idx, item)

    def _draw_chart(self, rows, horizon_text):
        ax = self.canvas_warning.ax
        ax.clear()
        if not rows:
            ax.set_title("暂无预警数据")
            self.canvas_warning.draw()
            return

        labels = [str(row["node_id"]) for row in rows]
        scores = [float(row["congestion_score"]) for row in rows]
        colors = [self._risk_color(score) for score in scores]
        x = np.arange(len(labels))
        ax.bar(x, scores, color=colors)
        ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.2, label="严重风险阈值")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"高风险节点拥堵指数（{horizon_text}）")
        ax.set_xlabel("节点")
        ax.set_ylabel("拥堵指数")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend()
        self.canvas_warning.figure.tight_layout()
        self.canvas_warning.draw()

    def _update_text(self, risks, severe_count, max_score, horizon_text):
        if not risks:
            self.label_summary.setText("暂无风险节点。")
            self.text_warning.setPlainText("当前预测步长下未生成风险节点列表。")
            return

        top = risks[0]
        if severe_count > 0:
            level_text = f"发现 {severe_count} 个严重风险节点，建议优先关注。"
        elif max_score >= 0.75:
            level_text = "未达到严重风险，但存在中度拥堵压力。"
        else:
            level_text = "整体风险较低，主要用于持续监测。"

        self.label_summary.setText(
            f"{horizon_text}\n"
            f"最高风险节点：{top['node_id']}，拥堵指数 {top['congestion_score']:.3f}\n"
            f"{level_text}"
        )
        lines = [
            f"预警时间：{horizon_text}",
            f"最高风险节点：{top['node_id']}，风险等级：{top['risk_level']}，拥堵指数：{top['congestion_score']:.3f}",
            f"严重风险节点数：{severe_count}",
            "",
            "处理建议：",
            "- 优先核查 Top-K 节点附近的道路状态和传感器数据。",
            "- 若严重风险节点位于主干路径，可结合“路线规划”页面生成避堵路线。",
            "- 参考流量为节点历史 85% 分位数，用作风险评估基准。",
        ]
        self.text_warning.setPlainText("\n".join(lines))
