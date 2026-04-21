import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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


class EventCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=3.4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(320)


class EventSimulationPage(QWidget):
    STRATEGY_LABELS = {
        "综合最优": RouteRecommendationService.STRATEGY_BALANCED,
        "距离最短": RouteRecommendationService.STRATEGY_DISTANCE,
        "优先避堵": RouteRecommendationService.STRATEGY_CONGESTION,
    }

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
        eyebrow = QLabel("Event Simulation")
        eyebrow.setObjectName("HeroEyebrow")
        title = QLabel("交通事件模拟")
        title.setObjectName("HeroTitle")
        subtitle = QLabel("模拟部分节点流量上升或下降后，观察推荐路线和风险指标如何变化。")
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
        hero_right_layout.addWidget(QLabel("模拟摘要"))
        hero_right_layout.addWidget(self.label_summary, 1)

        hero_layout.addLayout(hero_left, 3)
        hero_layout.addWidget(hero_right, 2)
        layout.addWidget(hero)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(16)

        setting_group = QGroupBox("模拟参数")
        setting_layout = QGridLayout(setting_group)
        setting_layout.setHorizontalSpacing(12)
        setting_layout.setVerticalSpacing(10)

        self.spin_sample_index = QSpinBox()
        self.spin_sample_index.setRange(0, 0)
        self.spin_sample_index.valueChanged.connect(self._on_controls_changed)

        self.spin_source = QSpinBox()
        self.spin_source.setRange(0, 0)
        self.spin_source.valueChanged.connect(self._on_controls_changed)

        self.spin_target = QSpinBox()
        self.spin_target.setRange(0, 0)
        self.spin_target.valueChanged.connect(self._on_controls_changed)

        self.spin_horizon_idx = QSpinBox()
        self.spin_horizon_idx.setRange(0, 0)
        self.spin_horizon_idx.valueChanged.connect(self._on_controls_changed)

        self.btn_last_horizon = QPushButton("选择最远预测步长")
        self.btn_last_horizon.clicked.connect(self._select_last_horizon)

        self.combo_strategy = QComboBox()
        self.combo_strategy.addItems(list(self.STRATEGY_LABELS.keys()))
        self.combo_strategy.currentIndexChanged.connect(self._on_controls_changed)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 5.0)
        self.spin_alpha.setSingleStep(0.1)
        self.spin_alpha.setValue(1.0)
        self.spin_alpha.valueChanged.connect(self._on_controls_changed)

        self.edit_event_nodes = QLineEdit("8,9,14,15")
        self.edit_event_nodes.editingFinished.connect(lambda: self.run_simulation(show_warnings=False))

        self.spin_multiplier = QDoubleSpinBox()
        self.spin_multiplier.setRange(0.0, 5.0)
        self.spin_multiplier.setDecimals(2)
        self.spin_multiplier.setSingleStep(0.1)
        self.spin_multiplier.setValue(1.6)
        self.spin_multiplier.valueChanged.connect(self._on_controls_changed)

        self.btn_run = QPushButton("运行事件模拟")
        self.btn_run.clicked.connect(self.run_simulation)

        setting_layout.addWidget(QLabel("样本索引:"), 0, 0)
        setting_layout.addWidget(self.spin_sample_index, 0, 1)
        setting_layout.addWidget(QLabel("起点节点:"), 1, 0)
        setting_layout.addWidget(self.spin_source, 1, 1)
        setting_layout.addWidget(QLabel("终点节点:"), 2, 0)
        setting_layout.addWidget(self.spin_target, 2, 1)
        setting_layout.addWidget(QLabel("预测步长:"), 3, 0)
        setting_layout.addWidget(self.spin_horizon_idx, 3, 1)
        setting_layout.addWidget(self.btn_last_horizon, 4, 0, 1, 2)
        setting_layout.addWidget(QLabel("策略:"), 5, 0)
        setting_layout.addWidget(self.combo_strategy, 5, 1)
        setting_layout.addWidget(QLabel("拥堵权重:"), 6, 0)
        setting_layout.addWidget(self.spin_alpha, 6, 1)
        setting_layout.addWidget(QLabel("事件节点:"), 7, 0)
        setting_layout.addWidget(self.edit_event_nodes, 7, 1)
        setting_layout.addWidget(QLabel("流量改变倍数:"), 8, 0)
        setting_layout.addWidget(self.spin_multiplier, 8, 1)
        setting_layout.addWidget(self.btn_run, 9, 0, 1, 2)

        metric_group = QGroupBox("事件影响指标")
        metric_layout = QGridLayout(metric_group)
        metric_layout.setHorizontalSpacing(18)
        metric_layout.setVerticalSpacing(10)

        self.label_original_route = self._make_metric_value("-")
        self.label_event_route = self._make_metric_value("-")
        self.label_distance_delta = self._make_metric_value("-")
        self.label_risk_delta = self._make_metric_value("-")

        metrics = [
            ("原始路线", self.label_original_route),
            ("事件后路线", self.label_event_route),
            ("距离变化", self.label_distance_delta),
            ("平均拥堵变化", self.label_risk_delta),
        ]
        for idx, (name, label) in enumerate(metrics):
            col = idx % 2
            row = (idx // 2) * 2
            metric_layout.addWidget(self._make_metric_title(name), row, col)
            metric_layout.addWidget(label, row + 1, col)

        top_layout.addWidget(setting_group, 1)
        top_layout.addWidget(metric_group, 2)
        layout.addLayout(top_layout)

        compare_group = QGroupBox("模拟前后路线对比")
        compare_layout = QVBoxLayout(compare_group)
        self.table_compare = QTableWidget()
        self.table_compare.setColumnCount(4)
        self.table_compare.setHorizontalHeaderLabels(["指标", "原始预测", "事件模拟后", "变化"])
        self._setup_table(self.table_compare)
        self.table_compare.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        compare_layout.addWidget(self.table_compare)
        layout.addWidget(compare_group)

        chart_group = QGroupBox("指标变化图（独立纵轴）")
        chart_layout = QVBoxLayout(chart_group)
        self.canvas_event = EventCanvas(self, width=10, height=3.4, dpi=100)
        chart_layout.addWidget(self.canvas_event)
        layout.addWidget(chart_group)

        text_group = QGroupBox("模拟说明")
        text_layout = QVBoxLayout(text_group)
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setMinimumHeight(180)
        text_layout.addWidget(self.text_result)
        layout.addWidget(text_group)

        root.addWidget(panel)

    @staticmethod
    def _setup_table(table: QTableWidget):
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setMinimumHeight(220)

    @staticmethod
    def _make_metric_title(text: str):
        label = QLabel(text)
        label.setStyleSheet("color: #64748b; font-size: 12px;")
        return label

    @staticmethod
    def _make_metric_value(text: str):
        label = QLabel(text)
        label.setStyleSheet("font-size: 18px; font-weight: bold; color: #111827;")
        label.setWordWrap(True)
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
            node_count = max(1, self.predictor.get_node_count())
            self._syncing_controls = True
            try:
                self.spin_sample_index.setMaximum(max(0, self.predictor.get_test_size() - 1))
                self.spin_source.setMaximum(node_count - 1)
                self.spin_target.setMaximum(node_count - 1)
                predict_steps = max(1, self.predictor.get_predict_steps())
                self.spin_horizon_idx.setMaximum(predict_steps - 1)
                self.btn_last_horizon.setText(f"选择最远预测步长 (h{predict_steps})")
                self.spin_source.setValue(0)
                self.spin_target.setValue(min(node_count - 1, 10))
                if self.spin_source.value() == self.spin_target.value() and node_count > 1:
                    self.spin_target.setValue(1)
            finally:
                self._syncing_controls = False
            self.run_simulation(show_warnings=False)
        except Exception as e:
            self.route_service = None
            self._reset_view()
            self.text_result.setPlainText(f"事件模拟初始化失败：{e}")

    def _reset_view(self):
        self.label_summary.setText("等待模型加载。")
        self.label_original_route.setText("-")
        self.label_event_route.setText("-")
        self.label_distance_delta.setText("-")
        self.label_risk_delta.setText("-")
        self.table_compare.setRowCount(0)
        self.text_result.setPlainText("请先在“模型管理”页面加载模型。")
        self.canvas_event.figure.clear()
        ax = self.canvas_event.figure.add_subplot(111)
        ax.set_title("暂无数据")
        ax.axis("off")
        self.canvas_event.draw()

    def _on_controls_changed(self, *args):
        if self._syncing_controls:
            return
        if self.predictor is not None and self.route_service is not None:
            self.run_simulation(show_warnings=False)

    def _select_last_horizon(self):
        max_horizon = self.spin_horizon_idx.maximum()
        if self.spin_horizon_idx.value() == max_horizon:
            self.run_simulation(show_warnings=False)
            return
        self.spin_horizon_idx.setValue(max_horizon)

    def _parse_event_nodes(self):
        text = self.edit_event_nodes.text().strip()
        if not text:
            return []
        nodes = []
        max_node = self.predictor.get_node_count() - 1 if self.predictor is not None else 0
        for part in text.replace("，", ",").split(","):
            part = part.strip()
            if not part:
                continue
            node = int(part)
            if node < 0 or node > max_node:
                raise ValueError(f"事件节点超出范围: {node}")
            nodes.append(node)
        return sorted(set(nodes))

    def run_simulation(self, show_warnings=True):
        if self.predictor is None or self.route_service is None:
            if show_warnings:
                QMessageBox.warning(self, "提示", "请先加载模型。")
            return

        try:
            sample_index = self.spin_sample_index.value()
            source = self.spin_source.value()
            target = self.spin_target.value()
            horizon_idx = self.spin_horizon_idx.value()
            strategy = self.STRATEGY_LABELS.get(self.combo_strategy.currentText(), RouteRecommendationService.STRATEGY_BALANCED)
            alpha = self.spin_alpha.value()
            multiplier = self.spin_multiplier.value()
            event_nodes = self._parse_event_nodes()

            sample = self.predictor.get_test_sample_detail(sample_index, source, horizon_idx=horizon_idx)
            prediction = np.asarray(sample["prediction_all_horizons"], dtype=float)
            event_prediction = prediction.copy()
            if event_nodes:
                event_prediction[event_nodes, horizon_idx] *= multiplier

            original = self.route_service.recommend_routes(
                prediction=prediction,
                source=source,
                target=target,
                horizon_idx=horizon_idx,
                strategy=strategy,
                alpha=alpha,
                topk=8,
                candidate_count=1,
            )
            simulated = self.route_service.recommend_routes(
                prediction=event_prediction,
                source=source,
                target=target,
                horizon_idx=horizon_idx,
                strategy=strategy,
                alpha=alpha,
                topk=8,
                candidate_count=1,
            )

            self._update_view(original, simulated, event_nodes, multiplier)
        except Exception as e:
            if show_warnings:
                QMessageBox.critical(self, "事件模拟失败", str(e))
            self.text_result.setPlainText(f"事件模拟失败：{e}")

    @staticmethod
    def _describe_multiplier(multiplier):
        if multiplier > 1.0:
            return f"放大为原来的 {multiplier:.2f} 倍"
        if multiplier < 1.0:
            return f"缩小为原来的 {multiplier:.2f} 倍"
        return "保持为原始预测流量"

    def _update_view(self, original, simulated, event_nodes, multiplier):
        if not original.get("reachable") or not simulated.get("reachable"):
            self.text_result.setPlainText("当前起终点不可达，无法进行路线对比。")
            return

        original_route = original["path_text"]
        simulated_route = simulated["path_text"]
        distance_delta = float(simulated["distance"] - original["distance"])
        risk_delta = float(simulated["avg_congestion_score"] - original["avg_congestion_score"])

        self.label_original_route.setText(original_route)
        self.label_event_route.setText(simulated_route)
        self.label_distance_delta.setText(f"{distance_delta:+.2f}")
        self.label_risk_delta.setText(f"{risk_delta:+.4f}")

        route_changed = original_route != simulated_route
        multiplier_text = self._describe_multiplier(multiplier)
        self.label_summary.setText(
            f"事件节点：{', '.join(str(n) for n in event_nodes) if event_nodes else '未设置'}\n"
            f"流量改变倍数：{multiplier:.2f}（{multiplier_text}）\n"
            f"路线{'发生变化' if route_changed else '保持不变'}。"
        )

        self._fill_compare_table(original, simulated)
        self._draw_chart(original, simulated)

        lines = [
            f"原始路线：{original_route}",
            f"事件后路线：{simulated_route}",
            f"事件节点：{', '.join(str(n) for n in event_nodes) if event_nodes else '未设置'}",
            f"事件设定：所选节点在当前预测步长的预测流量{multiplier_text}。",
            "",
            "解释：",
        ]
        if route_changed:
            lines.append("- 事件改变了候选路径代价，推荐路线发生调整。")
        else:
            lines.append("- 当前事件强度不足以改变最优路线，但风险指标可能已经变化。")
        lines.extend(f"- {item}" for item in RouteRecommendationService.explain_route(simulated))
        self.text_result.setPlainText("\n".join(lines))

    def _fill_compare_table(self, original, simulated):
        rows = [
            ("路线", original["path_text"], simulated["path_text"], "变化" if original["path_text"] != simulated["path_text"] else "不变"),
            ("距离", f"{original['distance']:.2f}", f"{simulated['distance']:.2f}", f"{simulated['distance'] - original['distance']:+.2f}"),
            ("平均拥堵", f"{original['avg_congestion_score']:.4f}", f"{simulated['avg_congestion_score']:.4f}", f"{simulated['avg_congestion_score'] - original['avg_congestion_score']:+.4f}"),
            ("最高拥堵", f"{original['max_congestion_score']:.4f}", f"{simulated['max_congestion_score']:.4f}", f"{simulated['max_congestion_score'] - original['max_congestion_score']:+.4f}"),
            ("高风险节点数", str(original["high_risk_node_count"]), str(simulated["high_risk_node_count"]), f"{simulated['high_risk_node_count'] - original['high_risk_node_count']:+d}"),
        ]
        self.table_compare.setRowCount(len(rows))
        for row_idx, values in enumerate(rows):
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.table_compare.setItem(row_idx, col_idx, item)

    def _draw_chart(self, original, simulated):
        fig = self.canvas_event.figure
        fig.clear()
        axes = fig.subplots(
            1,
            3,
            gridspec_kw={"width_ratios": [1.0, 1.35, 1.0], "wspace": 0.35},
        )
        colors = ["#64748b", "#dc2626"]

        distance_values = [float(original["distance"]), float(simulated["distance"])]
        axes[0].bar(["原始", "事件后"], distance_values, color=colors, width=0.58)
        axes[0].set_title("路线距离")
        axes[0].set_ylabel("距离")
        axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)

        congestion_labels = ["平均拥堵", "最高拥堵"]
        x = np.arange(len(congestion_labels))
        width = 0.34
        original_congestion = [
            float(original["avg_congestion_score"]),
            float(original["max_congestion_score"]),
        ]
        simulated_congestion = [
            float(simulated["avg_congestion_score"]),
            float(simulated["max_congestion_score"]),
        ]
        axes[1].bar(x - width / 2, original_congestion, width=width, color=colors[0], label="原始")
        axes[1].bar(x + width / 2, simulated_congestion, width=width, color=colors[1], label="事件后")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(congestion_labels)
        axes[1].set_title("拥堵指数")
        axes[1].set_ylabel("指数")
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)
        axes[1].legend(loc="upper right")

        risk_values = [float(original["high_risk_node_count"]), float(simulated["high_risk_node_count"])]
        axes[2].bar(["原始", "事件后"], risk_values, color=colors, width=0.58)
        axes[2].set_title("高风险节点数")
        axes[2].set_ylabel("节点数")
        axes[2].grid(True, axis="y", linestyle="--", alpha=0.3)

        for ax in axes:
            ax.margins(y=0.18)
        fig.suptitle("事件前后指标对比：距离、拥堵、节点数分别使用独立纵轴", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        self.canvas_event.draw()
