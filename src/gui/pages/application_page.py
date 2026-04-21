import csv

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
from matplotlib.lines import Line2D

from src.services.route_service import RouteRecommendationService


rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False


class RouteStatsCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=3.2, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.ax_flow = self.figure.add_subplot(121)
        self.ax_risk = self.figure.add_subplot(122)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(300)


class NetworkCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=4.4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(420)


class ApplicationPage(QWidget):
    STRATEGY_LABELS = {
        "综合最优": RouteRecommendationService.STRATEGY_BALANCED,
        "距离最短": RouteRecommendationService.STRATEGY_DISTANCE,
        "优先避堵": RouteRecommendationService.STRATEGY_CONGESTION,
    }
    ROUTE_COLORS = ["#0f766e", "#2563eb", "#f59e0b", "#9333ea", "#e11d48", "#0891b2"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = None
        self.route_service = None
        self.current_model_row = None
        self.current_recommendation = None
        self.current_selected_route = None
        self.current_prediction = None
        self.network_preview = None
        self._syncing_controls = False
        self._syncing_candidate_selection = False

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
        hero_left.setSpacing(10)
        eyebrow = QLabel("Route Planning")
        eyebrow.setObjectName("HeroEyebrow")
        title = QLabel("预测驱动的路线规划")
        title.setObjectName("HeroTitle")
        title.setWordWrap(True)
        subtitle = QLabel(
            "基于当前加载模型的未来流量预测，将节点级预测结果转化为拥堵风险，"
            "并结合路网连接关系生成多条可选路线，支持可达性查询和路线高亮。"
        )
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(10)
        for text in ["起终点可达性", "多候选路线", "图中路线高亮"]:
            badge = QLabel(text)
            badge.setObjectName("HeroBadge")
            badge.setAlignment(Qt.AlignCenter)
            badge_row.addWidget(badge)
        badge_row.addStretch()

        hero_left.addWidget(eyebrow)
        hero_left.addWidget(title)
        hero_left.addWidget(subtitle)
        hero_left.addLayout(badge_row)

        hero_right = QFrame()
        hero_right.setObjectName("HeroSummary")
        hero_right.setMinimumWidth(330)
        hero_right_layout = QVBoxLayout(hero_right)
        hero_right_layout.setContentsMargins(18, 18, 18, 18)
        hero_right_layout.setSpacing(8)
        hero_right_title = QLabel("当前路线摘要")
        hero_right_title.setObjectName("HeroSummaryTitle")
        self.label_summary = QLabel("等待模型加载。")
        self.label_summary.setObjectName("HeroSummaryText")
        self.label_summary.setWordWrap(True)
        self.label_summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        hero_right_layout.addWidget(hero_right_title)
        hero_right_layout.addWidget(self.label_summary, 1)

        hero_layout.addLayout(hero_left, 3)
        hero_layout.addWidget(hero_right, 2)
        layout.addWidget(hero)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(16)

        setting_group = QGroupBox("应用参数")
        setting_layout = QFormLayout(setting_group)

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

        self.spin_candidate_count = QSpinBox()
        self.spin_candidate_count.setRange(1, 6)
        self.spin_candidate_count.setValue(3)
        self.spin_candidate_count.valueChanged.connect(self._on_controls_changed)

        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(3, 30)
        self.spin_topk.setValue(10)
        self.spin_topk.valueChanged.connect(self._on_controls_changed)

        self.btn_recommend = QPushButton("生成路线建议")
        self.btn_recommend.clicked.connect(self.run_recommendation)

        self.btn_reachability = QPushButton("查询起终点可达性")
        self.btn_reachability.clicked.connect(self.run_reachability_query)

        self.btn_export = QPushButton("导出应用报告")
        self.btn_export.clicked.connect(self.export_report)

        setting_layout.addRow("样本索引:", self.spin_sample_index)
        setting_layout.addRow("起点节点:", self.spin_source)
        setting_layout.addRow("终点节点:", self.spin_target)
        setting_layout.addRow("预测步长索引:", self.spin_horizon_idx)
        setting_layout.addRow("", self.btn_last_horizon)
        setting_layout.addRow("推荐策略:", self.combo_strategy)
        setting_layout.addRow("拥堵权重:", self.spin_alpha)
        setting_layout.addRow("候选路线数:", self.spin_candidate_count)
        setting_layout.addRow("风险 Top-K:", self.spin_topk)
        setting_layout.addRow("", self.btn_reachability)
        setting_layout.addRow("", self.btn_recommend)
        setting_layout.addRow("", self.btn_export)

        metric_group = QGroupBox("当前路线指标 / 可达性")
        metric_layout = QGridLayout(metric_group)
        metric_layout.setHorizontalSpacing(18)
        metric_layout.setVerticalSpacing(10)

        self.label_distance = self._make_metric_value("-")
        self.label_avg_risk = self._make_metric_value("-")
        self.label_max_risk = self._make_metric_value("-")
        self.label_high_risk_count = self._make_metric_value("-")
        self.label_node_count = self._make_metric_value("-")
        self.label_delta = self._make_metric_value("-")
        self.label_future_time = self._make_metric_value("-")
        self.label_reachability_badge = QLabel("未查询")
        self.label_reachability_badge.setAlignment(Qt.AlignCenter)
        self.label_reachability_badge.setStyleSheet(
            "background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; "
            "border-radius: 10px; padding: 7px 10px; font-weight: 800;"
        )

        metrics = [
            ("路线距离", self.label_distance),
            ("平均拥堵", self.label_avg_risk),
            ("最高拥堵", self.label_max_risk),
            ("高风险节点", self.label_high_risk_count),
            ("经过节点", self.label_node_count),
            ("距最短路差值", self.label_delta),
            ("预测时间", self.label_future_time),
        ]
        for idx, (name, label) in enumerate(metrics):
            row = (idx // 3) * 2
            col = idx % 3
            metric_layout.addWidget(self._make_metric_title(name), row, col)
            metric_layout.addWidget(label, row + 1, col)

        self.text_reachability = QTextEdit()
        self.text_reachability.setReadOnly(True)
        self.text_reachability.setMinimumHeight(88)
        self.text_reachability.setMaximumHeight(120)
        metric_layout.addWidget(self._make_metric_title("起终点可达性"), 6, 0)
        metric_layout.addWidget(self.label_reachability_badge, 6, 1, 1, 2)
        metric_layout.addWidget(self.text_reachability, 7, 0, 1, 3)

        top_layout.addWidget(setting_group, 1)
        top_layout.addWidget(metric_group, 2)
        layout.addLayout(top_layout)

        route_compare_layout = QHBoxLayout()
        route_compare_layout.setSpacing(16)

        candidate_group = QGroupBox("候选路线")
        candidate_layout = QVBoxLayout(candidate_group)
        self.table_candidates = QTableWidget()
        self.table_candidates.setColumnCount(7)
        self.table_candidates.setHorizontalHeaderLabels(
            ["方案", "距离", "平均拥堵", "最高拥堵", "高风险节点", "节点数", "路线"]
        )
        self._setup_table(self.table_candidates)
        self.table_candidates.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.table_candidates.itemSelectionChanged.connect(self._on_candidate_selection_changed)
        candidate_layout.addWidget(self.table_candidates)

        compare_group = QGroupBox("最短路线 vs 当前路线")
        compare_layout = QVBoxLayout(compare_group)
        self.table_compare = QTableWidget()
        self.table_compare.setColumnCount(4)
        self.table_compare.setHorizontalHeaderLabels(["指标", "最短路线", "当前路线", "变化"])
        self._setup_table(self.table_compare)
        self.table_compare.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_compare.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_compare.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table_compare.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table_compare.setMinimumHeight(170)
        compare_layout.addWidget(self.table_compare)

        route_compare_layout.addWidget(candidate_group, 3)
        route_compare_layout.addWidget(compare_group, 2)
        layout.addLayout(route_compare_layout)

        network_group = QGroupBox("路网连接关系预览")
        network_layout = QVBoxLayout(network_group)
        self.canvas_network = NetworkCanvas(self, width=10, height=4.6, dpi=100)
        self.canvas_network.mpl_connect("button_press_event", self._on_network_clicked)
        network_layout.addWidget(self.canvas_network)
        layout.addWidget(network_group)

        route_group = QGroupBox("路线决策说明")
        route_layout = QVBoxLayout(route_group)
        self.text_route = QTextEdit()
        self.text_route.setReadOnly(True)
        self.text_route.setMinimumHeight(160)
        route_layout.addWidget(self.text_route)
        layout.addWidget(route_group)

        table_layout = QHBoxLayout()
        table_layout.setSpacing(16)

        route_nodes_group = QGroupBox("路线节点风险")
        route_nodes_layout = QVBoxLayout(route_nodes_group)
        self.table_route_nodes = QTableWidget()
        self.table_route_nodes.setColumnCount(6)
        self.table_route_nodes.setHorizontalHeaderLabels(
            ["序号", "节点", "预测流量", "参考流量", "拥堵指数", "风险等级"]
        )
        self._setup_table(self.table_route_nodes)
        route_nodes_layout.addWidget(self.table_route_nodes)

        risk_group = QGroupBox("全网拥堵风险 Top-K")
        risk_layout = QVBoxLayout(risk_group)
        self.table_top_risk = QTableWidget()
        self.table_top_risk.setColumnCount(6)
        self.table_top_risk.setHorizontalHeaderLabels(
            ["排名", "节点", "预测流量", "参考流量", "拥堵指数", "风险等级"]
        )
        self._setup_table(self.table_top_risk)
        risk_layout.addWidget(self.table_top_risk)

        table_layout.addWidget(route_nodes_group, 1)
        table_layout.addWidget(risk_group, 1)
        layout.addLayout(table_layout)

        chart_group = QGroupBox("当前路线预测流量与拥堵指数")
        chart_layout = QVBoxLayout(chart_group)
        self.canvas_route = RouteStatsCanvas(self, width=10, height=3.4, dpi=100)
        chart_layout.addWidget(self.canvas_route)
        layout.addWidget(chart_group)

        root.addWidget(panel)

    @staticmethod
    def _setup_table(table: QTableWidget):
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setMinimumHeight(180)

    @staticmethod
    def _make_metric_title(text: str):
        label = QLabel(text)
        label.setStyleSheet("color: #64748b; font-size: 12px;")
        return label

    @staticmethod
    def _make_metric_value(text: str):
        label = QLabel(text)
        label.setStyleSheet("font-size: 20px; font-weight: bold; color: #111827;")
        return label

    def set_predictor(self, predictor, current_model_row=None):
        self.predictor = predictor
        self.current_model_row = current_model_row
        self.current_recommendation = None
        self.current_selected_route = None
        self.current_prediction = None

        if self.predictor is None:
            self.route_service = None
            self.network_preview = None
            self._reset_view()
            return

        try:
            self.route_service = RouteRecommendationService(
                graph_path=self.predictor.graph_path,
                flow_path=self.predictor.flow_path,
                num_nodes=self.predictor.get_node_count(),
                preprocess_cfg=getattr(self.predictor, "preprocess_cfg", {}),
            )
            self.network_preview = self.route_service.get_network_preview()
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
            self.run_recommendation(show_warnings=False)
        except Exception as e:
            self.route_service = None
            self.network_preview = None
            self._reset_view()
            self.text_route.setPlainText(f"路线规划模块初始化失败：{e}")

    def _reset_view(self):
        self.label_summary.setText("等待模型加载。")
        for label in [
            self.label_distance,
            self.label_avg_risk,
            self.label_max_risk,
            self.label_high_risk_count,
            self.label_node_count,
            self.label_delta,
            self.label_future_time,
        ]:
            label.setText("-")
        self.label_reachability_badge.setText("未查询")
        self.label_reachability_badge.setStyleSheet(
            "background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; "
            "border-radius: 10px; padding: 7px 10px; font-weight: 800;"
        )
        self.btn_last_horizon.setText("选择最远预测步长")
        self.text_route.setPlainText("请先在“模型管理”页面加载模型，然后在本页生成候选路线和路网高亮。")
        self.text_reachability.setPlainText("加载模型后可查询当前起点和终点在路网连接关系中是否可达。")
        self.table_candidates.setRowCount(0)
        self.table_compare.setRowCount(0)
        self.table_route_nodes.setRowCount(0)
        self.table_top_risk.setRowCount(0)
        self._draw_network()
        self.canvas_route.ax_flow.clear()
        self.canvas_route.ax_flow.set_title("暂无数据")
        self.canvas_route.ax_risk.clear()
        self.canvas_route.ax_risk.set_title("暂无数据")
        self.canvas_route.draw()

    def _on_controls_changed(self):
        if self._syncing_controls:
            return
        if self.predictor is None or self.route_service is None:
            return
        self.run_recommendation(show_warnings=False)

    def _select_last_horizon(self):
        max_horizon = self.spin_horizon_idx.maximum()
        if self.spin_horizon_idx.value() == max_horizon:
            self.run_recommendation(show_warnings=False)
            return
        self.spin_horizon_idx.setValue(max_horizon)

    def _format_horizon_text(self, horizon_idx: int) -> str:
        steps = int(horizon_idx) + 1
        interval = 1
        try:
            interval = int(self.predictor.dataset_cfg.get("time_interval", 1))
        except Exception:
            interval = 1

        minutes = steps * max(1, interval)
        if minutes >= 60 and minutes % 60 == 0:
            time_text = f"未来 {minutes // 60} 小时"
        elif minutes >= 60:
            time_text = f"未来 {minutes / 60.0:.1f} 小时"
        else:
            time_text = f"未来 {minutes} 分钟"
        return f"h{steps} / {time_text}"

    def _on_candidate_selection_changed(self):
        if self._syncing_candidate_selection:
            return
        if not self.current_recommendation:
            return
        selected_items = self.table_candidates.selectedItems()
        if not selected_items:
            return
        row_idx = selected_items[0].row()
        candidates = self.current_recommendation.get("candidates", [])
        if row_idx < 0 or row_idx >= len(candidates):
            return
        self._set_selected_candidate(row_idx)

    def _on_network_clicked(self, event):
        if event.inaxes is not self.canvas_network.ax:
            return
        if event.xdata is None or event.ydata is None or not self.network_preview:
            return

        nodes = self.network_preview.get("nodes", [])
        if not nodes:
            return
        positions = np.array([[node["x"], node["y"]] for node in nodes], dtype=float)
        point = np.array([event.xdata, event.ydata], dtype=float)
        distances = np.sqrt(np.sum((positions - point) ** 2, axis=1))
        node_idx = int(np.argmin(distances))
        if distances[node_idx] > 0.08:
            return

        node_id = int(nodes[node_idx]["node_id"])
        self._syncing_controls = True
        try:
            if event.button == 3:
                self.spin_target.setValue(node_id)
            else:
                self.spin_source.setValue(node_id)
                if self.spin_target.value() == node_id and self.spin_target.maximum() > 0:
                    self.spin_target.setValue((node_id + 1) % (self.spin_target.maximum() + 1))
        finally:
            self._syncing_controls = False
        self.run_recommendation(show_warnings=False)

    def run_reachability_query(self):
        if self.route_service is None:
            QMessageBox.warning(self, "提示", "请先加载模型或数据。")
            return

        source = self.spin_source.value()
        target = self.spin_target.value()
        try:
            result = self.route_service.query_reachability(source, target)
            self._update_reachability_view(result)
        except Exception as e:
            QMessageBox.critical(self, "可达性查询失败", str(e))

    def _update_reachability_view(self, result):
        if result.get("reachable"):
            self.label_reachability_badge.setText("可达")
            self.label_reachability_badge.setStyleSheet(
                "background: #ecfdf5; color: #047857; border: 1px solid #86efac; "
                "border-radius: 10px; padding: 7px 10px; font-weight: 800;"
            )
            hop_count = result.get("hop_count", 0)
            path_text = result.get("path_text", "")
            self.text_reachability.setPlainText(
                f"查询结果：可达\n"
                f"起点 -> 终点：{result['source']} -> {result['target']}\n"
                f"最少连接跳数：{hop_count}\n"
                f"连接路径：{path_text}"
            )
        else:
            self.label_reachability_badge.setText("不可达")
            self.label_reachability_badge.setStyleSheet(
                "background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; "
                "border-radius: 10px; padding: 7px 10px; font-weight: 800;"
            )
            self.text_reachability.setPlainText(
                f"查询结果：不可达\n"
                f"起点 -> 终点：{result['source']} -> {result['target']}\n"
                f"说明：当前图结构中不存在连接路径。"
            )

    def run_recommendation(self, show_warnings=True):
        if self.predictor is None or self.route_service is None:
            if show_warnings:
                QMessageBox.warning(self, "提示", "请先在模型管理页面加载模型。")
            return

        sample_index = self.spin_sample_index.value()
        source = self.spin_source.value()
        target = self.spin_target.value()
        horizon_idx = self.spin_horizon_idx.value()
        strategy_label = self.combo_strategy.currentText()
        strategy = self.STRATEGY_LABELS.get(strategy_label, RouteRecommendationService.STRATEGY_BALANCED)
        alpha = self.spin_alpha.value()
        topk = self.spin_topk.value()
        candidate_count = self.spin_candidate_count.value()

        try:
            if source == target:
                raise ValueError("起点和终点不能相同。")

            sample = self.predictor.get_test_sample_detail(sample_index, source, horizon_idx=horizon_idx)
            prediction = np.asarray(sample["prediction_all_horizons"], dtype=float)
            recommendation = self.route_service.recommend_routes(
                prediction=prediction,
                source=source,
                target=target,
                horizon_idx=horizon_idx,
                strategy=strategy,
                alpha=alpha,
                topk=topk,
                candidate_count=candidate_count,
            )
            self.current_prediction = prediction
            self.current_recommendation = recommendation
            self._update_reachability_view(self.route_service.query_reachability(source, target))
            self._fill_candidate_table(recommendation.get("candidates", []))
            self._fill_top_risk_table(recommendation.get("top_risk_nodes", []))
            if recommendation.get("candidates"):
                self._select_candidate_row(0)
                self._set_selected_candidate(0)
            else:
                self.current_selected_route = None
                self._update_selected_route_view(recommendation)
        except Exception as e:
            if show_warnings:
                QMessageBox.critical(self, "路线推荐失败", str(e))
            self.text_route.setPlainText(f"路线推荐失败：{e}")

    def _select_candidate_row(self, row_idx: int):
        self._syncing_candidate_selection = True
        try:
            self.table_candidates.selectRow(row_idx)
        finally:
            self._syncing_candidate_selection = False

    def _set_selected_candidate(self, row_idx: int):
        candidates = self.current_recommendation.get("candidates", []) if self.current_recommendation else []
        if row_idx < 0 or row_idx >= len(candidates):
            return
        selected = dict(candidates[row_idx])
        selected["candidates"] = candidates
        selected["selected_index"] = row_idx
        selected["top_risk_nodes"] = self.current_recommendation.get("top_risk_nodes", [])
        self.current_selected_route = selected
        self._select_candidate_row(row_idx)
        self._update_selected_route_view(selected)

    def _update_selected_route_view(self, recommendation):
        if not recommendation.get("reachable", False):
            self.text_route.setPlainText(recommendation.get("message", "未找到可达路径。"))
            self.label_summary.setText("当前起终点不可达。")
            self.table_route_nodes.setRowCount(0)
            self._draw_network()
            return

        path_text = recommendation["path_text"]
        shortest_path_text = recommendation.get("shortest_path_text", "")
        avg_risk = float(recommendation["avg_congestion_score"])
        max_risk = float(recommendation["max_congestion_score"])
        distance = float(recommendation["distance"])
        high_risk_count = int(recommendation["high_risk_node_count"])
        distance_delta = float(recommendation["distance_delta"])
        congestion_delta = float(recommendation["congestion_delta"])

        self.label_distance.setText(f"{distance:.2f}")
        self.label_avg_risk.setText(f"{avg_risk:.3f}")
        self.label_max_risk.setText(f"{max_risk:.3f}")
        self.label_high_risk_count.setText(str(high_risk_count))
        self.label_node_count.setText(str(recommendation["node_count"]))
        self.label_delta.setText(f"{distance_delta:+.2f}")
        horizon_text = self._format_horizon_text(recommendation["horizon_idx"])
        self.label_future_time.setText(horizon_text)

        if high_risk_count > 0:
            decision = "建议避开或延后通过高风险节点。"
        elif avg_risk >= 0.75:
            decision = "路线整体存在中等拥堵压力，可作为备选方案。"
        else:
            decision = "路线整体风险较低，适合作为当前推荐方案。"

        lines = [
            f"当前方案：候选路线 {recommendation.get('route_rank', 1)}",
            f"推荐路线：{path_text}",
            f"最短距离路线：{shortest_path_text}",
            f"预测步长：{horizon_text}，策略：{self.combo_strategy.currentText()}，拥堵权重：{recommendation['alpha']:.2f}",
            f"路线距离：{distance:.2f}，平均拥堵指数：{avg_risk:.3f}，最高拥堵指数：{max_risk:.3f}",
            f"相对最短路：距离差 {distance_delta:+.2f}，平均拥堵差 {congestion_delta:+.3f}",
            "参考流量说明：图中的参考流量来自该节点历史流量的 85% 分位数，用作拥堵指数的基准，并不代表真实道路容量。",
            f"决策建议：{decision}",
            "",
            "推荐原因：",
        ]
        lines.extend(f"- {item}" for item in RouteRecommendationService.explain_route(recommendation))
        self.text_route.setPlainText("\n".join(lines))
        self.label_summary.setText(
            f"样本 {self.spin_sample_index.value()}，{recommendation['source']} -> {recommendation['target']}\n"
            f"{horizon_text}，当前选中候选路线 {recommendation.get('route_rank', 1)}。\n"
            f"平均拥堵 {avg_risk:.3f}，最高拥堵 {max_risk:.3f}。"
        )

        self._fill_comparison_table(recommendation)
        self._fill_route_table(recommendation.get("route_nodes", []))
        self._draw_route_charts(recommendation.get("route_nodes", []))
        self._draw_network(selected_route=recommendation)

    def _fill_candidate_table(self, rows):
        self.table_candidates.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            values = [
                row.get("route_rank", row_idx + 1),
                f"{row['distance']:.2f}",
                f"{row['avg_congestion_score']:.4f}",
                f"{row['max_congestion_score']:.4f}",
                row["high_risk_node_count"],
                row["node_count"],
                row["path_text"],
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter if col_idx == 6 else Qt.AlignCenter)
                self.table_candidates.setItem(row_idx, col_idx, item)

    def _fill_comparison_table(self, recommendation):
        rows = [
            (
                "距离",
                f"{recommendation.get('shortest_distance', 0.0):.2f}",
                f"{recommendation.get('distance', 0.0):.2f}",
                f"{recommendation.get('distance_delta', 0.0):+.2f}",
            ),
            (
                "平均拥堵",
                f"{recommendation.get('shortest_avg_congestion_score', 0.0):.4f}",
                f"{recommendation.get('avg_congestion_score', 0.0):.4f}",
                f"{recommendation.get('congestion_delta', 0.0):+.4f}",
            ),
            (
                "最高拥堵",
                f"{recommendation.get('shortest_max_congestion_score', 0.0):.4f}",
                f"{recommendation.get('max_congestion_score', 0.0):.4f}",
                f"{recommendation.get('max_congestion_score', 0.0) - recommendation.get('shortest_max_congestion_score', 0.0):+.4f}",
            ),
            (
                "高风险节点数",
                str(recommendation.get("shortest_high_risk_node_count", 0)),
                str(recommendation.get("high_risk_node_count", 0)),
                f"{recommendation.get('high_risk_node_count', 0) - recommendation.get('shortest_high_risk_node_count', 0):+d}",
            ),
            (
                "经过节点数",
                str(recommendation.get("shortest_node_count", 0)),
                str(recommendation.get("node_count", 0)),
                f"{recommendation.get('node_count', 0) - recommendation.get('shortest_node_count', 0):+d}",
            ),
        ]

        self.table_compare.setRowCount(len(rows))
        for row_idx, values in enumerate(rows):
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.table_compare.setItem(row_idx, col_idx, item)

    def _fill_route_table(self, rows):
        self.table_route_nodes.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            values = [
                row["order"],
                row["node_id"],
                f"{row['predicted_flow']:.4f}",
                f"{row['baseline_flow']:.4f}",
                f"{row['congestion_score']:.4f}",
                row["risk_level"],
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.table_route_nodes.setItem(row_idx, col_idx, item)

    def _fill_top_risk_table(self, rows):
        self.table_top_risk.setRowCount(len(rows))
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
                self.table_top_risk.setItem(row_idx, col_idx, item)

    @staticmethod
    def _risk_color(score: float) -> str:
        if score < 0.55:
            return "#0f766e"
        if score < 0.75:
            return "#ca8a04"
        if score < 1.0:
            return "#ea580c"
        return "#dc2626"

    def _draw_network(self, selected_route=None):
        ax = self.canvas_network.ax
        ax.clear()
        ax.set_title("路网连接关系")
        ax.set_axis_off()

        if not self.network_preview:
            ax.text(0.5, 0.5, "暂无路网数据", ha="center", va="center", transform=ax.transAxes)
            self.canvas_network.draw()
            return

        nodes = self.network_preview.get("nodes", [])
        edges = self.network_preview.get("edges", [])
        if not nodes:
            ax.text(0.5, 0.5, "暂无路网数据", ha="center", va="center", transform=ax.transAxes)
            self.canvas_network.draw()
            return

        positions = {
            int(node["node_id"]): (float(node["x"]), float(node["y"]))
            for node in nodes
        }

        for edge in edges:
            source = int(edge["from"])
            target = int(edge["to"])
            if source not in positions or target not in positions:
                continue
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            ax.plot([x1, x2], [y1, y2], color="#cbd5e1", linewidth=0.6, alpha=0.45, zorder=1)

        node_ids = [int(node["node_id"]) for node in nodes]
        xy = np.array([positions[node_id] for node_id in node_ids], dtype=float)

        if self.current_prediction is not None and self.route_service is not None:
            _, scores = self.route_service.compute_congestion_scores(
                self.current_prediction,
                horizon_idx=self.spin_horizon_idx.value(),
            )
            node_colors = [self._risk_color(float(scores[node_id])) for node_id in node_ids]
            node_sizes = [18 + min(float(scores[node_id]), 2.0) * 14 for node_id in node_ids]
        else:
            node_colors = ["#64748b" for _ in node_ids]
            node_sizes = [18 for _ in node_ids]

        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=node_sizes,
            c=node_colors,
            alpha=0.85,
            edgecolors="#ffffff",
            linewidths=0.4,
            zorder=3,
        )

        candidates = []
        if self.current_recommendation:
            candidates = self.current_recommendation.get("candidates", [])
        selected_path = selected_route.get("path", []) if selected_route else []

        for idx, candidate in enumerate(candidates):
            path = candidate.get("path", [])
            if len(path) < 2:
                continue
            color = self.ROUTE_COLORS[idx % len(self.ROUTE_COLORS)]
            linewidth = 2.0
            alpha = 0.38
            if path == selected_path:
                linewidth = 4.0
                alpha = 0.95
            for source, target in zip(path[:-1], path[1:]):
                if source not in positions or target not in positions:
                    continue
                x1, y1 = positions[source]
                x2, y2 = positions[target]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha, zorder=4)

        source = self.spin_source.value()
        target = self.spin_target.value()
        for node_id, marker, color, label in [
            (source, "o", "#16a34a", "S"),
            (target, "s", "#dc2626", "T"),
        ]:
            if node_id not in positions:
                continue
            x, y = positions[node_id]
            ax.scatter([x], [y], s=120, marker=marker, c=color, edgecolors="#ffffff", linewidths=1.4, zorder=5)
            ax.text(x, y, label, ha="center", va="center", color="#ffffff", fontsize=8, fontweight="bold", zorder=6)

        if selected_path:
            label_nodes = selected_path
            if len(label_nodes) > 12:
                label_nodes = selected_path[:6] + selected_path[-6:]
            for node_id in label_nodes:
                if node_id not in positions:
                    continue
                x, y = positions[node_id]
                ax.text(x, y + 0.018, str(node_id), ha="center", va="bottom", fontsize=7, color="#0f172a", zorder=7)

        legend_items = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#0f766e", markersize=8, label="畅通"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#ca8a04", markersize=8, label="轻度"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#ea580c", markersize=8, label="中度"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", markersize=8, label="严重"),
            Line2D([0], [0], color="#0f766e", linewidth=2.0, label="候选路线"),
            Line2D([0], [0], color="#0f766e", linewidth=4.0, label="当前路线"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#16a34a", markersize=9, label="起点 S"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#dc2626", markersize=9, label="终点 T"),
        ]
        ax.legend(
            handles=legend_items,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            fontsize=8,
            frameon=True,
            framealpha=0.88,
        )

        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        self.canvas_network.figure.tight_layout()
        self.canvas_network.draw()

    def _draw_route_charts(self, route_nodes):
        self.canvas_route.ax_flow.clear()
        self.canvas_route.ax_risk.clear()

        if not route_nodes:
            self.canvas_route.ax_flow.set_title("暂无路线数据")
            self.canvas_route.ax_risk.set_title("暂无路线数据")
            self.canvas_route.draw()
            return

        labels = [str(row["node_id"]) for row in route_nodes]
        flows = [float(row["predicted_flow"]) for row in route_nodes]
        baselines = [float(row["baseline_flow"]) for row in route_nodes]
        scores = [float(row["congestion_score"]) for row in route_nodes]
        colors = [self._risk_color(score) for score in scores]
        x = np.arange(len(labels))

        self.canvas_route.ax_flow.plot(x, flows, marker="o", color="#0f766e", label="预测流量")
        self.canvas_route.ax_flow.plot(
            x,
            baselines,
            marker="s",
            color="#64748b",
            linestyle="--",
            label="参考流量(历史85%分位)",
        )
        self.canvas_route.ax_flow.set_xticks(x)
        self.canvas_route.ax_flow.set_xticklabels(labels, rotation=45, ha="right")
        self.canvas_route.ax_flow.set_title("路线节点预测流量")
        self.canvas_route.ax_flow.set_xlabel("节点")
        self.canvas_route.ax_flow.set_ylabel("流量")
        self.canvas_route.ax_flow.grid(True, linestyle="--", alpha=0.35)
        self.canvas_route.ax_flow.legend()

        self.canvas_route.ax_risk.bar(x, scores, color=colors)
        self.canvas_route.ax_risk.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.2, label="高风险阈值")
        self.canvas_route.ax_risk.set_xticks(x)
        self.canvas_route.ax_risk.set_xticklabels(labels, rotation=45, ha="right")
        self.canvas_route.ax_risk.set_title("路线节点拥堵指数")
        self.canvas_route.ax_risk.set_xlabel("节点")
        self.canvas_route.ax_risk.set_ylabel("指数")
        self.canvas_route.ax_risk.grid(True, axis="y", linestyle="--", alpha=0.35)
        self.canvas_route.ax_risk.legend()

        self.canvas_route.figure.tight_layout()
        self.canvas_route.draw()

    def export_report(self):
        rec = self.current_selected_route or self.current_recommendation
        if not rec:
            QMessageBox.warning(self, "提示", "请先生成路线建议。")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出应用报告",
            "route_recommendation_report.csv",
            "CSV Files (*.csv)",
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["section", "key", "value"])
                writer.writerow(["summary", "source", rec.get("source")])
                writer.writerow(["summary", "target", rec.get("target")])
                writer.writerow(["summary", "sample_index", self.spin_sample_index.value()])
                writer.writerow(["summary", "horizon", rec.get("horizon_idx", 0) + 1])
                writer.writerow(["summary", "strategy", self.combo_strategy.currentText()])
                writer.writerow(["summary", "selected_route_rank", rec.get("route_rank", 1)])
                writer.writerow(["summary", "path", rec.get("path_text", "")])
                writer.writerow(["summary", "distance", f"{rec.get('distance', 0.0):.6f}"])
                writer.writerow(["summary", "avg_congestion_score", f"{rec.get('avg_congestion_score', 0.0):.6f}"])
                writer.writerow(["summary", "max_congestion_score", f"{rec.get('max_congestion_score', 0.0):.6f}"])
                writer.writerow([])
                writer.writerow(["candidate", "rank", "distance", "avg_congestion_score", "max_congestion_score", "path"])
                for row in rec.get("candidates", []):
                    writer.writerow(
                        [
                            "candidate",
                            row.get("route_rank"),
                            f"{row.get('distance', 0.0):.6f}",
                            f"{row.get('avg_congestion_score', 0.0):.6f}",
                            f"{row.get('max_congestion_score', 0.0):.6f}",
                            row.get("path_text", ""),
                        ]
                    )
                writer.writerow([])
                writer.writerow(["route_node", "node_id", "predicted_flow", "baseline_flow", "congestion_score", "risk_level"])
                for row in rec.get("route_nodes", []):
                    writer.writerow(
                        [
                            "route_node",
                            row["node_id"],
                            f"{row['predicted_flow']:.6f}",
                            f"{row['baseline_flow']:.6f}",
                            f"{row['congestion_score']:.6f}",
                            row["risk_level"],
                        ]
                    )
                writer.writerow([])
                writer.writerow(["top_risk", "node_id", "predicted_flow", "baseline_flow", "congestion_score", "risk_level"])
                for row in rec.get("top_risk_nodes", []):
                    writer.writerow(
                        [
                            "top_risk",
                            row["node_id"],
                            f"{row['predicted_flow']:.6f}",
                            f"{row['baseline_flow']:.6f}",
                            f"{row['congestion_score']:.6f}",
                            row["risk_level"],
                        ]
                    )
            QMessageBox.information(self, "导出成功", f"已保存到:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
