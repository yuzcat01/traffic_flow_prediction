from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.gui.pages.data_page import DataPage
from src.gui.pages.home_page import HomePage
from src.gui.pages.infer_page import InferPage
from src.gui.pages.application_page import ApplicationPage
from src.gui.pages.congestion_warning_page import CongestionWarningPage
from src.gui.pages.event_simulation_page import EventSimulationPage
from src.gui.pages.model_manage_page import ModelManagePage
from src.gui.pages.results_page import ResultsPage
from src.gui.pages.train_page import TrainPage
from src.gui.styles.qss import MAIN_QSS


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("交通流量预测系统")
        self.resize(1360, 860)

        self.predictor = None
        self.current_model_row = None

        self._init_ui()
        self.setStyleSheet(MAIN_QSS)

        self.model_manage_page.refresh_model_table()
        self.refresh_home_page()

        best_row = self.model_manage_page.get_best_model_row()
        if best_row is not None:
            self.load_model_from_row(best_row)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.sidebar = self._build_sidebar()
        self.content = self._build_content()

        root.addWidget(self.sidebar, 0)
        root.addLayout(self.content, 1)

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("SideBar")
        sidebar.setFixedWidth(208)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 18, 14, 18)
        layout.setSpacing(10)

        title = QLabel("交通流量\n预测平台")
        title.setObjectName("AppTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        subtitle = QLabel("Spatio-Temporal Graph Analytics")
        subtitle.setObjectName("SideNote")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(6)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        self.nav_home = self._make_nav_button("首页")
        self.nav_data = self._make_nav_button("数据管理")
        self.nav_train = self._make_nav_button("实验训练")
        self.nav_model = self._make_nav_button("模型管理")
        self.nav_infer = self._make_nav_button("在线推理")
        self.nav_results = self._make_nav_button("结果分析")

        self.nav_application = self._make_nav_button("路线规划")
        self.nav_warning = self._make_nav_button("拥堵预警")
        self.nav_event = self._make_nav_button("事件模拟")

        buttons = [
            self.nav_home,
            self.nav_data,
            self.nav_train,
            self.nav_model,
            self.nav_infer,
            self.nav_application,
            self.nav_warning,
            self.nav_event,
            self.nav_results,
        ]
        for i, btn in enumerate(buttons):
            self.btn_group.addButton(btn, i)
            layout.addWidget(btn)

        self.nav_home.setChecked(True)
        self.btn_group.buttonClicked[int].connect(self.switch_page)

        layout.addStretch(1)

        footer = QLabel("在首页查看系统概览，在结果分析页查看模型表现与实验对比。")
        footer.setObjectName("SideNote")
        footer.setWordWrap(True)
        layout.addWidget(footer)
        return sidebar

    @staticmethod
    def _make_nav_button(text):
        btn = QPushButton(text)
        btn.setObjectName("NavButton")
        btn.setCheckable(True)
        return btn

    @staticmethod
    def _wrap_scrollable_page(page_widget: QWidget, min_width=1160, min_height=800):
        page_widget.setMinimumSize(min_width, min_height)

        area = QScrollArea()
        area.setWidget(page_widget)
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setFrameShape(QFrame.NoFrame)
        return area

    def _build_content(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        topbar = QFrame()
        topbar.setObjectName("TopBar")
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(16, 12, 16, 12)
        topbar_layout.setSpacing(12)

        self.label_status = QLabel("状态：就绪")
        self.label_status.setObjectName("TopBadge")
        self.label_current_model = QLabel("当前模型：未加载")
        self.label_current_model.setObjectName("TopMeta")
        self.label_device = QLabel("设备：auto")
        self.label_device.setObjectName("TopMeta")

        topbar_layout.addWidget(self.label_status)
        topbar_layout.addStretch()
        topbar_layout.addWidget(self.label_current_model)
        topbar_layout.addSpacing(16)
        topbar_layout.addWidget(self.label_device)

        self.stack = QStackedWidget()

        self.home_page = HomePage()
        self.data_page = DataPage()
        self.train_page = TrainPage()
        self.model_manage_page = ModelManagePage()
        self.infer_page = InferPage()
        self.application_page = ApplicationPage()
        self.warning_page = CongestionWarningPage()
        self.event_page = EventSimulationPage()
        self.results_page = ResultsPage()

        self.stack.addWidget(self._wrap_scrollable_page(self.home_page, 1160, 820))
        self.stack.addWidget(self._wrap_scrollable_page(self.data_page, 1240, 900))
        self.stack.addWidget(self._wrap_scrollable_page(self.train_page, 1240, 940))
        self.stack.addWidget(self._wrap_scrollable_page(self.model_manage_page, 1240, 840))
        self.stack.addWidget(self._wrap_scrollable_page(self.infer_page, 1240, 840))
        self.stack.addWidget(self._wrap_scrollable_page(self.application_page, 1240, 900))
        self.stack.addWidget(self._wrap_scrollable_page(self.warning_page, 1240, 840))
        self.stack.addWidget(self._wrap_scrollable_page(self.event_page, 1240, 880))
        self.stack.addWidget(self._wrap_scrollable_page(self.results_page, 1240, 840))

        self.model_manage_page.set_load_callback(self.load_model_from_row)

        layout.addWidget(topbar)
        layout.addWidget(self.stack, 1)
        return layout

    def switch_page(self, index):
        self.stack.setCurrentIndex(index)

    def refresh_home_page(self):
        rows = self.model_manage_page.model_rows
        best_row = rows[0] if rows else None
        self.home_page.update_summary(
            best_row=best_row,
            total_runs=len(rows),
            rows=rows,
            current_model_text=self.label_current_model.text(),
            device_text=self.label_device.text(),
            status_text=self.label_status.text(),
        )

    def load_model_from_row(self, row):
        try:
            run_config_path = row.get("run_config_path")
            ckpt_path = row.get("ckpt_path")

            if not run_config_path:
                raise RuntimeError(f"未找到 run_config: {row.get('model_name', '')}")

            self.label_status.setText("状态：模型加载中...")

            from src.services.predictor import TrafficPredictor

            self.predictor = TrafficPredictor(
                run_config_path=run_config_path,
                checkpoint_path=ckpt_path,
                device="auto",
            )
            self.current_model_row = row

            current_model_text = (
                f"当前模型：{row.get('model_name', '')} | RMSE={row.get('rmse', 0.0):.4f}"
            )
            self.label_current_model.setText(current_model_text)

            train_device = "auto"
            try:
                train_device = self.predictor.train_cfg.get("device", "auto")
            except Exception:
                pass
            self.label_device.setText(f"设备：{train_device}")

            self.model_manage_page.current_model_row = row
            self.model_manage_page.label_current_model.setText(current_model_text)

            self.infer_page.set_predictor(self.predictor, row)
            self.application_page.set_predictor(self.predictor, row)
            self.warning_page.set_predictor(self.predictor, row)
            self.event_page.set_predictor(self.predictor, row)
            self.results_page.set_model_row(row)

            self.label_status.setText("状态：模型已加载")
            self.refresh_home_page()

        except Exception as e:
            self.label_status.setText("状态：加载失败")
            msg = str(e)
            if "No module named 'torch'" in msg:
                msg = (
                    "当前运行环境缺少 PyTorch（torch），无法进行模型加载、推理或训练。\n"
                    "请切换到已安装 torch 的环境后再运行图形界面。"
                )
            QMessageBox.critical(self, "模型加载失败", msg)
