from pathlib import Path

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
)

from services.config_service import ConfigService
from utils.config import load_yaml
from workers.train_worker import TrainWorker


class TrainPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_service = ConfigService()
        self.train_thread = None
        self.train_worker = None
        self._init_ui()
        self.refresh_config_options()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        panel = QFrame()
        panel.setObjectName("PagePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("实验训练")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        desc = QLabel(
            "先选择默认配置文件，再在本页覆盖常用参数。"
            "本页重点优化布局与参数分区，便于快速配置实验。"
        )
        desc.setStyleSheet("color: #6b7280; line-height: 1.7;")
        desc.setWordWrap(True)

        config_group = QGroupBox("默认配置文件")
        config_layout = QGridLayout(config_group)
        config_layout.setHorizontalSpacing(12)
        config_layout.setVerticalSpacing(12)

        self.combo_data_cfg = QComboBox()
        self.combo_train_cfg = QComboBox()
        self.combo_model_cfg = QComboBox()

        self.combo_data_cfg.currentIndexChanged.connect(self.load_defaults_from_selected_configs)
        self.combo_train_cfg.currentIndexChanged.connect(self.load_defaults_from_selected_configs)
        self.combo_model_cfg.currentIndexChanged.connect(self.load_defaults_from_selected_configs)

        self.combo_data_cfg.setMaximumWidth(360)
        self.combo_train_cfg.setMaximumWidth(360)
        self.combo_model_cfg.setMaximumWidth(360)

        config_layout.addWidget(QLabel("数据配置:"), 0, 0)
        config_layout.addWidget(self.combo_data_cfg, 0, 1)
        config_layout.addWidget(QLabel("训练配置:"), 1, 0)
        config_layout.addWidget(self.combo_train_cfg, 1, 1)
        config_layout.addWidget(QLabel("模型配置:"), 2, 0)
        config_layout.addWidget(self.combo_model_cfg, 2, 1)

        self.btn_generate_train_cfg = QPushButton("生成默认训练配置")
        self.btn_generate_model_cfg = QPushButton("生成默认模型配置")
        self.btn_generate_train_cfg.clicked.connect(self.generate_default_train_config)
        self.btn_generate_model_cfg.clicked.connect(self.generate_default_model_configs)

        generate_btn_layout = QHBoxLayout()
        generate_btn_layout.setSpacing(8)
        generate_btn_layout.addWidget(self.btn_generate_train_cfg)
        generate_btn_layout.addWidget(self.btn_generate_model_cfg)
        generate_btn_layout.addStretch()
        config_layout.addLayout(generate_btn_layout, 3, 0, 1, 2)
        config_layout.setColumnStretch(1, 1)

        params_group = QGroupBox("参数覆盖设置")
        params_main_layout = QVBoxLayout(params_group)
        params_main_layout.setSpacing(12)

        self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(1, 100000)
        self.spin_batch_size = QSpinBox(); self.spin_batch_size.setRange(1, 4096)
        self.spin_lr = QDoubleSpinBox(); self.spin_lr.setDecimals(6); self.spin_lr.setRange(0.000001, 10.0); self.spin_lr.setSingleStep(0.0001)
        self.spin_weight_decay = QDoubleSpinBox(); self.spin_weight_decay.setDecimals(6); self.spin_weight_decay.setRange(0.0, 1.0); self.spin_weight_decay.setSingleStep(0.0001)
        self.combo_optimizer = QComboBox(); self.combo_optimizer.addItems(["adamw", "adam"])
        self.spin_seed = QSpinBox(); self.spin_seed.setRange(0, 999999)
        self.spin_num_workers = QSpinBox(); self.spin_num_workers.setRange(0, 64)
        self.spin_grad_clip_norm = QDoubleSpinBox(); self.spin_grad_clip_norm.setDecimals(3); self.spin_grad_clip_norm.setRange(0.0, 1000.0); self.spin_grad_clip_norm.setSingleStep(0.5)
        self.combo_lr_scheduler = QComboBox(); self.combo_lr_scheduler.addItems(["plateau", "none"]); self.combo_lr_scheduler.currentIndexChanged.connect(self._refresh_param_enable_state)
        self.spin_lr_scheduler_factor = QDoubleSpinBox(); self.spin_lr_scheduler_factor.setDecimals(3); self.spin_lr_scheduler_factor.setRange(0.1, 0.99); self.spin_lr_scheduler_factor.setSingleStep(0.05)
        self.spin_lr_scheduler_patience = QSpinBox(); self.spin_lr_scheduler_patience.setRange(1, 100)
        self.spin_min_lr = QDoubleSpinBox(); self.spin_min_lr.setDecimals(6); self.spin_min_lr.setRange(0.000001, 1.0); self.spin_min_lr.setSingleStep(0.00001)
        self.check_shuffle = QCheckBox("打乱样本")
        self.combo_device = QComboBox(); self.combo_device.addItems(["auto", "cpu", "cuda"])
        self.spin_figure_node_id = QSpinBox(); self.spin_figure_node_id.setRange(0, 100000)
        self.spin_figure_points = QSpinBox(); self.spin_figure_points.setRange(1, 1000000)
        self.spin_figure_horizon_step = QSpinBox(); self.spin_figure_horizon_step.setRange(0, 0)

        self.combo_loss_fn = QComboBox(); self.combo_loss_fn.addItems(["mse", "mae", "huber"])
        self.combo_loss_fn.currentIndexChanged.connect(self._refresh_param_enable_state)
        self.spin_huber_delta = QDoubleSpinBox(); self.spin_huber_delta.setDecimals(4); self.spin_huber_delta.setRange(0.0001, 1000.0); self.spin_huber_delta.setSingleStep(0.1)
        self.combo_horizon_weight_mode = QComboBox(); self.combo_horizon_weight_mode.addItems(["uniform", "linear_decay", "exp_decay", "custom"])
        self.combo_horizon_weight_mode.currentIndexChanged.connect(self._refresh_param_enable_state)
        self.spin_horizon_weight_gamma = QDoubleSpinBox(); self.spin_horizon_weight_gamma.setDecimals(4); self.spin_horizon_weight_gamma.setRange(0.0001, 10.0); self.spin_horizon_weight_gamma.setSingleStep(0.05)
        self.edit_horizon_weights = QLineEdit(); self.edit_horizon_weights.setPlaceholderText("仅 custom 使用，例如 1,0.8,0.6,0.4")
        self.spin_val_ratio = QDoubleSpinBox(); self.spin_val_ratio.setDecimals(3); self.spin_val_ratio.setRange(0.0, 0.9); self.spin_val_ratio.setSingleStep(0.01)
        self.spin_early_stop_patience = QSpinBox(); self.spin_early_stop_patience.setRange(1, 1000)
        self.spin_early_stop_delta = QDoubleSpinBox(); self.spin_early_stop_delta.setDecimals(6); self.spin_early_stop_delta.setRange(0.0, 1.0); self.spin_early_stop_delta.setSingleStep(0.0001)

        self.label_model_template = QLabel("-")

        self.combo_spatial_type = QComboBox(); self.combo_spatial_type.addItems(["gcn", "chebnet", "gat"])
        self.combo_spatial_type.currentIndexChanged.connect(self._refresh_param_enable_state)

        self.combo_graph_type = QComboBox(); self.combo_graph_type.addItems(["connect", "distance", "correlation", "distance_correlation"])
        self.combo_graph_type.currentIndexChanged.connect(self._refresh_param_enable_state)

        self.spin_history_length = QSpinBox(); self.spin_history_length.setRange(1, 10000)
        self.spin_predict_steps = QSpinBox(); self.spin_predict_steps.setRange(1, 168); self.spin_predict_steps.valueChanged.connect(self._refresh_param_enable_state)

        self.spin_spatial_hidden = QSpinBox(); self.spin_spatial_hidden.setRange(1, 4096)
        self.spin_cheb_k = QSpinBox(); self.spin_cheb_k.setRange(1, 32)
        self.spin_gat_heads = QSpinBox(); self.spin_gat_heads.setRange(1, 16)

        self.spin_corr_topk = QSpinBox(); self.spin_corr_topk.setRange(1, 1024)
        self.spin_corr_threshold = QDoubleSpinBox(); self.spin_corr_threshold.setDecimals(3); self.spin_corr_threshold.setRange(0.0, 1.0); self.spin_corr_threshold.setSingleStep(0.01)
        self.check_use_abs_corr = QCheckBox("取绝对相关系数")
        self.spin_fusion_alpha = QDoubleSpinBox(); self.spin_fusion_alpha.setDecimals(3); self.spin_fusion_alpha.setRange(0.0, 1.0); self.spin_fusion_alpha.setSingleStep(0.05)

        self.combo_temporal_type = QComboBox(); self.combo_temporal_type.addItems(["gru", "none"])
        self.combo_temporal_type.currentIndexChanged.connect(self._refresh_param_enable_state)
        self.spin_temporal_hidden = QSpinBox(); self.spin_temporal_hidden.setRange(1, 4096)

        self.edit_run_suffix = QLineEdit(); self.edit_run_suffix.setPlaceholderText("可选，例如 exp1 / bs128 / testA")

        compact_controls = [
            self.spin_epochs, self.spin_batch_size, self.spin_lr, self.spin_weight_decay, self.combo_optimizer, self.spin_seed,
            self.spin_num_workers, self.spin_grad_clip_norm, self.combo_lr_scheduler, self.spin_lr_scheduler_factor,
            self.spin_lr_scheduler_patience, self.spin_min_lr, self.combo_device, self.spin_figure_node_id, self.spin_figure_points,
            self.spin_figure_horizon_step, self.combo_loss_fn, self.spin_huber_delta, self.combo_horizon_weight_mode,
            self.spin_horizon_weight_gamma, self.spin_val_ratio, self.spin_early_stop_patience, self.spin_early_stop_delta,
            self.combo_spatial_type, self.combo_graph_type, self.spin_history_length, self.spin_predict_steps, self.spin_spatial_hidden,
            self.spin_cheb_k, self.spin_gat_heads, self.spin_corr_topk, self.spin_corr_threshold, self.spin_fusion_alpha,
            self.combo_temporal_type, self.spin_temporal_hidden
        ]
        for ctrl in compact_controls:
            ctrl.setMaximumWidth(220)
        self.edit_horizon_weights.setMaximumWidth(420)
        self.edit_run_suffix.setMaximumWidth(420)

        train_group = QGroupBox("训练与优化参数")
        train_layout = QGridLayout(train_group)
        train_layout.setHorizontalSpacing(10)
        train_layout.setVerticalSpacing(8)
        row_l = 0
        train_layout.addWidget(QLabel("训练轮数(epochs):"), row_l, 0); train_layout.addWidget(self.spin_epochs, row_l, 1)
        train_layout.addWidget(QLabel("批大小(batch_size):"), row_l, 2); train_layout.addWidget(self.spin_batch_size, row_l, 3); row_l += 1
        train_layout.addWidget(QLabel("学习率(learning_rate):"), row_l, 0); train_layout.addWidget(self.spin_lr, row_l, 1)
        train_layout.addWidget(QLabel("权重衰减(weight_decay):"), row_l, 2); train_layout.addWidget(self.spin_weight_decay, row_l, 3); row_l += 1
        train_layout.addWidget(QLabel("优化器(optimizer):"), row_l, 0); train_layout.addWidget(self.combo_optimizer, row_l, 1)
        train_layout.addWidget(QLabel("随机种子(seed):"), row_l, 2); train_layout.addWidget(self.spin_seed, row_l, 3); row_l += 1
        train_layout.addWidget(QLabel("加载线程(num_workers):"), row_l, 0); train_layout.addWidget(self.spin_num_workers, row_l, 1)
        train_layout.addWidget(QLabel("梯度裁剪(grad_clip):"), row_l, 2); train_layout.addWidget(self.spin_grad_clip_norm, row_l, 3); row_l += 1
        train_layout.addWidget(QLabel("学习率调度器:"), row_l, 0); train_layout.addWidget(self.combo_lr_scheduler, row_l, 1)
        train_layout.addWidget(QLabel("衰减系数(factor):"), row_l, 2); train_layout.addWidget(self.spin_lr_scheduler_factor, row_l, 3); row_l += 1
        train_layout.addWidget(QLabel("调度耐心(patience):"), row_l, 0); train_layout.addWidget(self.spin_lr_scheduler_patience, row_l, 1)
        train_layout.addWidget(QLabel("最小学习率(min_lr):"), row_l, 2); train_layout.addWidget(self.spin_min_lr, row_l, 3); row_l += 1
        train_layout.addWidget(QLabel("设备(device):"), row_l, 0); train_layout.addWidget(self.combo_device, row_l, 1)
        train_layout.addWidget(QLabel("样本顺序:"), row_l, 2); train_layout.addWidget(self.check_shuffle, row_l, 3); row_l += 1
        train_layout.addWidget(QLabel("验证集比例(val_ratio):"), row_l, 0); train_layout.addWidget(self.spin_val_ratio, row_l, 1)
        train_layout.addWidget(QLabel("早停轮数(patience):"), row_l, 2); train_layout.addWidget(self.spin_early_stop_patience, row_l, 3); row_l += 1
        train_layout.addWidget(QLabel("早停阈值(min_delta):"), row_l, 0); train_layout.addWidget(self.spin_early_stop_delta, row_l, 1)
        train_layout.addWidget(QLabel("实验后缀(run_suffix):"), row_l, 2); train_layout.addWidget(self.edit_run_suffix, row_l, 3)

        model_group = QGroupBox("模型与图结构参数")
        model_layout = QGridLayout(model_group)
        model_layout.setHorizontalSpacing(10)
        model_layout.setVerticalSpacing(8)
        row_r = 0
        model_layout.addWidget(QLabel("模板模型名:"), row_r, 0); model_layout.addWidget(self.label_model_template, row_r, 1)
        model_layout.addWidget(QLabel("空间模块(spatial):"), row_r, 2); model_layout.addWidget(self.combo_spatial_type, row_r, 3); row_r += 1
        model_layout.addWidget(QLabel("时间模块(temporal):"), row_r, 0); model_layout.addWidget(self.combo_temporal_type, row_r, 1)
        model_layout.addWidget(QLabel("建图方式(graph):"), row_r, 2); model_layout.addWidget(self.combo_graph_type, row_r, 3); row_r += 1
        model_layout.addWidget(QLabel("历史长度(history):"), row_r, 0); model_layout.addWidget(self.spin_history_length, row_r, 1)
        model_layout.addWidget(QLabel("预测步数(steps):"), row_r, 2); model_layout.addWidget(self.spin_predict_steps, row_r, 3); row_r += 1
        model_layout.addWidget(QLabel("空间维度(hidden):"), row_r, 0); model_layout.addWidget(self.spin_spatial_hidden, row_r, 1)
        model_layout.addWidget(QLabel("时间维度(hidden):"), row_r, 2); model_layout.addWidget(self.spin_temporal_hidden, row_r, 3); row_r += 1
        model_layout.addWidget(QLabel("Cheb K:"), row_r, 0); model_layout.addWidget(self.spin_cheb_k, row_r, 1)
        model_layout.addWidget(QLabel("GAT 头数:"), row_r, 2); model_layout.addWidget(self.spin_gat_heads, row_r, 3); row_r += 1
        model_layout.addWidget(QLabel("相关图 Top-K:"), row_r, 0); model_layout.addWidget(self.spin_corr_topk, row_r, 1)
        model_layout.addWidget(QLabel("相关阈值(threshold):"), row_r, 2); model_layout.addWidget(self.spin_corr_threshold, row_r, 3); row_r += 1
        model_layout.addWidget(QLabel("绝对相关(use_abs):"), row_r, 0); model_layout.addWidget(self.check_use_abs_corr, row_r, 1)
        model_layout.addWidget(QLabel("融合系数(alpha):"), row_r, 2); model_layout.addWidget(self.spin_fusion_alpha, row_r, 3)

        loss_group = QGroupBox("损失与可视化参数")
        loss_layout = QGridLayout(loss_group)
        loss_layout.setHorizontalSpacing(10)
        loss_layout.setVerticalSpacing(8)
        row_b = 0
        loss_layout.addWidget(QLabel("损失函数(loss_fn):"), row_b, 0); loss_layout.addWidget(self.combo_loss_fn, row_b, 1)
        loss_layout.addWidget(QLabel("Huber delta:"), row_b, 2); loss_layout.addWidget(self.spin_huber_delta, row_b, 3); row_b += 1
        loss_layout.addWidget(QLabel("步长权重模式:"), row_b, 0); loss_layout.addWidget(self.combo_horizon_weight_mode, row_b, 1)
        loss_layout.addWidget(QLabel("gamma(exp_decay):"), row_b, 2); loss_layout.addWidget(self.spin_horizon_weight_gamma, row_b, 3); row_b += 1
        loss_layout.addWidget(QLabel("自定义权重(horizon_weights):"), row_b, 0); loss_layout.addWidget(self.edit_horizon_weights, row_b, 1, 1, 3); row_b += 1
        loss_layout.addWidget(QLabel("可视化节点ID:"), row_b, 0); loss_layout.addWidget(self.spin_figure_node_id, row_b, 1)
        loss_layout.addWidget(QLabel("绘图点数(points):"), row_b, 2); loss_layout.addWidget(self.spin_figure_points, row_b, 3); row_b += 1
        loss_layout.addWidget(QLabel("显示步索引(horizon_step):"), row_b, 0); loss_layout.addWidget(self.spin_figure_horizon_step, row_b, 1)

        upper_blocks = QHBoxLayout()
        upper_blocks.setSpacing(12)
        upper_blocks.addWidget(train_group, 1)
        upper_blocks.addWidget(model_group, 1)

        params_main_layout.addLayout(upper_blocks)
        params_main_layout.addWidget(loss_group)

        btn_layout = QHBoxLayout()
        self.btn_refresh_cfg = QPushButton("刷新配置列表")
        self.btn_reload_defaults = QPushButton("重载默认参数")
        self.btn_start_train = QPushButton("开始训练")
        self.btn_stop_train = QPushButton("停止训练")
        self.btn_stop_train.setEnabled(False)

        self.btn_refresh_cfg.clicked.connect(self.refresh_config_options)
        self.btn_reload_defaults.clicked.connect(self.load_defaults_from_selected_configs)
        self.btn_start_train.clicked.connect(self.start_training)
        self.btn_stop_train.clicked.connect(self.stop_training)

        btn_layout.addWidget(self.btn_refresh_cfg)
        btn_layout.addWidget(self.btn_reload_defaults)
        btn_layout.addWidget(self.btn_start_train)
        btn_layout.addWidget(self.btn_stop_train)
        btn_layout.addStretch()

        status_group = QGroupBox("训练状态")
        status_layout = QVBoxLayout(status_group)
        self.label_status = QLabel("当前状态：空闲")
        self.label_selected = QLabel("当前选择：")
        self.label_selected.setStyleSheet("color: #6b7280;")
        status_layout.addWidget(self.label_status)
        status_layout.addWidget(self.label_selected)

        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout(log_group)
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setPlaceholderText("训练日志将显示在此处...")
        log_layout.addWidget(self.text_log)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addWidget(config_group)
        layout.addWidget(params_group)
        layout.addLayout(btn_layout)
        layout.addWidget(status_group)
        layout.addWidget(log_group, 1)

        root.addWidget(panel)

    def refresh_config_options(self):
        data_items = self.config_service.list_data_configs()
        train_items = self.config_service.list_train_configs()
        model_items = self.config_service.list_model_configs()

        self._fill_combo(self.combo_data_cfg, data_items)
        self._fill_combo(self.combo_train_cfg, train_items)
        self._fill_combo(self.combo_model_cfg, model_items)

        self.append_log(">>> 配置列表已刷新")
        self.append_log(f"数据配置数量: {len(data_items)}")
        self.append_log(f"训练配置数量: {len(train_items)}")
        self.append_log(f"模型配置数量: {len(model_items)}")
        self.append_log("")

        self.load_defaults_from_selected_configs()

    @staticmethod
    def _select_combo_by_filename(combo: QComboBox, filename: str):
        if not filename:
            return
        idx = combo.findText(filename)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def generate_default_train_config(self):
        try:
            result = self.config_service.create_default_train_config(config_name="default_generated", overwrite=False)
            filename = Path(result.get("path", "")).name
            self.refresh_config_options()
            self._select_combo_by_filename(self.combo_train_cfg, filename)

            status = result.get("status", "")
            if status == "created":
                msg = f"已生成训练配置：{filename}"
            elif status == "exists":
                msg = f"训练配置已存在：{filename}"
            else:
                msg = f"训练配置处理完成：{filename}"

            self.append_log(f">>> {msg}")
            QMessageBox.information(self, "训练配置", msg)
        except Exception as e:
            QMessageBox.critical(self, "训练配置生成失败", str(e))

    def generate_default_model_configs(self):
        try:
            results = self.config_service.create_default_model_configs(overwrite=False)
            self.refresh_config_options()

            created_count = 0
            exists_count = 0
            created_names = []
            for item in results:
                name = Path(item.get("path", "")).name
                status = item.get("status", "")
                if status == "created":
                    created_count += 1
                    created_names.append(name)
                elif status == "exists":
                    exists_count += 1

            if created_names:
                self._select_combo_by_filename(self.combo_model_cfg, created_names[0])

            msg = f"默认模型配置处理完成：新增 {created_count} 个，已存在 {exists_count} 个。"
            self.append_log(f">>> {msg}")
            QMessageBox.information(self, "模型配置", msg)
        except Exception as e:
            QMessageBox.critical(self, "模型配置生成失败", str(e))

    def _fill_combo(self, combo: QComboBox, items):
        combo.blockSignals(True)
        combo.clear()
        for item in items:
            combo.addItem(item["name"], item["path"])
        combo.blockSignals(False)

    def _get_selected_paths(self):
        return self.combo_data_cfg.currentData(), self.combo_train_cfg.currentData(), self.combo_model_cfg.currentData()

    def _update_selected_label(self):
        text = (
            f"当前选择: data={self.combo_data_cfg.currentText() or '-'} | "
            f"train={self.combo_train_cfg.currentText() or '-'} | "
            f"model={self.combo_model_cfg.currentText() or '-'}"
        )
        self.label_selected.setText(text)

    def append_log(self, text: str):
        self.text_log.append(text)

    def load_defaults_from_selected_configs(self):
        _, train_cfg_path, model_cfg_path = self._get_selected_paths()
        self._update_selected_label()
        if not train_cfg_path or not model_cfg_path:
            return

        try:
            train_cfg = load_yaml(train_cfg_path).get("train", {})
            model_cfg = load_yaml(model_cfg_path).get("model", {})

            self.spin_epochs.setValue(int(train_cfg.get("epochs", 10)))
            self.spin_batch_size.setValue(int(train_cfg.get("batch_size", 64)))
            self.spin_lr.setValue(float(train_cfg.get("learning_rate", 0.001)))
            self.spin_weight_decay.setValue(float(train_cfg.get("weight_decay", 0.0)))
            optimizer = str(train_cfg.get("optimizer", "adamw"))
            idx = self.combo_optimizer.findText(optimizer)
            self.combo_optimizer.setCurrentIndex(idx if idx >= 0 else 0)
            self.spin_seed.setValue(int(train_cfg.get("seed", 42)))
            self.spin_num_workers.setValue(int(train_cfg.get("num_workers", 0)))
            self.spin_grad_clip_norm.setValue(float(train_cfg.get("grad_clip_norm", 0.0)))
            scheduler = str(train_cfg.get("lr_scheduler", "plateau"))
            idx = self.combo_lr_scheduler.findText(scheduler)
            self.combo_lr_scheduler.setCurrentIndex(idx if idx >= 0 else 0)
            self.spin_lr_scheduler_factor.setValue(float(train_cfg.get("lr_scheduler_factor", 0.5)))
            self.spin_lr_scheduler_patience.setValue(int(train_cfg.get("lr_scheduler_patience", 3)))
            self.spin_min_lr.setValue(float(train_cfg.get("min_lr", 0.00001)))
            self.check_shuffle.setChecked(bool(train_cfg.get("shuffle", True)))

            device = str(train_cfg.get("device", "auto"))
            idx = self.combo_device.findText(device)
            self.combo_device.setCurrentIndex(idx if idx >= 0 else 0)

            self.spin_figure_node_id.setValue(int(train_cfg.get("figure_node_id", 0)))
            self.spin_figure_points.setValue(int(train_cfg.get("figure_points", 300)))
            self.spin_figure_horizon_step.setValue(int(train_cfg.get("figure_horizon_step", 0)))

            loss_fn = str(train_cfg.get("loss_fn", "mse"))
            idx = self.combo_loss_fn.findText(loss_fn)
            self.combo_loss_fn.setCurrentIndex(idx if idx >= 0 else 0)
            self.spin_huber_delta.setValue(float(train_cfg.get("huber_delta", 1.0)))
            horizon_weight_mode = str(train_cfg.get("horizon_weight_mode", "uniform"))
            idx = self.combo_horizon_weight_mode.findText(horizon_weight_mode)
            self.combo_horizon_weight_mode.setCurrentIndex(idx if idx >= 0 else 0)
            self.spin_horizon_weight_gamma.setValue(float(train_cfg.get("horizon_weight_gamma", 0.9)))
            raw_weights = train_cfg.get("horizon_weights", [])
            if isinstance(raw_weights, (list, tuple)):
                self.edit_horizon_weights.setText(",".join([str(x) for x in raw_weights]))
            else:
                self.edit_horizon_weights.setText(str(raw_weights) if raw_weights is not None else "")

            self.spin_val_ratio.setValue(float(train_cfg.get("val_ratio", 0.1)))
            self.spin_early_stop_patience.setValue(int(train_cfg.get("early_stop_patience", 8)))
            self.spin_early_stop_delta.setValue(float(train_cfg.get("early_stop_min_delta", 0.0001)))

            self.label_model_template.setText(model_cfg.get("name", "-"))

            spatial_type = model_cfg.get("spatial", {}).get("type", "gcn")
            idx = self.combo_spatial_type.findText(spatial_type)
            self.combo_spatial_type.setCurrentIndex(idx if idx >= 0 else 0)

            graph_cfg = model_cfg.get("graph", {})
            graph_type = graph_cfg.get("type", "connect")
            idx = self.combo_graph_type.findText(graph_type)
            self.combo_graph_type.setCurrentIndex(idx if idx >= 0 else 0)

            self.spin_history_length.setValue(int(model_cfg.get("input", {}).get("history_length", 12)))
            self.spin_predict_steps.setValue(int(model_cfg.get("output", {}).get("predict_steps", 1)))
            self.spin_spatial_hidden.setValue(int(model_cfg.get("spatial", {}).get("hidden_dim", 16)))
            self.spin_cheb_k.setValue(int(model_cfg.get("spatial", {}).get("cheb_k", 3)))
            self.spin_gat_heads.setValue(int(model_cfg.get("spatial", {}).get("heads", 1)))

            self.spin_corr_topk.setValue(int(graph_cfg.get("correlation_topk", 8)))
            self.spin_corr_threshold.setValue(float(graph_cfg.get("correlation_threshold", 0.3)))
            self.check_use_abs_corr.setChecked(bool(graph_cfg.get("use_abs_corr", False)))
            self.spin_fusion_alpha.setValue(float(graph_cfg.get("fusion_alpha", 0.5)))

            temporal_type = model_cfg.get("temporal", {}).get("type", "gru")
            idx = self.combo_temporal_type.findText(temporal_type)
            self.combo_temporal_type.setCurrentIndex(idx if idx >= 0 else 0)
            self.spin_temporal_hidden.setValue(int(model_cfg.get("temporal", {}).get("hidden_dim", 32)))

            self._refresh_param_enable_state()

        except Exception as e:
            QMessageBox.critical(self, "读取默认参数失败", str(e))

    def _refresh_param_enable_state(self):
        spatial_type = self.combo_spatial_type.currentText().strip().lower()
        temporal_type = self.combo_temporal_type.currentText().strip().lower()
        graph_type = self.combo_graph_type.currentText().strip().lower()
        loss_fn = self.combo_loss_fn.currentText().strip().lower()
        horizon_weight_mode = self.combo_horizon_weight_mode.currentText().strip().lower()
        lr_scheduler = self.combo_lr_scheduler.currentText().strip().lower()

        self.spin_cheb_k.setEnabled(spatial_type == "chebnet")
        self.spin_gat_heads.setEnabled(spatial_type == "gat")
        self.spin_temporal_hidden.setEnabled(temporal_type == "gru")

        self.spin_figure_horizon_step.setRange(0, max(0, self.spin_predict_steps.value() - 1))
        if self.spin_figure_horizon_step.value() >= self.spin_predict_steps.value():
            self.spin_figure_horizon_step.setValue(max(0, self.spin_predict_steps.value() - 1))

        need_corr = graph_type in {"correlation", "distance_correlation"}
        self.spin_corr_topk.setEnabled(need_corr)
        self.spin_corr_threshold.setEnabled(need_corr)
        self.check_use_abs_corr.setEnabled(need_corr)
        self.spin_fusion_alpha.setEnabled(graph_type == "distance_correlation")

        self.spin_huber_delta.setEnabled(loss_fn == "huber")
        self.spin_horizon_weight_gamma.setEnabled(horizon_weight_mode == "exp_decay")
        self.edit_horizon_weights.setEnabled(horizon_weight_mode == "custom")
        scheduler_enabled = lr_scheduler == "plateau"
        self.spin_lr_scheduler_factor.setEnabled(scheduler_enabled)
        self.spin_lr_scheduler_patience.setEnabled(scheduler_enabled)
        self.spin_min_lr.setEnabled(scheduler_enabled)

    @staticmethod
    def _parse_horizon_weights_text(text: str):
        text = (text or "").strip()
        if text == "":
            return []

        values = []
        for token in text.replace(";", ",").split(","):
            token = token.strip()
            if token == "":
                continue
            value = float(token)
            if value <= 0:
                raise ValueError("horizon_weights 中每个值都必须 > 0")
            values.append(value)
        return values

    def _build_overrides(self):
        spatial_type = self.combo_spatial_type.currentText().strip().lower()
        temporal_type = self.combo_temporal_type.currentText().strip().lower()
        graph_type = self.combo_graph_type.currentText().strip().lower()
        horizon_weight_mode = self.combo_horizon_weight_mode.currentText().strip().lower()
        custom_horizon_weights = []
        if horizon_weight_mode == "custom":
            custom_horizon_weights = self._parse_horizon_weights_text(self.edit_horizon_weights.text())
            if len(custom_horizon_weights) != self.spin_predict_steps.value():
                raise ValueError(
                    f"自定义 horizon_weights 数量必须等于 predict_steps ({self.spin_predict_steps.value()})。"
                )

        graph_cfg = {"type": graph_type}
        if graph_type in {"correlation", "distance_correlation"}:
            graph_cfg.update({
                "correlation_topk": self.spin_corr_topk.value(),
                "correlation_threshold": float(self.spin_corr_threshold.value()),
                "use_abs_corr": self.check_use_abs_corr.isChecked(),
            })
        if graph_type == "distance_correlation":
            graph_cfg["fusion_alpha"] = float(self.spin_fusion_alpha.value())

        overrides = {
            "train": {
                "epochs": self.spin_epochs.value(),
                "batch_size": self.spin_batch_size.value(),
                "learning_rate": float(self.spin_lr.value()),
                "weight_decay": float(self.spin_weight_decay.value()),
                "optimizer": self.combo_optimizer.currentText(),
                "seed": self.spin_seed.value(),
                "num_workers": self.spin_num_workers.value(),
                "grad_clip_norm": float(self.spin_grad_clip_norm.value()),
                "lr_scheduler": self.combo_lr_scheduler.currentText(),
                "lr_scheduler_factor": float(self.spin_lr_scheduler_factor.value()),
                "lr_scheduler_patience": self.spin_lr_scheduler_patience.value(),
                "min_lr": float(self.spin_min_lr.value()),
                "shuffle": self.check_shuffle.isChecked(),
                "device": self.combo_device.currentText(),
                "figure_node_id": self.spin_figure_node_id.value(),
                "figure_points": self.spin_figure_points.value(),
                "figure_horizon_step": self.spin_figure_horizon_step.value(),
                "loss_fn": self.combo_loss_fn.currentText(),
                "huber_delta": float(self.spin_huber_delta.value()),
                "horizon_weight_mode": self.combo_horizon_weight_mode.currentText(),
                "horizon_weight_gamma": float(self.spin_horizon_weight_gamma.value()),
                "horizon_weights": custom_horizon_weights if horizon_weight_mode == "custom" else [],
                "val_ratio": float(self.spin_val_ratio.value()),
                "early_stop_patience": self.spin_early_stop_patience.value(),
                "early_stop_min_delta": float(self.spin_early_stop_delta.value()),
            },
            "model": {
                "graph": graph_cfg,
                "input": {"history_length": self.spin_history_length.value()},
                "output": {"predict_steps": self.spin_predict_steps.value(), "output_dim": 1},
                "spatial": {"type": spatial_type, "hidden_dim": self.spin_spatial_hidden.value()},
                "temporal": {"type": temporal_type},
            },
            "meta": {"run_suffix": self.edit_run_suffix.text().strip()},
        }

        if spatial_type == "chebnet":
            overrides["model"]["spatial"]["cheb_k"] = self.spin_cheb_k.value()
        if spatial_type == "gat":
            overrides["model"]["spatial"]["heads"] = self.spin_gat_heads.value()
        if temporal_type == "gru":
            overrides["model"]["temporal"]["hidden_dim"] = self.spin_temporal_hidden.value()

        return overrides

    def start_training(self):
        if self.train_thread is not None:
            QMessageBox.information(self, "提示", "已有训练任务正在运行。")
            return

        data_cfg, train_cfg, model_cfg = self._get_selected_paths()
        self._update_selected_label()
        if not data_cfg or not train_cfg or not model_cfg:
            QMessageBox.warning(self, "提示", "请先选择完整配置文件。")
            return

        try:
            overrides = self._build_overrides()
        except Exception as e:
            QMessageBox.warning(self, "参数无效", str(e))
            return

        self.text_log.clear()
        self.append_log(">>> 准备启动训练")
        self.append_log(f"data_cfg  = {data_cfg}")
        self.append_log(f"train_cfg = {train_cfg}")
        self.append_log(f"model_cfg = {model_cfg}")
        self.append_log(">>> 页面参数覆盖:")
        self.append_log(str(overrides))
        self.append_log("")

        self.label_status.setText("当前状态：训练中")
        self.btn_start_train.setEnabled(False)
        self.btn_refresh_cfg.setEnabled(False)
        self.btn_reload_defaults.setEnabled(False)
        self.btn_stop_train.setEnabled(True)

        self.train_thread = QThread()
        self.train_worker = TrainWorker(data_cfg=data_cfg, train_cfg=train_cfg, model_cfg=model_cfg, overrides=overrides)
        self.train_worker.moveToThread(self.train_thread)

        self.train_thread.started.connect(self.train_worker.run)
        self.train_worker.log_message.connect(self.append_log)
        self.train_worker.status_changed.connect(self._on_status_changed)
        self.train_worker.training_succeeded.connect(self._on_training_succeeded)
        self.train_worker.training_failed.connect(self._on_training_failed)
        self.train_worker.training_stopped.connect(self._on_training_stopped)
        self.train_worker.finished.connect(self._cleanup_training)

        self.train_thread.start()

    def stop_training(self):
        if self.train_worker is None:
            return
        self.append_log("")
        self.append_log(">>> 正在请求停止...")
        self.train_worker.stop()
        self.btn_stop_train.setEnabled(False)

    def _on_status_changed(self, status: str):
        self.label_status.setText(f"当前状态：{status}")

    def _on_training_succeeded(self, model_name: str):
        self.append_log("")
        self.append_log(">>> 训练完成")
        if model_name:
            self.append_log(f">>> 实验名称: {model_name}")

        self._refresh_related_pages()
        QMessageBox.information(self, "完成", f"训练完成。\n实验: {model_name or '-'}")

    def _on_training_failed(self, err_msg: str):
        self.append_log("")
        self.append_log(">>> 训练失败")
        self.append_log(err_msg)
        QMessageBox.critical(self, "训练失败", err_msg)

    def _on_training_stopped(self):
        self.append_log("")
        self.append_log(">>> 训练已停止")
        QMessageBox.information(self, "已停止", "训练任务已停止。")

    def _cleanup_training(self):
        self.btn_start_train.setEnabled(True)
        self.btn_refresh_cfg.setEnabled(True)
        self.btn_reload_defaults.setEnabled(True)
        self.btn_stop_train.setEnabled(False)

        if self.train_thread is not None:
            self.train_thread.quit()
            self.train_thread.wait()
            self.train_thread.deleteLater()

        if self.train_worker is not None:
            self.train_worker.deleteLater()

        self.train_thread = None
        self.train_worker = None

    def _refresh_related_pages(self):
        main_window = self.window()

        if hasattr(main_window, "model_manage_page"):
            try:
                main_window.model_manage_page.refresh_model_table()
            except Exception as e:
                self.append_log(f">>> 刷新模型页失败: {e}")

        if hasattr(main_window, "refresh_home_page"):
            try:
                main_window.refresh_home_page()
            except Exception as e:
                self.append_log(f">>> 刷新首页失败: {e}")
