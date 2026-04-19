import csv

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
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

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(300)


class InferPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.predictor = None
        self.current_model_row = None
        self.current_result = None
        self.current_topk_rows = []
        self._syncing_topk_selection = False

        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        panel = QFrame()
        panel.setObjectName("PagePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("在线推理")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        desc = QLabel(
            "这里用于查看测试集样本的单点预测、全节点误差和多步趋势。"
            "切换样本索引、节点编号或预测步后，页面会自动联动刷新。"
        )
        desc.setStyleSheet("color: #6b7280; line-height: 1.6;")
        desc.setWordWrap(True)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(16)

        setting_group = QGroupBox("推理设置")
        setting_layout = QFormLayout(setting_group)

        self.spin_sample_index = QSpinBox()
        self.spin_sample_index.setMinimum(0)
        self.spin_sample_index.setMaximum(0)
        self.spin_sample_index.valueChanged.connect(self._on_controls_changed)

        self.spin_node_index = QSpinBox()
        self.spin_node_index.setMinimum(0)
        self.spin_node_index.setMaximum(0)
        self.spin_node_index.valueChanged.connect(self._on_controls_changed)

        self.spin_horizon_idx = QSpinBox()
        self.spin_horizon_idx.setMinimum(0)
        self.spin_horizon_idx.setMaximum(0)
        self.spin_horizon_idx.valueChanged.connect(self._on_controls_changed)

        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(3, 20)
        self.spin_topk.setValue(5)
        self.spin_topk.valueChanged.connect(self._on_controls_changed)

        self.spin_batch_samples = QSpinBox()
        self.spin_batch_samples.setRange(0, 1000000)
        self.spin_batch_samples.setValue(200)
        self.spin_batch_samples.setToolTip("0 表示导出全部测试样本")

        self.label_sample_help = QLabel(
            "样本索引表示测试集滑动窗口编号。"
            "索引 0 是测试集里的第一个窗口，索引越大表示越靠后的时间片。"
        )
        self.label_sample_help.setWordWrap(True)
        self.label_sample_help.setStyleSheet(
            "padding: 10px 12px; background: #f8fafc; color: #475569; "
            "border: 1px solid #e2e8f0; border-radius: 8px;"
        )

        self.btn_predict = QPushButton("执行预测")
        self.btn_predict.clicked.connect(self.run_prediction)

        self.btn_export_current = QPushButton("导出当前样本")
        self.btn_export_current.clicked.connect(self.export_current_prediction)

        self.btn_export_batch = QPushButton("导出批量指标")
        self.btn_export_batch.clicked.connect(self.export_batch_metrics)

        setting_layout.addRow("样本索引:", self.spin_sample_index)
        setting_layout.addRow("节点编号:", self.spin_node_index)
        setting_layout.addRow("预测步索引:", self.spin_horizon_idx)
        setting_layout.addRow("误差节点 Top-K:", self.spin_topk)
        setting_layout.addRow("导出样本数(0=全部):", self.spin_batch_samples)
        setting_layout.addRow("说明:", self.label_sample_help)
        setting_layout.addRow("", self.btn_predict)
        setting_layout.addRow("", self.btn_export_current)
        setting_layout.addRow("", self.btn_export_batch)

        sample_metric_group = QGroupBox("当前样本指标")
        sample_metric_layout = QGridLayout(sample_metric_group)
        sample_metric_layout.setHorizontalSpacing(20)
        sample_metric_layout.setVerticalSpacing(12)

        self.label_pred_value = self._make_metric_value("-")
        self.label_target_value = self._make_metric_value("-")
        self.label_error_value = self._make_metric_value("-")
        self.label_sample_rmse = self._make_metric_value("-")
        self.label_sample_mae = self._make_metric_value("-")
        self.label_sample_mape = self._make_metric_value("-")

        sample_metric_layout.addWidget(self._make_metric_title("预测值"), 0, 0)
        sample_metric_layout.addWidget(self.label_pred_value, 1, 0)
        sample_metric_layout.addWidget(self._make_metric_title("真实值"), 0, 1)
        sample_metric_layout.addWidget(self.label_target_value, 1, 1)
        sample_metric_layout.addWidget(self._make_metric_title("绝对误差"), 0, 2)
        sample_metric_layout.addWidget(self.label_error_value, 1, 2)

        sample_metric_layout.addWidget(self._make_metric_title("RMSE"), 2, 0)
        sample_metric_layout.addWidget(self.label_sample_rmse, 3, 0)
        sample_metric_layout.addWidget(self._make_metric_title("MAE"), 2, 1)
        sample_metric_layout.addWidget(self.label_sample_mae, 3, 1)
        sample_metric_layout.addWidget(self._make_metric_title("MAPE"), 2, 2)
        sample_metric_layout.addWidget(self.label_sample_mape, 3, 2)

        top_layout.addWidget(setting_group, 1)
        top_layout.addWidget(sample_metric_group, 2)

        chart_layout = QHBoxLayout()
        chart_layout.setSpacing(16)

        node_chart_group = QGroupBox("单节点历史与多步预测")
        node_chart_layout = QVBoxLayout(node_chart_group)
        self.canvas_node = MplCanvas(self, width=8, height=4, dpi=100)
        node_chart_layout.addWidget(self.canvas_node)

        all_chart_group = QGroupBox("全节点绝对误差")
        all_chart_layout = QVBoxLayout(all_chart_group)
        self.canvas_all_nodes = MplCanvas(self, width=8, height=4, dpi=100)
        all_chart_layout.addWidget(self.canvas_all_nodes)

        chart_layout.addWidget(node_chart_group, 1)
        chart_layout.addWidget(all_chart_group, 1)

        text_layout = QHBoxLayout()
        text_layout.setSpacing(16)

        info_group = QGroupBox("推理详情")
        info_group_layout = QVBoxLayout(info_group)
        self.text_predict_info = QTextEdit()
        self.text_predict_info.setReadOnly(True)
        self.text_predict_info.setMinimumHeight(260)
        info_group_layout.addWidget(self.text_predict_info)

        topk_group = QGroupBox("误差 Top-K 节点")
        topk_group_layout = QVBoxLayout(topk_group)
        self.table_topk = QTableWidget()
        self.table_topk.setColumnCount(5)
        self.table_topk.setHorizontalHeaderLabels(["排名", "节点", "预测值", "真实值", "绝对误差"])
        self.table_topk.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_topk.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_topk.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_topk.verticalHeader().setVisible(False)
        self.table_topk.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_topk.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_topk.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table_topk.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table_topk.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table_topk.itemSelectionChanged.connect(self._on_topk_selection_changed)

        self.text_topk = QTextEdit()
        self.text_topk.setReadOnly(True)
        self.text_topk.setMinimumHeight(120)

        topk_group_layout.addWidget(self.table_topk)
        topk_group_layout.addWidget(self.text_topk)

        text_layout.addWidget(info_group, 2)
        text_layout.addWidget(topk_group, 1)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addLayout(top_layout)
        layout.addLayout(chart_layout)
        layout.addLayout(text_layout)

        root.addWidget(panel)

    def _make_metric_title(self, text: str):
        label = QLabel(text)
        label.setStyleSheet("color: #6b7280; font-size: 12px;")
        label.setWordWrap(True)
        return label

    def _make_metric_value(self, text: str):
        label = QLabel(text)
        label.setStyleSheet("font-size: 20px; font-weight: bold; color: #111827;")
        return label

    def set_predictor(self, predictor, current_model_row=None):
        self.predictor = predictor
        self.current_model_row = current_model_row
        self.current_result = None
        self.current_topk_rows = []

        if self.predictor is not None:
            self.spin_sample_index.setMaximum(max(0, self.predictor.get_test_size() - 1))
            self.spin_node_index.setMaximum(max(0, self.predictor.get_node_count() - 1))
            self.spin_horizon_idx.setMaximum(max(0, self.predictor.get_predict_steps() - 1))
            self.run_prediction(show_warnings=False)
        else:
            self._reset_view()

    def _reset_view(self):
        self.label_pred_value.setText("-")
        self.label_target_value.setText("-")
        self.label_error_value.setText("-")
        self.label_sample_rmse.setText("-")
        self.label_sample_mae.setText("-")
        self.label_sample_mape.setText("-")
        self.text_predict_info.setPlainText("尚未加载模型，请先在模型管理页加载模型。")
        self.text_topk.setPlainText("暂无误差 Top-K 数据。")
        self.table_topk.setRowCount(0)

        self.canvas_node.ax.clear()
        self.canvas_node.ax.set_title("暂无数据")
        self.canvas_node.draw()

        self.canvas_all_nodes.ax.clear()
        self.canvas_all_nodes.ax.set_title("暂无数据")
        self.canvas_all_nodes.draw()

    def _on_controls_changed(self):
        if self.predictor is None:
            return
        self.run_prediction(show_warnings=False)

    def run_prediction(self, show_warnings=True):
        if self.predictor is None:
            if show_warnings:
                QMessageBox.warning(self, "提示", "尚未加载模型，请先在模型管理页加载模型。")
            return

        sample_index = self.spin_sample_index.value()
        node_id = self.spin_node_index.value()
        horizon_idx = self.spin_horizon_idx.value()
        topk = self.spin_topk.value()

        try:
            result = self.predictor.get_test_sample_detail(sample_index, node_id, horizon_idx=horizon_idx)
            self.current_result = result

            pred_node = float(result["prediction"])
            target_node = float(result["target"])
            abs_error = float(result["abs_error"])

            pred_all = np.asarray(result["prediction_all_nodes"], dtype=float)
            target_all = np.asarray(result["target_all_nodes"], dtype=float)
            abs_err_all = np.abs(pred_all - target_all)
            pred_node_all_h = np.asarray(result["prediction_node_all_horizons"], dtype=float)
            target_node_all_h = np.asarray(result["target_node_all_horizons"], dtype=float)

            mae = float(np.mean(abs_err_all))
            rmse = float(np.sqrt(np.mean((pred_all - target_all) ** 2)))
            mask = target_all > 1e-6
            mape = (
                float(np.mean(np.abs((pred_all[mask] - target_all[mask]) / target_all[mask])) * 100.0)
                if np.any(mask)
                else 0.0
            )

            topk_indices = np.argsort(-abs_err_all)[:topk]
            self.current_topk_rows = [
                {
                    "rank": rank,
                    "node_id": int(idx),
                    "prediction": float(pred_all[idx]),
                    "target": float(target_all[idx]),
                    "abs_error": float(abs_err_all[idx]),
                }
                for rank, idx in enumerate(topk_indices, start=1)
            ]

            self.label_pred_value.setText(f"{pred_node:.4f}")
            self.label_target_value.setText(f"{target_node:.4f}")
            self.label_error_value.setText(f"{abs_error:.4f}")
            self.label_sample_rmse.setText(f"{rmse:.4f}")
            self.label_sample_mae.setText(f"{mae:.4f}")
            self.label_sample_mape.setText(f"{mape:.2f}%")

            self._draw_node_chart(
                history=result["history"],
                pred_node_all_h=pred_node_all_h,
                target_node_all_h=target_node_all_h,
                sample_index=sample_index,
                node_id=node_id,
                horizon_idx=horizon_idx,
            )
            self._draw_all_nodes_error_chart(
                abs_err_all=abs_err_all,
                pred_all=pred_all,
                target_all=target_all,
                node_id=node_id,
                sample_index=sample_index,
                horizon_idx=horizon_idx,
                topk_indices=topk_indices,
            )
            self._update_info_text(
                sample_index=sample_index,
                node_id=node_id,
                horizon_idx=horizon_idx,
                predict_steps=int(result.get("predict_steps", 1)),
                history=result["history"],
                pred_node=pred_node,
                target_node=target_node,
                abs_error=abs_error,
                mae=mae,
                rmse=rmse,
                mape=mape,
                pred_node_all_h=pred_node_all_h,
                target_node_all_h=target_node_all_h,
                pred_all_h=result["prediction_all_horizons"],
                target_all_h=result["target_all_horizons"],
            )
            self._update_topk_view(current_node_id=node_id)

        except Exception as e:
            if show_warnings:
                QMessageBox.critical(self, "推理失败", str(e))

    def export_current_prediction(self):
        if self.predictor is None:
            QMessageBox.warning(self, "提示", "尚未加载模型，请先在模型管理页加载模型。")
            return
        if self.current_result is None:
            QMessageBox.warning(self, "提示", "请先执行预测。")
            return

        sample_index = self.spin_sample_index.value()
        node_id = self.spin_node_index.value()

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前样本",
            f"sample_{sample_index}_node_{node_id}_prediction.csv",
            "CSV Files (*.csv)",
        )
        if not save_path:
            return

        try:
            history = np.asarray(self.current_result["history"], dtype=float)
            pred_all_h = np.asarray(self.current_result["prediction_all_horizons"], dtype=float)
            target_all_h = np.asarray(self.current_result["target_all_horizons"], dtype=float)

            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["section", "node_id", "horizon_idx", "pred", "target", "abs_error", "value"])

                for i, v in enumerate(history):
                    writer.writerow(["history", node_id, "", "", "", "", f"{v:.6f}"])

                for h in range(pred_all_h.shape[1]):
                    pred_v = float(pred_all_h[node_id, h])
                    target_v = float(target_all_h[node_id, h])
                    writer.writerow(
                        ["node_all_horizons", node_id, h, f"{pred_v:.6f}", f"{target_v:.6f}", f"{abs(pred_v - target_v):.6f}", ""]
                    )

                for n in range(pred_all_h.shape[0]):
                    for h in range(pred_all_h.shape[1]):
                        pred_v = float(pred_all_h[n, h])
                        target_v = float(target_all_h[n, h])
                        writer.writerow(
                            ["all_nodes_all_horizons", n, h, f"{pred_v:.6f}", f"{target_v:.6f}", f"{abs(pred_v - target_v):.6f}", ""]
                        )

            QMessageBox.information(self, "导出成功", f"已保存到:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def export_batch_metrics(self):
        if self.predictor is None:
            QMessageBox.warning(self, "提示", "尚未加载模型，请先在模型管理页加载模型。")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出批量指标",
            "inference_batch_metrics.csv",
            "CSV Files (*.csv)",
        )
        if not save_path:
            return

        try:
            total_samples = self.predictor.get_test_size()
            limit = self.spin_batch_samples.value()
            sample_count = total_samples if limit == 0 else min(limit, total_samples)
            predict_steps = self.predictor.get_predict_steps()

            rows = []
            horizon_headers = []
            for h in range(predict_steps):
                horizon_headers.extend(
                    [
                        f"mae_h{h + 1}",
                        f"mape_h{h + 1}_percent",
                        f"rmse_h{h + 1}",
                    ]
                )

            for idx in range(sample_count):
                sample = self.predictor.predict_test_sample(idx)
                pred_full = np.asarray(sample["prediction_full"], dtype=float)
                target_full = np.asarray(sample["target_full"], dtype=float)

                abs_err = np.abs(pred_full - target_full)
                mae_all = float(np.mean(abs_err))
                rmse_all = float(np.sqrt(np.mean((pred_full - target_full) ** 2)))
                mask = target_full > 1e-6
                mape_all = (
                    float(np.mean(np.abs((pred_full[mask] - target_full[mask]) / target_full[mask])) * 100.0)
                    if np.any(mask)
                    else 0.0
                )

                pred_h0 = pred_full[:, 0]
                target_h0 = target_full[:, 0]
                abs_h0 = np.abs(pred_h0 - target_h0)
                mae_h0 = float(np.mean(abs_h0))
                rmse_h0 = float(np.sqrt(np.mean((pred_h0 - target_h0) ** 2)))

                row = [
                    idx,
                    predict_steps,
                    f"{mae_all:.6f}",
                    f"{mape_all:.6f}",
                    f"{rmse_all:.6f}",
                    f"{mae_h0:.6f}",
                    f"{rmse_h0:.6f}",
                ]
                for h in range(predict_steps):
                    pred_h = pred_full[:, h]
                    target_h = target_full[:, h]
                    abs_h = np.abs(pred_h - target_h)
                    mae_h = float(np.mean(abs_h))
                    rmse_h = float(np.sqrt(np.mean((pred_h - target_h) ** 2)))
                    mask_h = target_h > 1e-6
                    mape_h = (
                        float(np.mean(np.abs((pred_h[mask_h] - target_h[mask_h]) / target_h[mask_h])) * 100.0)
                        if np.any(mask_h)
                        else 0.0
                    )
                    row.extend([f"{mae_h:.6f}", f"{mape_h:.6f}", f"{rmse_h:.6f}"])

                rows.append(row)

            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "sample_index",
                        "predict_steps",
                        "mae_all_horizons",
                        "mape_all_horizons_percent",
                        "rmse_all_horizons",
                        "mae_horizon0",
                        "rmse_horizon0",
                    ]
                    + horizon_headers
                )
                writer.writerows(rows)

            QMessageBox.information(self, "导出成功", f"已保存 {sample_count} 行到:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _draw_node_chart(self, history, pred_node_all_h, target_node_all_h, sample_index, node_id, horizon_idx):
        ax = self.canvas_node.ax
        ax.clear()

        history = np.asarray(history, dtype=float)
        pred_node_all_h = np.asarray(pred_node_all_h, dtype=float)
        target_node_all_h = np.asarray(target_node_all_h, dtype=float)
        x_hist = np.arange(-len(history) + 1, 1)
        x_future = np.arange(1, len(pred_node_all_h) + 1)

        ax.plot(x_hist, history, marker="o", label="History", color="#2563eb")
        ax.plot(x_future, target_node_all_h, marker="x", label="Future Target", color="#059669")
        ax.plot(x_future, pred_node_all_h, marker="o", label="Future Prediction", color="#f59e0b")

        selected_x = horizon_idx + 1
        if 0 <= horizon_idx < len(pred_node_all_h):
            ax.scatter(
                [selected_x],
                [target_node_all_h[horizon_idx]],
                marker="x",
                s=120,
                color="#047857",
                linewidths=2,
                label="Selected Target",
                zorder=4,
            )
            ax.scatter(
                [selected_x],
                [pred_node_all_h[horizon_idx]],
                marker="o",
                s=120,
                color="#b45309",
                label="Selected Prediction",
                zorder=4,
            )
        ax.axvline(x=0.0, color="#9ca3af", linestyle="--", alpha=0.8)

        ax.set_title(f"Sample {sample_index} | Node {node_id} | Future Horizons ({len(pred_node_all_h)} steps)")
        if len(history) > 0 and len(pred_node_all_h) > 0:
            ax.set_xlim(x_hist[0], x_future[-1])
        ax.set_xlabel("Relative Step (0 = last observed, 1..H = forecast horizon)")
        ax.set_ylabel("Traffic Flow")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

        self.canvas_node.figure.tight_layout()
        self.canvas_node.draw()

    def _draw_all_nodes_error_chart(self, abs_err_all, pred_all, target_all, node_id, sample_index, horizon_idx, topk_indices):
        ax = self.canvas_all_nodes.ax
        ax.clear()

        abs_err_all = np.asarray(abs_err_all, dtype=float)
        x = np.arange(len(abs_err_all))
        colors = np.full(len(abs_err_all), "#cbd5e1", dtype=object)
        colors[np.asarray(topk_indices, dtype=int)] = "#f59e0b"
        colors[int(node_id)] = "#0f766e"

        ax.bar(x, abs_err_all, color=colors, edgecolor="#94a3b8", linewidth=0.6)
        ax.scatter([node_id], [abs_err_all[node_id]], color="#0f766e", s=80, zorder=4, label="Selected Node")
        if len(topk_indices) > 0:
            ax.scatter(
                np.asarray(topk_indices, dtype=int),
                abs_err_all[np.asarray(topk_indices, dtype=int)],
                color="#b45309",
                s=45,
                zorder=4,
                label=f"Top-{len(topk_indices)} Error Nodes",
            )

        ax.set_title(f"All Nodes Absolute Error | Sample {sample_index} | Horizon {horizon_idx + 1}")
        ax.set_xlabel("Node ID")
        if len(abs_err_all) > 0:
            ax.set_xlim(-0.5, len(abs_err_all) - 0.5)
        ax.set_ylabel("Absolute Error")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend()

        detail = (
            f"selected pred={pred_all[node_id]:.4f} | "
            f"target={target_all[node_id]:.4f} | abs_error={abs_err_all[node_id]:.4f}"
        )
        ax.text(
            0.01,
            0.98,
            detail,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
        )

        self.canvas_all_nodes.figure.tight_layout()
        self.canvas_all_nodes.draw()

    def _update_info_text(
        self,
        sample_index,
        node_id,
        horizon_idx,
        predict_steps,
        history,
        pred_node,
        target_node,
        abs_error,
        mae,
        rmse,
        mape,
        pred_node_all_h,
        target_node_all_h,
        pred_all_h,
        target_all_h,
    ):
        model_name = "-"
        if self.current_model_row is not None:
            model_name = self.current_model_row.get("model_name", "-")

        pred_node_all_h = np.asarray(pred_node_all_h, dtype=float)
        target_node_all_h = np.asarray(target_node_all_h, dtype=float)
        future_abs = np.abs(pred_node_all_h - target_node_all_h)
        future_mae = float(np.mean(future_abs)) if future_abs.size > 0 else 0.0
        future_rmse = float(np.sqrt(np.mean((pred_node_all_h - target_node_all_h) ** 2))) if future_abs.size > 0 else 0.0
        pred_all_h = np.asarray(pred_all_h, dtype=float)
        target_all_h = np.asarray(target_all_h, dtype=float)

        lines = [
            f"模型: {model_name}",
            f"样本索引: {sample_index}",
            "样本含义: 测试集滑动窗口编号，0 表示测试集第一段输入窗口。",
            f"节点 ID: {node_id}",
            f"预测步: {horizon_idx + 1}/{predict_steps}",
            f"历史长度: {len(history)}",
            "",
            f"预测值: {pred_node:.4f}",
            f"真实值: {target_node:.4f}",
            f"绝对误差: {abs_error:.4f}",
            "",
            f"样本 MAE: {mae:.4f}",
            f"样本 RMSE: {rmse:.4f}",
            f"样本 MAPE: {mape:.2f}%",
            "",
            f"该节点未来 {predict_steps} 步 MAE: {future_mae:.4f}",
            f"该节点未来 {predict_steps} 步 RMSE: {future_rmse:.4f}",
            "",
            "节点逐步详情:",
        ]
        for h in range(min(predict_steps, len(pred_node_all_h))):
            lines.append(
                f"H{h + 1}: pred={pred_node_all_h[h]:.4f} | target={target_node_all_h[h]:.4f} | abs_error={abs(pred_node_all_h[h] - target_node_all_h[h]):.4f}"
            )

        if pred_all_h.ndim == 2 and target_all_h.ndim == 2 and pred_all_h.shape == target_all_h.shape:
            lines.append("")
            lines.append("全节点逐步指标:")
            for h in range(pred_all_h.shape[1]):
                pred_h = pred_all_h[:, h]
                target_h = target_all_h[:, h]
                abs_h = np.abs(pred_h - target_h)
                mae_h = float(np.mean(abs_h))
                rmse_h = float(np.sqrt(np.mean((pred_h - target_h) ** 2)))
                mask_h = target_h > 1e-6
                mape_h = (
                    float(np.mean(np.abs((pred_h[mask_h] - target_h[mask_h]) / target_h[mask_h])) * 100.0)
                    if np.any(mask_h)
                    else 0.0
                )
                lines.append(f"H{h + 1}: MAE={mae_h:.4f} | RMSE={rmse_h:.4f} | MAPE={mape_h:.2f}%")
        self.text_predict_info.setPlainText("\n".join(lines))

    def _update_topk_view(self, current_node_id: int):
        self._syncing_topk_selection = True
        self.table_topk.setRowCount(len(self.current_topk_rows))

        selected_row = -1
        lines = [f"Top-{len(self.current_topk_rows)} 误差节点", "点击表格中的节点，可直接联动切换当前节点。", ""]
        for row_idx, row in enumerate(self.current_topk_rows):
            values = [
                str(row["rank"]),
                str(row["node_id"]),
                f"{row['prediction']:.4f}",
                f"{row['target']:.4f}",
                f"{row['abs_error']:.4f}",
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.table_topk.setItem(row_idx, col_idx, item)
            if row["node_id"] == current_node_id:
                selected_row = row_idx
            lines.append(
                f"{row['rank']}. node {row['node_id']} | pred={row['prediction']:.4f} | target={row['target']:.4f} | abs_error={row['abs_error']:.4f}"
            )

        self.text_topk.setPlainText("\n".join(lines))
        if selected_row >= 0:
            self.table_topk.selectRow(selected_row)
        else:
            self.table_topk.clearSelection()
        self._syncing_topk_selection = False

    def _on_topk_selection_changed(self):
        if self._syncing_topk_selection:
            return
        selected_rows = self.table_topk.selectionModel().selectedRows()
        if not selected_rows:
            return

        row_idx = selected_rows[0].row()
        if row_idx < 0 or row_idx >= len(self.current_topk_rows):
            return

        node_id = int(self.current_topk_rows[row_idx]["node_id"])
        if node_id == self.spin_node_index.value():
            return

        self.spin_node_index.setValue(node_id)
