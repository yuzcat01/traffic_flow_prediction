import csv
import numpy as np

from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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

        desc = QLabel("查看单样本预测结果，支持切换预测步，并可导出当前样本或批量评估结果。")
        desc.setStyleSheet("color: #6b7280; line-height: 1.6;")
        desc.setWordWrap(True)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(16)

        setting_group = QGroupBox("推理设置")
        setting_layout = QFormLayout(setting_group)

        self.spin_sample_index = QSpinBox()
        self.spin_sample_index.setMinimum(0)
        self.spin_sample_index.setMaximum(0)

        self.spin_node_index = QSpinBox()
        self.spin_node_index.setMinimum(0)
        self.spin_node_index.setMaximum(0)

        self.spin_horizon_idx = QSpinBox()
        self.spin_horizon_idx.setMinimum(0)
        self.spin_horizon_idx.setMaximum(0)

        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(3, 20)
        self.spin_topk.setValue(5)

        self.spin_batch_samples = QSpinBox()
        self.spin_batch_samples.setRange(0, 1000000)
        self.spin_batch_samples.setValue(200)
        self.spin_batch_samples.setToolTip("0 表示导出全部测试样本")

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

        all_chart_group = QGroupBox("全节点对比")
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
        self.text_predict_info.setMinimumHeight(220)
        info_group_layout.addWidget(self.text_predict_info)

        topk_group = QGroupBox("误差Top-K节点")
        topk_group_layout = QVBoxLayout(topk_group)
        self.text_topk = QTextEdit()
        self.text_topk.setReadOnly(True)
        self.text_topk.setMinimumHeight(220)
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

        if self.predictor is not None:
            self.spin_sample_index.setMaximum(max(0, self.predictor.get_test_size() - 1))
            self.spin_node_index.setMaximum(max(0, self.predictor.get_node_count() - 1))
            self.spin_horizon_idx.setMaximum(max(0, self.predictor.get_predict_steps() - 1))

    def run_prediction(self):
        if self.predictor is None:
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
            mape = float(np.mean(np.abs((pred_all[mask] - target_all[mask]) / target_all[mask])) * 100.0) if np.any(mask) else 0.0

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
            self._draw_all_nodes_chart(pred_all, target_all, node_id, sample_index, horizon_idx)
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
            )
            self._update_topk_text(abs_err_all, pred_all, target_all, topk)

        except Exception as e:
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
                    writer.writerow(["node_all_horizons", node_id, h, f"{pred_v:.6f}", f"{target_v:.6f}", f"{abs(pred_v - target_v):.6f}", ""])

                for n in range(pred_all_h.shape[0]):
                    for h in range(pred_all_h.shape[1]):
                        pred_v = float(pred_all_h[n, h])
                        target_v = float(target_all_h[n, h])
                        writer.writerow(["all_nodes_all_horizons", n, h, f"{pred_v:.6f}", f"{target_v:.6f}", f"{abs(pred_v - target_v):.6f}", ""])

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
            for idx in range(sample_count):
                sample = self.predictor.predict_test_sample(idx)
                pred_full = np.asarray(sample["prediction_full"], dtype=float)
                target_full = np.asarray(sample["target_full"], dtype=float)

                abs_err = np.abs(pred_full - target_full)
                mae_all = float(np.mean(abs_err))
                rmse_all = float(np.sqrt(np.mean((pred_full - target_full) ** 2)))
                mask = target_full > 1e-6
                mape_all = float(np.mean(np.abs((pred_full[mask] - target_full[mask]) / target_full[mask])) * 100.0) if np.any(mask) else 0.0

                pred_h0 = pred_full[:, 0]
                target_h0 = target_full[:, 0]
                abs_h0 = np.abs(pred_h0 - target_h0)
                mae_h0 = float(np.mean(abs_h0))
                rmse_h0 = float(np.sqrt(np.mean((pred_h0 - target_h0) ** 2)))

                rows.append([idx, predict_steps, f"{mae_all:.6f}", f"{mape_all:.6f}", f"{rmse_all:.6f}", f"{mae_h0:.6f}", f"{rmse_h0:.6f}"])

            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "sample_index",
                    "predict_steps",
                    "mae_all_horizons",
                    "mape_all_horizons_percent",
                    "rmse_all_horizons",
                    "mae_horizon0",
                    "rmse_horizon0",
                ])
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
        x_hist = list(range(len(history)))
        x_future = np.arange(len(history), len(history) + len(pred_node_all_h))

        ax.plot(x_hist, history, marker="o", label="History", color="#2563eb")
        ax.plot(x_future, target_node_all_h, marker="x", label="Future Target", color="#059669")
        ax.plot(x_future, pred_node_all_h, marker="o", label="Future Prediction", color="#f59e0b")

        selected_x = len(history) + horizon_idx
        if 0 <= horizon_idx < len(pred_node_all_h):
            ax.scatter([selected_x], [target_node_all_h[horizon_idx]], marker="x", s=110, color="#047857", label="Selected Target")
            ax.scatter([selected_x], [pred_node_all_h[horizon_idx]], marker="o", s=110, color="#b45309", label="Selected Prediction")
        ax.axvline(x=len(history) - 0.5, color="#9ca3af", linestyle="--", alpha=0.8)

        ax.set_title(f"Sample {sample_index} | Node {node_id} | Future Horizons ({len(pred_node_all_h)} steps)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Traffic Flow")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

        self.canvas_node.figure.tight_layout()
        self.canvas_node.draw()

    def _draw_all_nodes_chart(self, pred_all, target_all, node_id, sample_index, horizon_idx):
        ax = self.canvas_all_nodes.ax
        ax.clear()

        x = np.arange(len(pred_all))

        ax.plot(x, target_all, label="Target")
        ax.plot(x, pred_all, label="Prediction")
        ax.scatter([node_id], [target_all[node_id]], marker="x", s=90, label="Selected Target")
        ax.scatter([node_id], [pred_all[node_id]], marker="o", s=90, label="Selected Pred")

        ax.set_title(f"All Nodes | Sample {sample_index} | Horizon {horizon_idx + 1}")
        ax.set_xlabel("Node ID")
        ax.set_ylabel("Traffic Flow")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

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
    ):
        model_name = "-"
        if self.current_model_row is not None:
            model_name = self.current_model_row.get("model_name", "-")

        pred_node_all_h = np.asarray(pred_node_all_h, dtype=float)
        target_node_all_h = np.asarray(target_node_all_h, dtype=float)
        future_abs = np.abs(pred_node_all_h - target_node_all_h)
        future_mae = float(np.mean(future_abs)) if future_abs.size > 0 else 0.0
        future_rmse = float(np.sqrt(np.mean((pred_node_all_h - target_node_all_h) ** 2))) if future_abs.size > 0 else 0.0

        lines = [
            f"模型: {model_name}",
            f"样本索引: {sample_index}",
            f"节点ID: {node_id}",
            f"预测步: {horizon_idx + 1}/{predict_steps}",
            f"历史长度: {len(history)}",
            "",
            f"预测值: {pred_node:.4f}",
            f"真实值: {target_node:.4f}",
            f"绝对误差: {abs_error:.4f}",
            "",
            f"样本MAE: {mae:.4f}",
            f"样本RMSE: {rmse:.4f}",
            f"样本MAPE: {mape:.2f}%",
            "",
            f"节点未来MAE（{predict_steps}步）: {future_mae:.4f}",
            f"节点未来RMSE（{predict_steps}步）: {future_rmse:.4f}",
            "",
            "节点逐步详情:",
        ]
        for h in range(min(predict_steps, len(pred_node_all_h))):
            lines.append(
                f"H{h + 1}: pred={pred_node_all_h[h]:.4f} | target={target_node_all_h[h]:.4f} | abs_error={abs(pred_node_all_h[h] - target_node_all_h[h]):.4f}"
            )
        self.text_predict_info.setPlainText("\n".join(lines))

    def _update_topk_text(self, abs_err_all, pred_all, target_all, topk):
        indices = np.argsort(-abs_err_all)[:topk]

        lines = [f"Top-{len(indices)} 误差节点", ""]
        for rank, idx in enumerate(indices, start=1):
            lines.append(
                f"{rank}. node {idx} | pred={pred_all[idx]:.4f} | target={target_all[idx]:.4f} | abs_error={abs_err_all[idx]:.4f}"
            )

        self.text_topk.setPlainText("\n".join(lines))
