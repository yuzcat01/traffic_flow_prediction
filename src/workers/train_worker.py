from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from services.train_service import TrainService


class TrainWorker(QObject):
    log_message = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    training_succeeded = pyqtSignal(str)   # 返回 model_name
    training_failed = pyqtSignal(str)
    training_stopped = pyqtSignal()
    finished = pyqtSignal()

    def __init__(
        self,
        data_cfg: str,
        train_cfg: str,
        model_cfg: str,
        overrides: dict | None = None,
        parent=None
    ):
        super().__init__(parent)
        self.service = TrainService(
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            overrides=overrides or {},
        )

    @pyqtSlot()
    def run(self):
        try:
            self.status_changed.emit("训练中")
            result = self.service.run(line_callback=self.log_message.emit)

            if result.get("status") == "stopped":
                self.status_changed.emit("已停止")
                self.training_stopped.emit()
            else:
                self.status_changed.emit("训练完成")
                self.training_succeeded.emit(result.get("model_name", ""))

        except Exception as e:
            self.status_changed.emit("训练失败")
            self.training_failed.emit(str(e))

        finally:
            self.finished.emit()

    @pyqtSlot()
    def stop(self):
        self.service.stop()