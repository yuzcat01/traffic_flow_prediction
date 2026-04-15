import sys
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.project_paths import PROJECT_ROOT


def build_app_icon(size: int = 256) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # background
    painter.setBrush(QBrush(QColor("#2563eb")))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(8, 8, size - 16, size - 16, 40, 40)

    # road shape
    road = QPainterPath()
    road.moveTo(size * 0.30, size * 0.90)
    road.cubicTo(size * 0.40, size * 0.64, size * 0.58, size * 0.36, size * 0.72, size * 0.12)
    road.lineTo(size * 0.86, size * 0.20)
    road.cubicTo(size * 0.70, size * 0.46, size * 0.54, size * 0.70, size * 0.42, size * 0.92)
    road.closeSubpath()
    painter.setBrush(QBrush(QColor("#eff6ff")))
    painter.drawPath(road)

    # lane marks
    painter.setPen(QPen(QColor("#1d4ed8"), max(3, size // 40), Qt.SolidLine, Qt.RoundCap))
    painter.drawLine(int(size * 0.52), int(size * 0.78), int(size * 0.62), int(size * 0.60))
    painter.drawLine(int(size * 0.60), int(size * 0.58), int(size * 0.70), int(size * 0.40))
    painter.drawLine(int(size * 0.68), int(size * 0.38), int(size * 0.78), int(size * 0.22))

    # trend line
    painter.setPen(QPen(QColor("#f59e0b"), max(4, size // 36), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.drawLine(int(size * 0.12), int(size * 0.70), int(size * 0.30), int(size * 0.56))
    painter.drawLine(int(size * 0.30), int(size * 0.56), int(size * 0.46), int(size * 0.60))
    painter.drawLine(int(size * 0.46), int(size * 0.60), int(size * 0.62), int(size * 0.42))
    painter.drawLine(int(size * 0.62), int(size * 0.42), int(size * 0.84), int(size * 0.30))

    painter.end()
    return QIcon(pix)


def resolve_resource_path(relative_path: str) -> Path:
    base_dir = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
    candidates = [
        base_dir / relative_path,
        base_dir / "src" / relative_path,
        PROJECT_ROOT / relative_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main():
    app = QApplication(sys.argv)

    icon = None
    for rel in ("src/gui/assets/app_icon.ico", "src/gui/assets/app_icon.png"):
        candidate = resolve_resource_path(rel)
        if candidate.exists():
            icon = QIcon(str(candidate))
            if not icon.isNull():
                break

    if icon is None or icon.isNull():
        icon = build_app_icon()

    app.setWindowIcon(icon)
    win = MainWindow()
    win.setWindowIcon(icon)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
