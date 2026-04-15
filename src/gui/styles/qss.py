MAIN_QSS = """
QMainWindow {
    background: #f4f6fa;
}

QWidget {
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
    color: #1f2937;
}

QFrame#SideBar {
    background: #eef2f7;
    border: 1px solid #d6dbe6;
    border-radius: 10px;
}

QLabel#AppTitle {
    color: #111827;
    font-size: 16px;
    font-weight: 700;
}

QPushButton#NavButton {
    background: transparent;
    color: #374151;
    border: 1px solid transparent;
    text-align: left;
    padding: 8px 10px;
    border-radius: 8px;
}

QPushButton#NavButton:hover {
    background: #e5ebf5;
}

QPushButton#NavButton:checked {
    background: #dbe7ff;
    color: #1d4ed8;
    border: 1px solid #c0d4ff;
    font-weight: 700;
}

QFrame#TopBar {
    background: #ffffff;
    border: 1px solid #dce3ef;
    border-radius: 10px;
}

QFrame#PagePanel {
    background: #ffffff;
    border: 1px solid #dce3ef;
    border-radius: 10px;
}

QFrame#MetricCard {
    background: #ffffff;
    border: 1px solid #dce3ef;
    border-radius: 10px;
}

QLabel#MetricTitle {
    color: #6b7280;
    font-size: 12px;
}

QLabel#MetricValue {
    color: #111827;
    font-size: 24px;
    font-weight: 700;
}

QPushButton {
    background: #2563eb;
    color: #ffffff;
    border: 1px solid #1d4ed8;
    border-radius: 8px;
    padding: 6px 12px;
}

QPushButton:hover {
    background: #1d4ed8;
}

QPushButton:pressed {
    background: #1e40af;
}

QLineEdit, QTextEdit, QAbstractSpinBox, QComboBox, QTableWidget {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    color: #111827;
    selection-background-color: #bfdbfe;
    selection-color: #111827;
}

QLineEdit:focus, QTextEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QTableWidget:focus {
    border: 1px solid #60a5fa;
}

QComboBox QAbstractItemView {
    background: #ffffff;
    color: #111827;
    border: 1px solid #d1d5db;
    selection-background-color: #dbeafe;
}

QGroupBox {
    font-weight: 700;
    border: 1px solid #d9e0ea;
    border-radius: 10px;
    margin-top: 10px;
    padding-top: 10px;
    color: #111827;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #374151;
}

QHeaderView::section {
    background: #f3f4f6;
    color: #111827;
    padding: 6px;
    border: 0;
    border-bottom: 1px solid #d1d5db;
    font-weight: 700;
}

QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: #f3f4f6;
    width: 10px;
    margin: 1px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #cbd5e1;
    min-height: 30px;
    border-radius: 5px;
}

QScrollBar:horizontal {
    background: #f3f4f6;
    height: 10px;
    margin: 1px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #cbd5e1;
    min-width: 30px;
    border-radius: 5px;
}
"""
