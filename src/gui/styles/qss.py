MAIN_QSS = """
QMainWindow {
    background: #edf2f7;
}

QWidget {
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
    color: #1e293b;
}

QFrame#SideBar {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 20px;
}

QLabel#AppTitle {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 0.4px;
}

QLabel#SideNote {
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.6;
}

QPushButton#NavButton {
    background: transparent;
    color: #cbd5e1;
    border: 1px solid transparent;
    text-align: left;
    padding: 10px 12px;
    border-radius: 12px;
    font-weight: 600;
}

QPushButton#NavButton:hover {
    background: rgba(148, 163, 184, 0.12);
    color: #f8fafc;
}

QPushButton#NavButton:checked {
    background: #f59e0b;
    color: #0f172a;
    border: 1px solid #fbbf24;
    font-weight: 700;
}

QFrame#TopBar {
    background: #ffffff;
    border: 1px solid #d9e2ef;
    border-radius: 16px;
}

QLabel#TopBadge {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 6px 12px;
    font-weight: 700;
}

QLabel#TopMeta {
    color: #334155;
    font-weight: 600;
}

QFrame#PagePanel {
    background: #ffffff;
    border: 1px solid #d9e2ef;
    border-radius: 20px;
}

QFrame#HeroPanel {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #fff7ed,
        stop: 0.55 #f8fafc,
        stop: 1 #ecfeff
    );
    border: 1px solid #e2e8f0;
    border-radius: 18px;
}

QFrame#HeroSummary {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
}

QLabel#HeroEyebrow {
    color: #b45309;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

QLabel#HeroTitle {
    color: #0f172a;
    font-size: 28px;
    font-weight: 800;
}

QLabel#HeroSubtitle {
    color: #475569;
    font-size: 14px;
}

QLabel#HeroBadge {
    background: rgba(15, 118, 110, 0.10);
    color: #115e59;
    border: 1px solid rgba(15, 118, 110, 0.18);
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 700;
}

QLabel#HeroSummaryTitle {
    color: #0f172a;
    font-size: 16px;
    font-weight: 800;
}

QLabel#HeroSummaryText {
    color: #334155;
    font-size: 13px;
    line-height: 1.7;
}

QFrame#MetricCard {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #ffffff,
        stop: 1 #f8fafc
    );
    border: 1px solid #d9e2ef;
    border-radius: 16px;
}

QLabel#MetricTitle {
    color: #64748b;
    font-size: 12px;
    font-weight: 600;
}

QLabel#MetricValue {
    color: #0f172a;
    font-size: 24px;
    font-weight: 800;
}

QPushButton {
    background: #0f766e;
    color: #ffffff;
    border: 1px solid #115e59;
    border-radius: 10px;
    padding: 8px 14px;
    font-weight: 700;
}

QPushButton:hover {
    background: #115e59;
}

QPushButton:pressed {
    background: #134e4a;
}

QLineEdit, QTextEdit, QAbstractSpinBox, QComboBox, QTableWidget {
    background: #ffffff;
    border: 1px solid #d1d9e6;
    border-radius: 10px;
    color: #0f172a;
    selection-background-color: #bfdbfe;
    selection-color: #0f172a;
}

QLineEdit, QAbstractSpinBox, QComboBox {
    padding: 6px 8px;
}

QTextEdit {
    padding: 8px 10px;
}

QLineEdit:focus, QTextEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QTableWidget:focus {
    border: 1px solid #60a5fa;
}

QComboBox QAbstractItemView {
    background: #ffffff;
    color: #111827;
    border: 1px solid #d1d9e6;
    selection-background-color: #dbeafe;
}

QTableWidget {
    gridline-color: #e2e8f0;
    alternate-background-color: #f8fafc;
}

QGroupBox {
    font-weight: 800;
    border: 1px solid #d9e2ef;
    border-radius: 16px;
    margin-top: 12px;
    padding-top: 12px;
    color: #0f172a;
    background: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #475569;
    background: #ffffff;
}

QHeaderView::section {
    background: #eef4fb;
    color: #0f172a;
    padding: 8px;
    border: 0;
    border-bottom: 1px solid #d1d9e6;
    font-weight: 800;
}

QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: #f1f5f9;
    width: 10px;
    margin: 2px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #cbd5e1;
    min-height: 30px;
    border-radius: 5px;
}

QScrollBar:horizontal {
    background: #f1f5f9;
    height: 10px;
    margin: 2px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #cbd5e1;
    min-width: 30px;
    border-radius: 5px;
}
"""
