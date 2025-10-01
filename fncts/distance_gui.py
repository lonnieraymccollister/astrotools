#!/usr/bin/env python3
"""
distance_gui.py
Simple PyQt6 GUI to convert parallax to distance (parsecs and light years).
"""
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QComboBox, QGridLayout, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt

class DistanceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parallax -> Distance")
        self._build_ui()
        self.resize(520, 260)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        grid.addWidget(QLabel("Parallax value:"), 0, 0)
        self.parallax_edit = QLineEdit()
        self.parallax_edit.setPlaceholderText("e.g. 7.53")
        grid.addWidget(self.parallax_edit, 0, 1)

        grid.addWidget(QLabel("Units:"), 0, 2)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["milliarcseconds (mas)", "arcseconds (arcsec)"])
        grid.addWidget(self.unit_combo, 0, 3)

        self.compute_btn = QPushButton("Compute Distance")
        self.compute_btn.clicked.connect(self._compute)
        grid.addWidget(self.compute_btn, 1, 0, 1, 2)

        self.copy_btn = QPushButton("Copy Result")
        self.copy_btn.clicked.connect(self._copy_result)
        grid.addWidget(self.copy_btn, 1, 2, 1, 2)

        grid.addWidget(QLabel("Results:"), 2, 0)
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        grid.addWidget(self.result_box, 3, 0, 1, 4)

        grid.addWidget(QLabel("Log:"), 4, 0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(60)
        grid.addWidget(self.log, 5, 0, 1, 4)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))

    def _compute(self):
        val_text = self.parallax_edit.text().strip()
        if not val_text:
            QMessageBox.warning(self, "Input required", "Enter a parallax value.")
            return
        try:
            p = float(val_text)
        except Exception:
            QMessageBox.critical(self, "Parse error", "Parallax must be a number.")
            return

        unit = self.unit_combo.currentIndex()  # 0 = mas, 1 = arcsec
        if unit == 0:
            # milliarcseconds -> convert to arcseconds
            if p == 0:
                QMessageBox.critical(self, "Math error", "Parallax must be non-zero.")
                return
            p_arcsec = p / 1000.0
        else:
            p_arcsec = p

        if p_arcsec == 0:
            QMessageBox.critical(self, "Math error", "Parallax must be non-zero.")
            return

        # distance in parsecs = 1 / parallax(arcsec)
        distance_pc = 1.0 / p_arcsec
        # distance in light years
        distance_ly = 3.26156 * distance_pc  # 1 pc ≈ 3.26156 ly

        result_lines = [
            f"Input parallax: {p:.9g} {'mas' if unit==0 else 'arcsec'}",
            f"Parallax (arcsec): {p_arcsec:.9g} arcsec",
            f"Distance: {distance_pc:.9f} parsec",
            f"Distance: {distance_ly:.9f} light years"
        ]
        self.result_box.setPlainText("\n".join(result_lines))
        self._log(f"Computed distance for parallax {p:.9g} ({'mas' if unit==0 else 'arcsec'}) -> {distance_pc:.6g} pc, {distance_ly:.6g} ly")

    def _copy_result(self):
        txt = self.result_box.toPlainText().strip()
        if not txt:
            QMessageBox.information(self, "Nothing", "No result to copy.")
            return
        QApplication.clipboard().setText(txt)
        QMessageBox.information(self, "Copied", "Result copied to clipboard.")

def main():
    app = QApplication(sys.argv)
    w = DistanceWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()