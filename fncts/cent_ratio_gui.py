#!/usr/bin/env python3
"""
cent_ratio_gui.py
Simple PyQt6 GUI to compute distances between two star pairs in two images and their ratio.
"""
import sys
import math
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QGridLayout, QComboBox, QTextEdit, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt

class CentRatioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CentRatio — Distance and Ratio Calculator")
        self._build_ui()
        self.resize(560, 300)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        g = QGridLayout()
        central.setLayout(g)

        # Operation dropdown (example: ratio or scale)
        g.addWidget(QLabel("Operation:"), 0, 0)
        self.op_combo = QComboBox()
        self.op_combo.addItems(["Distance and Ratio", "Distance Only"])
        g.addWidget(self.op_combo, 0, 1, 1, 3)

        # Image 1 coordinates
        g.addWidget(QLabel("Image 1 — Star 1 X:"), 1, 0)
        self.i1_s1_x = QLineEdit("0.0")
        g.addWidget(self.i1_s1_x, 1, 1)
        g.addWidget(QLabel("Y:"), 1, 2)
        self.i1_s1_y = QLineEdit("0.0")
        g.addWidget(self.i1_s1_y, 1, 3)

        g.addWidget(QLabel("Image 1 — Star 2 X:"), 2, 0)
        self.i1_s2_x = QLineEdit("0.0")
        g.addWidget(self.i1_s2_x, 2, 1)
        g.addWidget(QLabel("Y:"), 2, 2)
        self.i1_s2_y = QLineEdit("0.0")
        g.addWidget(self.i1_s2_y, 2, 3)

        # Image 2 coordinates
        g.addWidget(QLabel("Image 2 — Star 1 X:"), 3, 0)
        self.i2_s1_x = QLineEdit("0.0")
        g.addWidget(self.i2_s1_x, 3, 1)
        g.addWidget(QLabel("Y:"), 3, 2)
        self.i2_s1_y = QLineEdit("0.0")
        g.addWidget(self.i2_s1_y, 3, 3)

        g.addWidget(QLabel("Image 2 — Star 2 X:"), 4, 0)
        self.i2_s2_x = QLineEdit("0.0")
        g.addWidget(self.i2_s2_x, 4, 1)
        g.addWidget(QLabel("Y:"), 4, 2)
        self.i2_s2_y = QLineEdit("0.0")
        g.addWidget(self.i2_s2_y, 4, 3)

        # Buttons
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._compute)
        g.addWidget(self.compute_btn, 5, 0, 1, 2)

        self.copy_btn = QPushButton("Copy Results")
        self.copy_btn.clicked.connect(self._copy_results)
        g.addWidget(self.copy_btn, 5, 2)

        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self._save_results)
        g.addWidget(self.save_btn, 5, 3)

        # Results box
        g.addWidget(QLabel("Results:"), 6, 0)
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        g.addWidget(self.results, 7, 0, 1, 4)

    def _parse_float(self, widget, name):
        txt = widget.text().strip()
        try:
            return float(txt)
        except Exception:
            raise ValueError(f"Invalid numeric value for {name}: '{txt}'")

    def _euclidean_distance(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def _compute(self):
        try:
            x1a = self._parse_float(self.i1_s1_x, "Image1 Star1 X")
            y1a = self._parse_float(self.i1_s1_y, "Image1 Star1 Y")
            x1b = self._parse_float(self.i1_s2_x, "Image1 Star2 X")
            y1b = self._parse_float(self.i1_s2_y, "Image1 Star2 Y")
            x2a = self._parse_float(self.i2_s1_x, "Image2 Star1 X")
            y2a = self._parse_float(self.i2_s1_y, "Image2 Star1 Y")
            x2b = self._parse_float(self.i2_s2_x, "Image2 Star2 X")
            y2b = self._parse_float(self.i2_s2_y, "Image2 Star2 Y")

            d1 = self._euclidean_distance(x1a, y1a, x1b, y1b)
            d2 = self._euclidean_distance(x2a, y2a, x2b, y2b)

            if d2 == 0:
                ratio = float('inf')
            else:
                ratio = d1 / d2

            lines = []
            lines.append(f"Distance Image 1: {d1:.6f} pixels")
            lines.append(f"Distance Image 2: {d2:.6f} pixels")
            lines.append(f"Ratio (d1 / d2): {ratio:.6f}")

            op = self.op_combo.currentText()
            if op == "Distance Only":
                # show just distances
                out_text = "\n".join(lines[:2])
            else:
                out_text = "\n".join(lines)

            self.results.setPlainText(out_text)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _copy_results(self):
        txt = self.results.toPlainText().strip()
        if not txt:
            QMessageBox.information(self, "Nothing", "No results to copy")
            return
        QApplication.clipboard().setText(txt)
        QMessageBox.information(self, "Copied", "Results copied to clipboard")

    def _save_results(self):
        txt = self.results.toPlainText().strip()
        if not txt:
            QMessageBox.information(self, "Nothing", "No results to save")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save results as text", "", "Text Files (*.txt);;All Files (*)")
        if not fn:
            return
        try:
            with open(fn, "w", encoding="utf-8") as f:
                f.write(txt + "\n")
            QMessageBox.information(self, "Saved", f"Results saved to:\n{fn}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

def main():
    app = QApplication(sys.argv)
    w = CentRatioWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()