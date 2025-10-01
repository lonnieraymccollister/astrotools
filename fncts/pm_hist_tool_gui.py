#!/usr/bin/env python3
"""
pm_hist_tool_gui.py
PyQt6 GUI combining:
- pm_vector_line: draw a vector line on a color image and save
- hist_match: match histogram of an image to a reference and save
"""
import sys
import os
import traceback
from pathlib import Path

import cv2
import numpy as np
from skimage.exposure import match_histograms

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

# ---------- helpers ----------
def read_image_for_preview(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    # convert BGR->RGB for display
    if img.ndim == 3 and img.shape[2] >= 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        rgb = img
    return img, rgb

def qpix_from_ndarray(rgb):
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        h, w = arr.shape
        arr8 = np.clip(arr, 0, 255).astype(np.uint8)
        qimg = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w, ch = arr.shape
        if arr.dtype != np.uint8:
            # scale floats/large ints to 0..255 for preview
            a = arr.astype(np.float64)
            mn = np.nanmin(a); mx = np.nanmax(a)
            if mx > mn:
                a = (a - mn) / (mx - mn) * 255.0
            else:
                a = np.zeros_like(a)
            arr8 = np.clip(a, 0, 255).astype(np.uint8)
        else:
            arr8 = arr
        if ch == 3:
            bytes_per_line = 3 * w
            qimg = QImage(arr8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        elif ch == 4:
            bytes_per_line = 4 * w
            qimg = QImage(arr8.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        else:
            # collapse channels for preview
            arr8 = arr8[..., :3]
            bytes_per_line = 3 * w
            qimg = QImage(arr8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

# ---------- GUI ----------
class ToolWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PM Vector Line & Histogram Match Tool")
        self._build_ui()
        self.resize(920, 520)
        self._loaded_preview = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        g = QGridLayout()
        central.setLayout(g)

        # Operation selector
        g.addWidget(QLabel("Operation:"), 0, 0)
        self.op_combo = QComboBox()
        self.op_combo.addItems(["Draw Vector Line", "Histogram Match"])
        self.op_combo.currentIndexChanged.connect(self._update_ui_for_op)
        g.addWidget(self.op_combo, 0, 1, 1, 3)

        # Common: input file 1
        g.addWidget(QLabel("Input image (target/ref):"), 1, 0)
        self.input1_edit = QLineEdit()
        g.addWidget(self.input1_edit, 1, 1, 1, 3)
        btn_in1 = QPushButton("Browse")
        btn_in1.clicked.connect(lambda: self._browse_file(self.input1_edit))
        g.addWidget(btn_in1, 1, 4)

        # For hist match: reference image
        g.addWidget(QLabel("Reference image (for hist):"), 2, 0)
        self.ref_edit = QLineEdit()
        g.addWidget(self.ref_edit, 2, 1, 1, 3)
        btn_ref = QPushButton("Browse")
        btn_ref.clicked.connect(lambda: self._browse_file(self.ref_edit))
        g.addWidget(btn_ref, 2, 4)

        # Output file
        g.addWidget(QLabel("Output filename:"), 3, 0)
        self.output_edit = QLineEdit("output.png")
        g.addWidget(self.output_edit, 3, 1, 1, 3)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_save)
        g.addWidget(btn_out, 3, 4)

        # ---------- Vector line params ----------
        g.addWidget(QLabel("Start X:"), 4, 0)
        self.start_x = QLineEdit("0")
        g.addWidget(self.start_x, 4, 1)
        g.addWidget(QLabel("Start Y:"), 4, 2)
        self.start_y = QLineEdit("0")
        g.addWidget(self.start_y, 4, 3)

        g.addWidget(QLabel("Delta X (mas_x):"), 5, 0)
        self.delta_x = QLineEdit("0")
        g.addWidget(self.delta_x, 5, 1)
        g.addWidget(QLabel("Delta Y (mas_y):"), 5, 2)
        self.delta_y = QLineEdit("0")
        g.addWidget(self.delta_y, 5, 3)

        g.addWidget(QLabel("Color B:"), 6, 0)
        self.col_b = QSpinBox(); self.col_b.setRange(0,255); self.col_b.setValue(255)
        g.addWidget(self.col_b, 6, 1)
        g.addWidget(QLabel("Color G:"), 6, 2)
        self.col_g = QSpinBox(); self.col_g.setRange(0,255); self.col_g.setValue(255)
        g.addWidget(self.col_g, 6, 3)
        g.addWidget(QLabel("Color R:"), 7, 0)
        self.col_r = QSpinBox(); self.col_r.setRange(0,255); self.col_r.setValue(255)
        g.addWidget(self.col_r, 7, 1)

        g.addWidget(QLabel("Thickness:"), 7, 2)
        self.thickness = QSpinBox(); self.thickness.setRange(1,50); self.thickness.setValue(1)
        g.addWidget(self.thickness, 7, 3)

        # ---------- Histogram match note (no extra numeric fields) ----------
        g.addWidget(QLabel("Note: hist match will match target (Input image) to Reference image"), 8, 0, 1, 4)

        # Buttons
        self.load_preview_btn = QPushButton("Load & Preview Input")
        self.load_preview_btn.clicked.connect(self._load_preview)
        g.addWidget(self.load_preview_btn, 9, 0)

        self.run_btn = QPushButton("Run Operation")
        self.run_btn.clicked.connect(self._on_run)
        g.addWidget(self.run_btn, 9, 1)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(lambda: self.log.clear())
        g.addWidget(self.clear_btn, 9, 2)

        # Preview label and log
        self.preview_label = QLabel("Preview")
        self.preview_label.setFixedSize(560, 380)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        g.addWidget(self.preview_label, 1, 5, 9, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        g.addWidget(self.log, 10, 0, 1, 5)

        self._update_ui_for_op()

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_file(self, line_edit):
        fn, _ = QFileDialog.getOpenFileName(self, "Select file", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)")
        if fn:
            line_edit.setText(fn)

    def _browse_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save output", "", "Images (*.png *.jpg *.tif *.tiff);;All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def _update_ui_for_op(self):
        op = self.op_combo.currentText()
        is_line = (op == "Draw Vector Line")
        # reference only needed for hist match
        self.ref_edit.setEnabled(not is_line)
        # vector fields enabled only for line
        for w in (self.start_x, self.start_y, self.delta_x, self.delta_y, self.col_b, self.col_g, self.col_r, self.thickness):
            w.setEnabled(is_line)

    def _load_preview(self):
        inpath = self.input1_edit.text().strip()
        if not inpath:
            QMessageBox.information(self, "Input required", "Select an input image first")
            return
        try:
            img, rgb = read_image_for_preview(inpath)
            pix = qpix_from_ndarray(rgb)
            if pix:
                self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._loaded_preview = img
            self._log(f"Loaded {Path(inpath).name} shape={getattr(img,'shape',None)} dtype={getattr(img,'dtype',None)}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"{e}\n\n{tb}")

    def _on_run(self):
        op = self.op_combo.currentText()
        if op == "Draw Vector Line":
            self._run_draw_line()
        else:
            self._run_hist_match()

    def _run_draw_line(self):
        inpath = self.input1_edit.text().strip()
        outpath = self.output_edit.text().strip()
        if not inpath or not os.path.exists(inpath):
            QMessageBox.warning(self, "Missing input", "Select an existing input image")
            return
        if not outpath:
            QMessageBox.warning(self, "Missing output", "Provide an output filename")
            return
        try:
            sx = int(self.start_x.text().strip())
            sy = int(self.start_y.text().strip())
            dx = int(self.delta_x.text().strip())
            dy = int(self.delta_y.text().strip())
            b = int(self.col_b.value()); g = int(self.col_g.value()); r = int(self.col_r.value())
            thickness = int(self.thickness.value())

            img = cv2.imread(inpath, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(inpath)

            # compute end point: original code used end = (sx + dx * -1, sy + dy * -1)
            end_x = sx + (dx * -1)
            end_y = sy + (dy * -1)
            start_pt = (sx, sy)
            end_pt = (end_x, end_y)
            color = (b, g, r)

            # if image is single channel, convert to BGR for color line
            write_img = img.copy()
            if write_img.ndim == 2:
                write_img = cv2.cvtColor(write_img, cv2.COLOR_GRAY2BGR)
            # draw line
            cv2.line(write_img, start_pt, end_pt, color, thickness=thickness)

            # save
            ok = cv2.imwrite(outpath, write_img)
            if not ok:
                raise IOError("Failed to write output image")
            self._log(f"Drew line {start_pt} -> {end_pt} color={color} thickness={thickness} saved to {outpath}")
            # update preview
            rgb = cv2.cvtColor(write_img, cv2.COLOR_BGR2RGB)
            pix = qpix_from_ndarray(rgb)
            if pix:
                self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            QMessageBox.information(self, "Done", f"Wrote {outpath}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

    def _run_hist_match(self):
        target = self.input1_edit.text().strip()
        reference = self.ref_edit.text().strip()
        outpath = self.output_edit.text().strip()
        if not target or not os.path.exists(target):
            QMessageBox.warning(self, "Missing target", "Select an existing target image")
            return
        if not reference or not os.path.exists(reference):
            QMessageBox.warning(self, "Missing reference", "Select an existing reference image")
            return
        if not outpath:
            QMessageBox.warning(self, "Missing output", "Provide an output filename")
            return
        try:
            img = cv2.imread(target, cv2.IMREAD_UNCHANGED)
            ref = cv2.imread(reference, cv2.IMREAD_UNCHANGED)
            if img is None or ref is None:
                raise FileNotFoundError("Could not read target or reference")

            # Ensure channel order and types for match_histograms
            # Convert BGR -> RGB for processing, preserve alpha if present
            def cv_to_proc(arr):
                if arr.ndim == 2:
                    return arr
                if arr.shape[2] >= 3:
                    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
                return arr

            proc_img = cv_to_proc(img)
            proc_ref = cv_to_proc(ref)

            # perform histogram matching; handle grayscale or multichannel
            if proc_img.ndim == 2:
                matched = match_histograms(proc_img, proc_ref, channel_axis=None)
                out = matched.astype(img.dtype)
            else:
                matched = match_histograms(proc_img, proc_ref, channel_axis=-1)
                # convert RGB back to BGR to save with OpenCV, and reattach extra channels if needed
                bgr = cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_RGB2BGR)
                if img.ndim == 3 and img.shape[2] > 3:
                    # reattach alpha or extra channels from original (preserve originals beyond 3)
                    extra = img[:, :, 3:]
                    out = np.concatenate([bgr, extra], axis=2)
                else:
                    out = bgr

            ok = cv2.imwrite(outpath, out)
            if not ok:
                raise IOError("Failed to write matched image")
            self._log(f"Histogram matched {Path(target).name} to {Path(reference).name} -> {outpath}")
            pix = qpix_from_ndarray(cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2RGB) if out.ndim == 3 else out)
            if pix:
                self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            QMessageBox.information(self, "Done", f"Wrote {outpath}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = ToolWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()