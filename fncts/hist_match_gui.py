#!/usr/bin/env python3
"""
hist_match_gui.py
PyQt6 GUI to match histogram of a target image to a reference image and save result.
"""
import sys
import os
import traceback

import cv2
import numpy as np
from skimage.exposure import match_histograms

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QTextEdit, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

def read_image_cv(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img

def to_rgb_for_display(img):
    if img.ndim == 2:
        arr = img
        # convert to 8-bit for display
        if arr.dtype != np.uint8:
            a = arr.astype(np.float64)
            mn, mx = np.nanmin(a), np.nanmax(a)
            if mx > mn:
                arr8 = ((a - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                arr8 = np.zeros_like(a, dtype=np.uint8)
        else:
            arr8 = arr
        qimg = QImage(arr8.data, arr8.shape[1], arr8.shape[0], arr8.shape[1], QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    else:
        # Multi-channel: convert BGR->RGB for display
        if img.shape[2] >= 3:
            rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            rgb = img
        if rgb.dtype != np.uint8:
            a = rgb.astype(np.float64)
            mn, mx = np.nanmin(a), np.nanmax(a)
            if mx > mn:
                rgb8 = ((a - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                rgb8 = np.zeros_like(a, dtype=np.uint8)
        else:
            rgb8 = rgb
        h, w, ch = rgb8.shape
        bytes_per_line = ch * w
        fmt = QImage.Format.Format_RGB888 if ch == 3 else QImage.Format.Format_RGBA8888
        qimg = QImage(rgb8.data, w, h, bytes_per_line, fmt)
        return QPixmap.fromImage(qimg)

class HistMatchWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histogram Match")
        self._build_ui()
        self.resize(920, 520)
        self.ref_img = None
        self.tgt_img = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        grid.addWidget(QLabel("Reference image:"), 0, 0)
        self.ref_edit = QLineEdit()
        grid.addWidget(self.ref_edit, 0, 1, 1, 3)
        btn_ref = QPushButton("Browse")
        btn_ref.clicked.connect(lambda: self._browse_file(self.ref_edit, self._load_ref))
        grid.addWidget(btn_ref, 0, 4)

        grid.addWidget(QLabel("Target image:"), 1, 0)
        self.tgt_edit = QLineEdit()
        grid.addWidget(self.tgt_edit, 1, 1, 1, 3)
        btn_tgt = QPushButton("Browse")
        btn_tgt.clicked.connect(lambda: self._browse_file(self.tgt_edit, self._load_tgt))
        grid.addWidget(btn_tgt, 1, 4)

        grid.addWidget(QLabel("Output filename:"), 2, 0)
        self.out_edit = QLineEdit("matched_output.png")
        grid.addWidget(self.out_edit, 2, 1, 1, 3)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_save)
        grid.addWidget(btn_out, 2, 4)

        grid.addWidget(QLabel("Channel mode:"), 3, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Auto (color if both images color)", "Force grayscale", "Force color (RGB)"])
        grid.addWidget(self.mode_combo, 3, 1, 1, 3)

        self.preview_ref_btn = QPushButton("Preview Reference")
        self.preview_ref_btn.clicked.connect(self._preview_ref)
        grid.addWidget(self.preview_ref_btn, 4, 0)

        self.preview_tgt_btn = QPushButton("Preview Target")
        self.preview_tgt_btn.clicked.connect(self._preview_tgt)
        grid.addWidget(self.preview_tgt_btn, 4, 1)

        self.run_btn = QPushButton("Run Histogram Match")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 4, 2)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(lambda: self.log.clear())
        grid.addWidget(self.clear_btn, 4, 3)

        # Preview panes
        self.ref_preview = QLabel("Reference preview")
        self.ref_preview.setFixedSize(280, 210)
        self.ref_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.ref_preview, 5, 0, 4, 1)

        self.tgt_preview = QLabel("Target preview")
        self.tgt_preview.setFixedSize(280, 210)
        self.tgt_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.tgt_preview, 5, 1, 4, 1)

        self.out_preview = QLabel("Result preview")
        self.out_preview.setFixedSize(280, 210)
        self.out_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.out_preview, 5, 2, 4, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 5, 3, 4, 2)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_file(self, edit_widget, callback=None):
        fn, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)")
        if fn:
            edit_widget.setText(fn)
            if callback:
                callback(fn)

    def _browse_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save matched image", "", "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;All Files (*)")
        if fn:
            self.out_edit.setText(fn)

    def _load_ref(self, path):
        try:
            img = read_image_cv(path)
            self.ref_img = img
            pix = to_rgb_for_display(img)
            if pix:
                self.ref_preview.setPixmap(pix.scaled(self.ref_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._log(f"Loaded reference: {os.path.basename(path)} shape={getattr(img,'shape',None)} dtype={getattr(img,'dtype',None)}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"Failed to load reference: {e}\n\n{tb}")

    def _load_tgt(self, path):
        try:
            img = read_image_cv(path)
            self.tgt_img = img
            pix = to_rgb_for_display(img)
            if pix:
                self.tgt_preview.setPixmap(pix.scaled(self.tgt_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._log(f"Loaded target: {os.path.basename(path)} shape={getattr(img,'shape',None)} dtype={getattr(img,'dtype',None)}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"Failed to load target: {e}\n\n{tb}")

    def _preview_ref(self):
        path = self.ref_edit.text().strip()
        if not path:
            QMessageBox.information(self, "No file", "Select a reference image first")
            return
        self._load_ref(path)

    def _preview_tgt(self):
        path = self.tgt_edit.text().strip()
        if not path:
            QMessageBox.information(self, "No file", "Select a target image first")
            return
        self._load_tgt(path)

    def _determine_mode(self):
        if self.mode_combo.currentIndex() == 1:
            return "gray"
        if self.mode_combo.currentIndex() == 2:
            return "color"
        # auto
        if self.ref_img is None or self.tgt_img is None:
            return "auto"
        ref_ch = 1 if self.ref_img.ndim == 2 else self.ref_img.shape[2]
        tgt_ch = 1 if self.tgt_img.ndim == 2 else self.tgt_img.shape[2]
        return "color" if (ref_ch >= 3 and tgt_ch >= 3) else "gray"

    def _on_run(self):
        ref_path = self.ref_edit.text().strip()
        tgt_path = self.tgt_edit.text().strip()
        out_path = self.out_edit.text().strip()

        if not ref_path or not os.path.exists(ref_path):
            QMessageBox.warning(self, "Missing reference", "Choose an existing reference image")
            return
        if not tgt_path or not os.path.exists(tgt_path):
            QMessageBox.warning(self, "Missing target", "Choose an existing target image")
            return
        if not out_path:
            QMessageBox.warning(self, "Missing output", "Specify an output filename")
            return

        try:
            ref = read_image_cv(ref_path)
            tgt = read_image_cv(tgt_path)
            mode = self._determine_mode()
            self._log(f"Running histogram match mode={mode}")

            # Prepare arrays for match_histograms
            if mode == "gray":
                # convert both to single-channel grayscale arrays
                def to_gray(a):
                    if a.ndim == 2:
                        return a
                    if a.shape[2] >= 3:
                        return cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2GRAY)
                    return a[:, :, 0]
                ref_g = to_gray(ref)
                tgt_g = to_gray(tgt)
                matched = match_histograms(tgt_g, ref_g, channel_axis=None)
                # matched could be float; convert to same dtype as target
                out = np.clip(matched, np.iinfo(tgt_g.dtype).min if np.issubdtype(tgt_g.dtype, np.integer) else np.nanmin(matched),
                              np.iinfo(tgt_g.dtype).max if np.issubdtype(tgt_g.dtype, np.integer) else np.nanmax(matched))
                out = out.astype(tgt_g.dtype)
            else:
                # color: convert BGR->RGB for processing
                def to_rgb(a):
                    if a.ndim == 2:
                        # replicate channels
                        return np.stack([a, a, a], axis=-1)
                    if a.shape[2] >= 3:
                        return cv2.cvtColor(a[:, :, :3], cv2.COLOR_BGR2RGB)
                    # if less channels, replicate
                    return np.repeat(a[:, :, :1], 3, axis=2)
                ref_rgb = to_rgb(ref)
                tgt_rgb = to_rgb(tgt)
                matched = match_histograms(tgt_rgb, ref_rgb, channel_axis=-1)
                # convert back to BGR for saving with OpenCV
                matched_uint = np.clip(matched, 0, 255).astype(np.uint8) if matched.dtype != np.uint8 else matched
                out = cv2.cvtColor(matched_uint, cv2.COLOR_RGB2BGR)

            # Save result
            ok = cv2.imwrite(out_path, out)
            if not ok:
                raise IOError("cv2.imwrite failed to save result")
            self._log(f"Saved matched image to {out_path}")
            # preview result
            pix = to_rgb_for_display(out if out is not None else matched)
            if pix:
                self.out_preview.setPixmap(pix.scaled(self.out_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            QMessageBox.information(self, "Done", f"Wrote matched image: {out_path}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = HistMatchWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()