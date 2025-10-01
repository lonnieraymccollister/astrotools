#!/usr/bin/env python3
"""
jpgcomp_gui.py
Simple PyQt6 GUI to compress images to JPEG with a user-selected quality.
"""

import sys
import os
import traceback
import numpy as np
import cv2
import tifffile

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QSpinBox, QTextEdit, QMessageBox, QHBoxLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

class JpgCompWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JPEG Compressor")
        self._build_ui()
        self.resize(800, 520)
        self._loaded_image = None  # numpy array in BGR or grayscale

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Input file
        grid.addWidget(QLabel("Input image:"), 0, 0)
        self.input_edit = QLineEdit()
        grid.addWidget(self.input_edit, 0, 1, 1, 3)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self._browse_input)
        grid.addWidget(btn_in, 0, 4)

        # Quality
        grid.addWidget(QLabel("JPEG Quality (1-100):"), 1, 0)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(90)
        grid.addWidget(self.quality_spin, 1, 1)

        # Output file
        grid.addWidget(QLabel("Output filename (.jpg):"), 1, 2)
        self.output_edit = QLineEdit("output.jpg")
        grid.addWidget(self.output_edit, 1, 3)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_output)
        grid.addWidget(btn_out, 1, 4)

        # Buttons
        self.run_btn = QPushButton("Compress")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 2, 0)

        self.preview_btn = QPushButton("Preview Input")
        self.preview_btn.clicked.connect(self._show_preview)
        grid.addWidget(self.preview_btn, 2, 1)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(self._clear_log)
        grid.addWidget(self.clear_btn, 2, 2)

        # Image preview area (left) and log (right)
        # Left: preview label
        self.preview_label = QLabel("Preview will appear here")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFixedSize(480, 360)
        grid.addWidget(self.preview_label, 3, 0, 4, 3)

        # Right: log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 3, 3, 4, 2)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _clear_log(self):
        self.log.clear()

    def _browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select input image", "", 
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;All Files (*)")
        if fn:
            self.input_edit.setText(fn)
            self._load_image(fn)

    def _browse_output(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Select output JPEG", "", 
            "JPEG (*.jpg *.jpeg);;All Files (*)")
        if fn:
            # ensure .jpg extension for clarity
            if not fn.lower().endswith((".jpg", ".jpeg")):
                fn = fn + ".jpg"
            self.output_edit.setText(fn)

    def _load_image(self, path):
        try:
            # try tifffile first (handles multi-page/tiff)
            try:
                img = tifffile.imread(path)
                # tifffile returns shape (H,W) or (H,W,C)
                if img is None:
                    raise ValueError("tifffile returned None")
                # convert to uint8 if needed for preview
                if np.issubdtype(img.dtype, np.floating):
                    mn, mx = np.nanmin(img), np.nanmax(img)
                    if mx > mn:
                        img = (255.0 * (img - mn) / (mx - mn)).astype(np.uint8)
                    else:
                        img = np.zeros_like(img, dtype=np.uint8)
                else:
                    img = img.astype(np.uint8)
            except Exception:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"Could not read image: {path}")

            self._loaded_image = img  # BGR or grayscale or (H,W,C)
            self._show_preview()
            st = self._image_stats(img)
            self._log(f"Loaded: {os.path.basename(path)} shape={img.shape} dtype={img.dtype} stats: {st}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"Failed to load image: {e}\n\n{tb}")
            self._log("Load error:", e)

    def _image_stats(self, img):
        try:
            a = np.asarray(img)
            a = a[~np.isnan(a)]
            mn = int(a.min()) if a.size else 0
            mx = int(a.max()) if a.size else 0
            return f"min={mn} max={mx}"
        except Exception:
            return "stats unavailable"

    def _show_preview(self):
        if self._loaded_image is None:
            path = self.input_edit.text().strip()
            if not path:
                QMessageBox.information(self, "No input", "Select an input image first")
                return
            self._load_image(path)
            if self._loaded_image is None:
                return

        img = self._loaded_image
        # Convert to QImage for display; handle grayscale and color
        try:
            if img.ndim == 2:
                h, w = img.shape
                qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
            else:
                # OpenCV uses BGR; convert to RGB for Qt
                if img.shape[2] == 3:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.shape[2] == 4:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                else:
                    rgb = img
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888 if ch == 3 else QImage.Format.Format_RGBA8888)
            pix = QPixmap.fromImage(qimg).scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.preview_label.setPixmap(pix)
        except Exception as e:
            self.preview_label.setText("Preview error")
            self._log("Preview error:", e)

    def _on_run(self):
        in_path = self.input_edit.text().strip()
        out_path = self.output_edit.text().strip()
        quality = int(self.quality_spin.value())

        if not in_path:
            QMessageBox.warning(self, "Missing input", "Please select an input image")
            return
        if not out_path:
            QMessageBox.warning(self, "Missing output", "Please specify an output filename")
            return
        try:
            # Read image with OpenCV (preserving channels)
            img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                # fallback to tifffile read and convert to uint8
                img = tifffile.imread(in_path)
                if img is None:
                    raise FileNotFoundError("Could not read input image")

            # If single-channel float, scale to 8-bit
            if np.issubdtype(img.dtype, np.floating):
                mn = np.nanmin(img)
                mx = np.nanmax(img)
                if mx > mn:
                    img8 = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)
                else:
                    img8 = np.zeros(img.shape, dtype=np.uint8)
                img = img8

            # If image has alpha channel, drop it for JPEG (JPEG doesn't support alpha)
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Convert image to BGR if it's RGB; OpenCV wrote image as BGR by default when reading
            # Save using OpenCV JPEG quality flag
            ok = cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                raise IOError("cv2.imwrite failed to write the JPEG file")

            out_size = os.path.getsize(out_path)
            self._log(f"Compressed saved: {out_path}  quality={quality} size={out_size} bytes")
            QMessageBox.information(self, "Done", f"Saved {out_path}\nQuality: {quality}\nSize: {out_size} bytes")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Compression error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = JpgCompWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()