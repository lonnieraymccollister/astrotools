#!/usr/bin/env python3
"""
resize_gui.py
PyQt6 GUI for resizing images with three modes:
- FITS cubic (float64) using scipy.ndimage.zoom (order=3)
- FITS Lanczos4 using OpenCV (channels handled)
- Other image files (tiff/png/...) using OpenCV (LANCZOS4)
"""
import sys
import os
import traceback
import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom
import cv2
import tifffile

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt

class ResizeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resize Tool")
        self._build_ui()
        self.resize(820, 320)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        grid.addWidget(QLabel("Mode:"), 0, 0)
        self.mode = QComboBox()
        self.mode.addItems([
            "FITS cubic float64 (order=3)",
            "FITS Lanczos4 (OpenCV, keep channels)",
            "Other image Lanczos4 (tif/png/jpg)"
        ])
        grid.addWidget(self.mode, 0, 1, 1, 3)

        grid.addWidget(QLabel("Input file:"), 1, 0)
        self.input_edit = QLineEdit()
        grid.addWidget(self.input_edit, 1, 1, 1, 2)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self._browse_input)
        grid.addWidget(btn_in, 1, 3)

        grid.addWidget(QLabel("Scale numerator (int):"), 2, 0)
        self.num_spin = QSpinBox(); self.num_spin.setRange(1, 64); self.num_spin.setValue(1)
        grid.addWidget(self.num_spin, 2, 1)
        grid.addWidget(QLabel("Scale denominator (int):"), 2, 2)
        self.den_spin = QSpinBox(); self.den_spin.setRange(1, 64); self.den_spin.setValue(1)
        grid.addWidget(self.den_spin, 2, 3)

        grid.addWidget(QLabel("Output file:"), 3, 0)
        self.output_edit = QLineEdit("resized_output.fits")
        grid.addWidget(self.output_edit, 3, 1, 1, 2)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_output)
        grid.addWidget(btn_out, 3, 3)

        self.run_btn = QPushButton("Run Resize")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 4, 0)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(self._clear_log)
        grid.addWidget(self.clear_btn, 4, 1)

        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self._show_help)
        grid.addWidget(self.help_btn, 4, 2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 5, 0, 1, 4)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _clear_log(self):
        self.log.clear()

    def _show_help(self):
        QMessageBox.information(self, "Help",
            "Modes:\n"
            "1) FITS cubic float64: uses scipy.ndimage.zoom(order=3) on a 2D FITS image (keeps float64 output).\n"
            "2) FITS Lanczos4: uses OpenCV LANCZOS4 on multi-plane FITS expected shape (C,Y,X) or (Y,X,C).\n"
            "3) Other image: reads with tifffile or OpenCV and resizes with OpenCV LANCZOS4.\n\n"
            "Scale = numerator / denominator. Example: 2/1 doubles size, 1/2 halves size."
        )

    def _browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select input file", "", "All Files (*)")
        if fn:
            self.input_edit.setText(fn)

    def _browse_output(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Select output file", "", "All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def _on_run(self):
        in_path = self.input_edit.text().strip()
        out_path = self.output_edit.text().strip()
        num = int(self.num_spin.value())
        den = int(self.den_spin.value())
        if not in_path:
            QMessageBox.warning(self, "Input required", "Select an input file.")
            return
        if not out_path:
            QMessageBox.warning(self, "Output required", "Specify an output filename.")
            return
        scale = float(num) / float(den)
        mode_idx = self.mode.currentIndex()

        try:
            if mode_idx == 0:
                # FITS cubic float64
                self._log("Mode: FITS cubic float64 (order=3)")
                self._resize_fits_cubic(in_path, out_path, scale)
            elif mode_idx == 1:
                # FITS Lanczos4 with OpenCV
                self._log("Mode: FITS Lanczos4 (OpenCV)")
                self._resize_fits_lanczos(in_path, out_path, scale)
            else:
                # Other image via tifffile/OpenCV
                self._log("Mode: Other image Lanczos4")
                self._resize_other_image(in_path, out_path, scale)
            QMessageBox.information(self, "Done", f"Saved resized file: {out_path}")
            self._log("Done:", out_path)
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

    def _resize_fits_cubic(self, in_path, out_path, scale):
        # Load FITS, expect 2D; perform scipy.ndimage.zoom(order=3)
        with fits.open(in_path) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
        if data is None:
            raise ValueError("No data found in FITS primary HDU.")
        if data.ndim != 2:
            raise ValueError("FITS cubic mode expects a 2D image in primary HDU.")
        self._log(f"Original shape: {data.shape}, dtype={data.dtype}")
        # ensure float64
        arr = data.astype(np.float64)
        zoom_factors = (scale, scale)
        self._log(f"Applying cubic zoom factors: {zoom_factors}")
        resized = zoom(arr, zoom_factors, order=3)
        resized = resized.astype(np.float64)
        hdu = fits.PrimaryHDU(resized, header=hdr)
        hdu.writeto(out_path, overwrite=True)
        self._log(f"Wrote FITS (float64) shape {resized.shape}")

    def _resize_fits_lanczos(self, in_path, out_path, scale):
        # Load FITS and handle shapes: could be (C,Y,X), (Y,X,C), or (Y,X)
        with fits.open(in_path) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
        if data is None:
            raise ValueError("No data found in FITS primary HDU.")
        self._log(f"Input FITS shape: {data.shape}, dtype={data.dtype}")
        # Normalize to 0..65535 for OpenCV processing if needed, then convert back
        if data.ndim == 3:
            # guess channel-first if first dim small
            if data.shape[0] <= 4:
                arr = np.transpose(data, (1,2,0))  # (C,Y,X)->(Y,X,C)
            else:
                arr = data  # assume (Y,X,C)
        elif data.ndim == 2:
            arr = data[..., None]  # make 3D (Y,X,1)
        else:
            raise ValueError("Unsupported FITS data ndim for Lanczos mode.")
        # scale to float32 range for OpenCV
        finite = np.isfinite(arr)
        if not finite.any():
            raise ValueError("FITS contains no finite pixels.")
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        if arr_max == arr_min:
            norm = np.zeros_like(arr, dtype=np.float32)
        else:
            norm = ((arr - arr_min) / (arr_max - arr_min) * 65535.0).astype(np.float32)
        # OpenCV expects channel-last 8/16/32-bit; we'll use float32 and treat channels
        h, w = norm.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        self._log(f"Resizing from ({h},{w}) to ({new_h},{new_w}) using LANCZOS4")
        # OpenCV resize expects HxW or HxWxC
        resized = cv2.resize(norm, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        # convert back to original numeric range
        resized_back = resized.astype(np.float64) / 65535.0 * (arr_max - arr_min) + arr_min
        # if original was 2D, squeeze channel
        if data.ndim == 2:
            resized_back = resized_back[...,0]
        else:
            # convert to channel-first if original was channel-first
            if data.ndim == 3 and data.shape[0] <= 4:
                resized_back = np.transpose(resized_back, (2,0,1))  # (Y,X,C)->(C,Y,X)
        hdu = fits.PrimaryHDU(resized_back, header=hdr)
        hdu.writeto(out_path, overwrite=True)
        self._log(f"Wrote FITS (Lanczos) shape {resized_back.shape}")

    def _resize_other_image(self, in_path, out_path, scale):
        # Try tifffile first for multi-page/tiff; otherwise cv2.imread
        try:
            img = tifffile.imread(in_path)
            self._log("Read image with tifffile")
        except Exception:
            img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {in_path}")
            self._log("Read image with OpenCV")
        self._log(f"Original image shape: {img.shape}, dtype={img.dtype}")
        h = img.shape[0]; w = img.shape[1]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        self._log(f"Resizing to ({new_h},{new_w}) with LANCZOS4")
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        # Save using tifffile for multi-channel or float; fallback to cv2.imwrite for common formats
        try:
            tifffile.imwrite(out_path, resized)
            self._log("Saved with tifffile")
        except Exception:
            ok = cv2.imwrite(out_path, resized)
            if not ok:
                raise IOError("Failed to write output image with cv2.imwrite")
            self._log("Saved with OpenCV imwrite")

def main():
    app = QApplication(sys.argv)
    w = ResizeWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()