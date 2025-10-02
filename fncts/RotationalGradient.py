#!/usr/bin/env python3
"""
RotationalGradient.py
PyQt6 GUI to create 16-bit PNGs from image crops or save processed result as FITS (float32).
Implements the symmetric subtraction and rescale in the original PNGcreateimage16().
"""
import sys
import os
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

def ndarray_to_qpix(arr):
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 2:
        # scale for display to 0..255
        mn, mx = np.nanmin(a), np.nanmax(a)
        if mx > mn:
            a8 = ((a - mn) / (mx - mn) * 255.0).astype(np.uint8)
        else:
            a8 = np.zeros_like(a, dtype=np.uint8)
        h, w = a8.shape
        qimg = QImage(a8.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    else:
        # convert BGR->RGB if needed
        if a.ndim == 3 and a.shape[2] == 3:
            rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        else:
            rgb = a
        if rgb.dtype != np.uint8:
            mn, mx = np.nanmin(rgb), np.nanmax(rgb)
            if mx > mn:
                rgb8 = ((rgb - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                rgb8 = np.zeros_like(rgb, dtype=np.uint8)
        else:
            rgb8 = rgb
        h, w, ch = rgb8.shape
        bytes_per_line = ch * w
        fmt = QImage.Format.Format_RGB888 if ch == 3 else QImage.Format.Format_RGBA8888
        qimg = QImage(rgb8.data, w, h, bytes_per_line, fmt)
        return QPixmap.fromImage(qimg)

def compute_symmetry_crop(arr2d):
    """
    Given a 2D numpy array arr2d (crop), compute symmetric-subtraction result:
      out[x,y] = arr[x,y] - min(arr[x,y], arr[x, D-y], arr[D-x,y], arr[D-x,D-y])
    Expect arr2d to be square with odd size (2*radius+1).
    Returns float64 array with same shape (can be negative before rescale).
    """
    a = np.asarray(arr2d, dtype=np.float64)
    H, W = a.shape
    # ensure square
    if H != W:
        # center-trim to min dimension
        m = min(H, W)
        sy = (H - m) // 2
        sx = (W - m) // 2
        a = a[sy:sy+m, sx:sx+m]
        H = W = m
    D = H - 1
    out = np.empty_like(a)
    for x in range(H):
        for y in range(W):
            s0 = a[x, y]
            s1 = a[x, D - y]
            s2 = a[D - x, y]
            s3 = a[D - x, D - y]
            out[x, y] = s0 - min(s0, s1, s2, s3)
    return out

class PNG16Gui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Create 16-bit PNG or FITS from crop")
        self._build_ui()
        self.resize(1000, 560)
        self.loaded_image = None  # original array (2D or 3D)
        self.preview_crop = None
        self.processed = None

    def _build_ui(self):
        grid = QGridLayout(self)

        grid.addWidget(QLabel("Input image (PNG/TIFF/JPG or FITS):"), 0, 0)
        self.input_le = QLineEdit()
        grid.addWidget(self.input_le, 0, 1, 1, 3)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self._browse_file)
        grid.addWidget(btn_in, 0, 4)

        grid.addWidget(QLabel("Radius (pixels):"), 1, 0)
        self.radius_le = QLineEdit("60")
        grid.addWidget(self.radius_le, 1, 1)

        grid.addWidget(QLabel("Centroid X:"), 1, 2)
        self.cx_le = QLineEdit("60")
        grid.addWidget(self.cx_le, 1, 3)

        grid.addWidget(QLabel("Centroid Y:"), 1, 4)
        self.cy_le = QLineEdit("60")
        grid.addWidget(self.cy_le, 1, 5)

        self.neigh_chk = QCheckBox("Process 3x3 neighborhood around centroid (like original loop)")
        self.neigh_chk.setChecked(False)
        grid.addWidget(self.neigh_chk, 2, 0, 1, 3)

        grid.addWidget(QLabel("Output format:"), 2, 3)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["16-bit PNG (uint16)", "FITS float32"])
        grid.addWidget(self.format_combo, 2, 4)

        grid.addWidget(QLabel("Output filename:"), 3, 0)
        self.out_le = QLineEdit("")
        grid.addWidget(self.out_le, 3, 1, 1, 3)
        btn_out = QPushButton("Browse Save")
        btn_out.clicked.connect(self._browse_save)
        grid.addWidget(btn_out, 3, 4)

        # Buttons
        btn_load = QPushButton("Load & Preview Crop")
        btn_load.clicked.connect(self._load_preview)
        grid.addWidget(btn_load, 4, 0)

        btn_process = QPushButton("Process and Save")
        btn_process.clicked.connect(self._process_and_save)
        grid.addWidget(btn_process, 4, 1)

        btn_preview_proc = QPushButton("Show Processed Preview")
        btn_preview_proc.clicked.connect(self._show_processed_preview)
        grid.addWidget(btn_preview_proc, 4, 2)

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(lambda: self.log.clear())
        grid.addWidget(clear_btn, 4, 3)

        # Previews
        self.crop_label = QLabel("Crop preview")
        self.crop_label.setFixedSize(320, 320)
        self.crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.crop_label, 5, 0, 3, 1)

        self.proc_label = QLabel("Processed preview")
        self.proc_label.setFixedSize(320, 320)
        self.proc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.proc_label, 5, 1, 3, 1)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 5, 2, 3, 4)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_file(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open image or FITS", "", "Images (*.png *.tif *.tiff *.jpg *.jpeg);;FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.input_le.setText(fn)

    def _browse_save(self):
        fmt = self.format_combo.currentText()
        default = "output.fits" if "FITS" in fmt else "output.png"
        fn, _ = QFileDialog.getSaveFileName(self, "Save output", default, "FITS (*.fits);;PNG (*.png);;All Files (*)")
        if fn:
            self.out_le.setText(fn)

    def _read_input(self, path):
        p = Path(path)
        if p.suffix.lower() in (".fit", ".fits"):
            with fits.open(path, memmap=False) as hd:
                data = hd[0].data
                # convert to 2D if multi-plane; prefer first plane if 3D
                if data is None:
                    raise IOError("No data in FITS primary HDU")
                arr = np.array(data, dtype=np.float64)
                if arr.ndim == 3:
                    # try (C,Y,X) => take first channel; else (Z,Y,X) take first
                    if arr.shape[0] in (3,4) or arr.shape[0] <= 10:
                        arr2 = arr[0].astype(np.float64)
                    else:
                        arr2 = arr[0].astype(np.float64)
                elif arr.ndim == 2:
                    arr2 = arr
                else:
                    raise ValueError("Unsupported FITS data ndim: " + str(arr.ndim))
                return arr2
        else:
            img = Image.open(path)
            arr = np.asarray(img)
            # convert to grayscale if color
            if arr.ndim == 3:
                # RGB or RGBA -> grayscale using luminance
                if arr.shape[2] >= 3:
                    r, g, b = arr[...,0], arr[...,1], arr[...,2]
                    gray = 0.299*r + 0.587*g + 0.114*b
                else:
                    gray = arr[...,0]
            else:
                gray = arr
            return np.asarray(gray, dtype=np.float64)

    def _crop_at(self, arr, cx, cy, radius):
        cx = int(round(cx)); cy = int(round(cy)); r = int(round(radius))
        one = cx - r
        two = cy - r
        three = cx + r + 1
        four = cy + r + 1
        # clamp
        H, W = arr.shape
        one = max(0, one); two = max(0, two)
        three = min(W, three); four = min(H, four)
        crop = arr[two:four, one:three]  # note y first
        return crop

    def _load_preview(self):
        try:
            path = self.input_le.text().strip()
            if not path:
                QMessageBox = getattr(__import__("PyQt6.QtWidgets"), "QMessageBox")
                QMessageBox.warning(self, "Input required", "Select an input file first")
                return
            arr = self._read_input(path)
            self.loaded_image = arr
            radius = float(self.radius_le.text().strip())
            cx = float(self.cx_le.text().strip())
            cy = float(self.cy_le.text().strip())
            crop = self._crop_at(arr, cx, cy, radius)
            if crop.size == 0:
                raise ValueError("Crop is empty; check centroid/radius")
            self.preview_crop = crop
            pix = ndarray_to_qpix(crop)
            if pix:
                self.crop_label.setPixmap(pix.scaled(self.crop_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._log(f"Loaded input {Path(path).name}, crop shape {crop.shape}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Load preview error:", e)
            self._log(tb)

    def _process_single(self, crop):
        # compute symmetric subtraction and rescale to positive
        proc = compute_symmetry_crop(crop)
        # shift to positive and scale to chosen output later
        return proc

    def _process_and_save(self):
        try:
            if self.loaded_image is None:
                self._load_preview()
                if self.loaded_image is None:
                    raise RuntimeError("No image loaded")
            radius = float(self.radius_le.text().strip())
            cx = int(round(float(self.cx_le.text().strip())))
            cy = int(round(float(self.cy_le.text().strip())))
            outpath = self.out_le.text().strip()
            if not outpath:
                raise ValueError("Choose an output filename")

            do_neigh = bool(self.neigh_chk.isChecked())
            mode = self.format_combo.currentText()

            processed_accum = None
            count = 0

            coords = [(cx, cy)]
            if do_neigh:
                # 3x3 neighborhood centered on cx,cy (offsets -1..1)
                coords = [(cx + dx, cy + dy) for dx in (-1,0,1) for dy in (-1,0,1)]

            for (cx0, cy0) in coords:
                crop = self._crop_at(self.loaded_image, cx0, cy0, radius)
                if crop.size == 0:
                    self._log(f"Skipping empty crop at {cx0},{cy0}")
                    continue
                proc = self._process_single(crop)
                if processed_accum is None:
                    processed_accum = np.zeros_like(proc, dtype=np.float64)
                # If shapes differ because of edge clipping, pad to common size (center)
                if proc.shape != processed_accum.shape:
                    # center-pad smaller to larger
                    H = max(proc.shape[0], processed_accum.shape[0])
                    W = max(proc.shape[1], processed_accum.shape[1])
                    new_acc = np.zeros((H,W), dtype=np.float64)
                    new_acc[:processed_accum.shape[0], :processed_accum.shape[1]] = processed_accum
                    tmp = np.zeros((H,W), dtype=np.float64)
                    tmp[:proc.shape[0], :proc.shape[1]] = proc
                    processed_accum = new_acc + tmp
                else:
                    processed_accum += proc
                count += 1

            if processed_accum is None or count == 0:
                raise RuntimeError("No valid crops processed")

            # average if multiple
            final = processed_accum / float(count)

            # shift to positive and optionally scale
            valid = np.isfinite(final)
            if not np.any(valid):
                raise RuntimeError("Processed image has no finite pixels")
            mn = float(np.nanmin(final[valid]))
            mx = float(np.nanmax(final[valid]))
            if mx <= mn:
                scaled_png = np.zeros_like(final, dtype=np.uint16)
            else:
                # scale to full uint16 range
                scaled = (final - mn) / (mx - mn)
                scaled_png = np.clip((scaled * 65535.0), 0, 65535).astype(np.uint16)

            # Save as requested
            if "PNG" in mode:
                # write 16-bit PNG
                im = Image.fromarray(scaled_png)
                im.save(outpath)
                self._log(f"Wrote 16-bit PNG: {outpath} (shape {scaled_png.shape})")
            else:
                # save as FITS float32 of the computed final (not scaled to uint16)
                hdu = fits.PrimaryHDU(final.astype(np.float32))
                hdu.header['PROCMD'] = "symmetry-subtract averaged"
                hdu.writeto(outpath, overwrite=True)
                self._log(f"Wrote FITS float32: {outpath} (shape {final.shape})")

            # update processed preview
            self.processed = final
            pix = ndarray_to_qpix(scaled_png)
            if pix:
                self.proc_label.setPixmap(pix.scaled(self.proc_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

            self._log("Processing complete.")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Process/save error:", e)
            self._log(tb)

    def _show_processed_preview(self):
        try:
            if self.processed is None:
                QMessageBox = getattr(__import__("PyQt6.QtWidgets"), "QMessageBox")
                QMessageBox.information(self, "No processed", "Run Process and Save first to generate preview")
                return
            valid = np.isfinite(self.processed)
            mn = float(np.nanmin(self.processed[valid]))
            mx = float(np.nanmax(self.processed[valid]))
            if mx > mn:
                disp = (self.processed - mn) / (mx - mn)
                disp8 = (disp * 255.0).astype(np.uint8)
            else:
                disp8 = np.zeros_like(self.processed, dtype=np.uint8)
            pix = ndarray_to_qpix(disp8)
            if pix:
                self.proc_label.setPixmap(pix.scaled(self.proc_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._log("Displayed processed preview.")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Preview error:", e)
            self._log(tb)

def main():
    app = QApplication(sys.argv)
    w = PNG16Gui()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()