#!/usr/bin/env python3
"""
clahe_gui.py

PyQt6 GUI to apply CLAHE to color images or 3-plane FITS (RGB) and save result.

Change:
 - FITS output is now 32-bit float (float32) channel-first (C, Y, X), scaled 0..1 so Siril can read it.
"""
import sys
import os
import traceback
import numpy as np
import cv2
import tifffile
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QMessageBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

# ---------- helpers ----------
def read_fits_image(path):
    with fits.open(path) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header
    return data, hdr

def write_fits_image(path, data, header=None):
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(path, overwrite=True)

def to_preview_qpix(img):
    # img: HxW or HxWxC (RGB)
    if img is None:
        return None
    if img.ndim == 2:
        h, w = img.shape
        arr8 = np.clip(img, 0, 255).astype(np.uint8)
        qimg = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        rgb = img
        if rgb.shape[2] == 3:
            rgb8 = np.clip(rgb, 0, 255).astype(np.uint8)
            h, w, ch = rgb8.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        elif rgb.shape[2] == 4:
            rgba8 = np.clip(rgb, 0, 255).astype(np.uint8)
            h, w, ch = rgba8.shape
            bytes_per_line = ch * w
            qimg = QImage(rgba8.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        else:
            # fallback: take first three channels
            rgb8 = np.clip(rgb[..., :3], 0, 255).astype(np.uint8)
            h, w, ch = rgb8.shape
            qimg = QImage(rgb8.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

def normalize_to_uint16(arr):
    a = np.array(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a, dtype=np.uint16)
    mn = np.nanmin(a[finite])
    mx = np.nanmax(a[finite])
    if mx == mn:
        return np.zeros_like(a, dtype=np.uint16)
    scaled = (a - mn) / (mx - mn)
    out = (scaled * 65535.0).astype(np.uint16)
    return out

# ---------- GUI ----------
class ClaheWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CLAHE (color image / FITS)")
        self._build_ui()
        self.resize(900, 520)
        self.loaded = None     # numpy array for preview (H,W or H,W,C) in uint8/uint16
        self.header = None     # for FITS
        self.input_path = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # File type radio
        grid.addWidget(QLabel("Input type:"), 0, 0)
        self.fits_radio = QRadioButton("FITS (3-plane RGB preferred)")
        self.img_radio = QRadioButton("Image (png/tif/jpg)")
        self.fits_radio.setChecked(True)
        rg = QButtonGroup(self)
        rg.addButton(self.fits_radio); rg.addButton(self.img_radio)
        hbox = QHBoxLayout(); hbox.addWidget(self.fits_radio); hbox.addWidget(self.img_radio)
        grid.addLayout(hbox, 0, 1, 1, 4)

        # Input file
        grid.addWidget(QLabel("Input file:"), 1, 0)
        self.input_edit = QLineEdit()
        grid.addWidget(self.input_edit, 1, 1, 1, 3)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self._browse_input)
        grid.addWidget(btn_in, 1, 4)

        # Parameters
        grid.addWidget(QLabel("Clip limit (float):"), 2, 0)
        self.clip_spin = QDoubleSpinBox(); self.clip_spin.setDecimals(3); self.clip_spin.setRange(0.1, 100.0); self.clip_spin.setValue(3.0)
        grid.addWidget(self.clip_spin, 2, 1)

        grid.addWidget(QLabel("Tile grid size (int):"), 2, 2)
        self.tile_spin = QSpinBox(); self.tile_spin.setRange(1, 64); self.tile_spin.setValue(8)
        grid.addWidget(self.tile_spin, 2, 3)

        # Output file
        grid.addWidget(QLabel("Output filename:"), 3, 0)
        self.output_edit = QLineEdit("clahe_output.fits")
        grid.addWidget(self.output_edit, 3, 1, 1, 3)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_save)
        grid.addWidget(btn_out, 3, 4)

        # Buttons
        self.preview_btn = QPushButton("Load & Preview")
        self.preview_btn.clicked.connect(self._load_and_preview)
        grid.addWidget(self.preview_btn, 4, 0)

        self.run_btn = QPushButton("Apply CLAHE & Save")
        self.run_btn.clicked.connect(self._apply_and_save)
        grid.addWidget(self.run_btn, 4, 1)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(lambda: self.log.clear())
        grid.addWidget(self.clear_btn, 4, 2)

        # Preview label and log
        self.preview_label = QLabel("Preview")
        self.preview_label.setFixedSize(560, 360)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.preview_label, 5, 0, 6, 3)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 5, 3, 6, 2)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_input(self):
        if self.fits_radio.isChecked():
            fn, _ = QFileDialog.getOpenFileName(self, "Open FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        else:
            fn, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if fn:
            self.input_edit.setText(fn)

    def _browse_save(self):
        if self.fits_radio.isChecked():
            fn, _ = QFileDialog.getSaveFileName(self, "Save FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        else:
            fn, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def _load_and_preview(self):
        path = self.input_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Input required", "Select an existing input file first")
            return
        try:
            self.input_path = path
            if self.fits_radio.isChecked():
                data, hdr = read_fits_image(path)
                self.header = hdr
                if data is None:
                    raise ValueError("No data in FITS primary HDU")
                # expect shape (3,H,W) or (H,W,3). Try common forms -> convert to H,W,3
                arr = np.array(data)
                if arr.ndim == 3:
                    if arr.shape[0] == 3:
                        arr = np.transpose(arr, (1,2,0))
                elif arr.ndim == 2:
                    # single plane grayscale -> expand to 3 channels by duplicating
                    arr = np.stack([arr]*3, axis=-1)
                else:
                    raise ValueError(f"Unsupported FITS ndim {arr.ndim}")
                # normalize to uint8 for preview
                if arr.dtype.kind == 'f' or arr.dtype.itemsize > 1:
                    arr8 = np.nan_to_num(arr)
                    arr8 = arr8 - np.nanmin(arr8); mx = np.nanmax(arr8)
                    if mx > 0:
                        arr8 = (arr8 / mx * 255.0).astype(np.uint8)
                    else:
                        arr8 = np.zeros_like(arr8, dtype=np.uint8)
                else:
                    arr8 = arr.astype(np.uint8)
                self.loaded = arr8
                self._show_preview(arr8)
                self._log(f"Loaded FITS {os.path.basename(path)} shape={arr.shape} dtype={data.dtype}")
            else:
                # image file
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError("Could not read image")
                # convert BGR->RGB for preview
                if img.ndim == 3 and img.shape[2] >= 3:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.ndim == 2:
                    rgb = img
                else:
                    rgb = img
                # scale floats to 0..255 for preview
                if np.issubdtype(rgb.dtype, np.floating):
                    mn = np.nanmin(rgb); mx = np.nanmax(rgb)
                    if mx > mn:
                        rgb = ((rgb - mn) / (mx - mn) * 255.0).astype(np.uint8)
                    else:
                        rgb = np.zeros_like(rgb, dtype=np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
                self.loaded = rgb
                self.header = None
                self._show_preview(rgb)
                self._log(f"Loaded image {os.path.basename(path)} shape={rgb.shape} dtype={rgb.dtype}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"{e}\n\n{tb}")

    def _show_preview(self, arr):
        try:
            # downscale for preview if large
            h, w = arr.shape[:2]
            maxw, maxh = self.preview_label.width(), self.preview_label.height()
            scale = min(1.0, maxw / w, maxh / h)
            if scale < 1.0:
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                disp = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                disp = arr
            pix = to_preview_qpix(disp)
            if pix is not None:
                self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            self._log("Preview error:", e)

    def _apply_and_save(self):
        if self.loaded is None:
            QMessageBox.warning(self, "No input", "Load an input first and preview")
            return
        outpath = self.output_edit.text().strip()
        if not outpath:
            QMessageBox.warning(self, "No output", "Choose an output filename")
            return
        clip = float(self.clip_spin.value())
        tile = int(self.tile_spin.value())
        try:
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            if self.fits_radio.isChecked():
                # read raw data again to preserve full precision and header
                raw, hdr = read_fits_image(self.input_path)
                arr = np.array(raw)
                # Normalize and ensure shape H,W,3
                if arr.ndim == 3 and arr.shape[0] == 3:
                    arr = np.transpose(arr, (1,2,0))
                elif arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                elif arr.ndim == 3 and arr.shape[2] == 3:
                    pass
                else:
                    raise ValueError("Unsupported FITS shape for CLAHE; expected (3,H,W) or (H,W,3) or (H,W)")

                # Map original data to uint16 scale for CLAHE processing, but remember to produce float32 output.
                # We'll normalize the CLAHE result to 0..1 (float32) for Siril compatibility.
                u16 = normalize_to_uint16(arr)  # H,W,C uint16 in 0..65535

                # Ensure per-channel processing
                if u16.ndim == 3:
                    channels = [u16[..., c].astype(np.uint16) for c in range(u16.shape[2])]
                else:
                    channels = [u16.astype(np.uint16)]

                clahe_channels = []
                for ch in channels:
                    # apply CLAHE on 8-bit by downscaling to 8-bit, then scale back to 0..65535 range
                    ch8 = (ch / 256).astype(np.uint8)
                    c_out8 = clahe.apply(ch8)
                    c_back = (c_out8.astype(np.uint16) * 257)  # use 257 to map 0..255 -> 0..65535 (approximately)
                    clahe_channels.append(c_back)

                merged = np.stack(clahe_channels, axis=-1)  # H,W,3 uint16 in 0..65535

                # Convert merged uint16 to float32 normalized to 0..1 for Siril
                merged_float = merged.astype(np.float32) / 65535.0  # H,W,3 float32 in 0..1

                # Convert to channel-first (C, Y, X) which many FITS pipelines (and Siril) accept
                out_arr = np.transpose(merged_float, (2, 0, 1)).astype(np.float32)  # (C, Y, X)

                # Ensure header is appropriate: set BITPIX to -32 for float32
                if hdr is None:
                    hdr = fits.Header()
                # write FITS as float32 primary
                write_fits_image(outpath, out_arr, header=hdr)
                self._log(f"Wrote FITS (CLAHE float32 0..1) to {outpath} shape={out_arr.shape} dtype={out_arr.dtype}")
            else:
                # image file path: loaded is already uint8/uint16 RGB or gray in self.loaded
                img = self.loaded
                # ensure color channels present
                if img.ndim == 2:
                    # single channel: apply CLAHE directly
                    img_out = clahe.apply(img.astype(np.uint8))
                else:
                    # process per-channel; loaded is RGB currently
                    if img.shape[2] >= 3:
                        # split into channels (RGB)
                        if img.dtype != np.uint8:
                            work = img.astype(np.uint8)
                        else:
                            work = img
                        r = clahe.apply(work[:, :, 0])
                        g = clahe.apply(work[:, :, 1])
                        b = clahe.apply(work[:, :, 2])
                        img_out = np.stack((r, g, b), axis=2)
                        # convert RGB back to BGR for cv2.imwrite
                        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
                    else:
                        # other channel counts: apply to each channel
                        chans = []
                        for c in range(img.shape[2]):
                            chans.append(clahe.apply(img[:,:,c].astype(np.uint8)))
                        img_out = np.stack(chans, axis=2)
                # write output (use tifffile for better numeric preservation)
                ext = os.path.splitext(outpath)[1].lower()
                if ext in (".tif", ".tiff"):
                    tifffile.imwrite(outpath, img_out)
                else:
                    ok = cv2.imwrite(outpath, img_out)
                    if not ok:
                        raise IOError("Failed to write image with cv2.imwrite")
                self._log(f"Wrote image (CLAHE) to {outpath} dtype={img_out.dtype} shape={getattr(img_out,'shape',None)}")
            QMessageBox.information(self, "Done", f"Saved CLAHE result to:\n{outpath}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = ClaheWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()