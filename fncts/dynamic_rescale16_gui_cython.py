#!/usr/bin/env python3
"""
dynamic_rescale16_gui_cython.py
PyQt6 GUI for DynamicRescale16 that uses a compiled Cython function
warp_affine_mask_rescale from module warpaffinemaskrescale when available,
and falls back to a pure-Python block rescaler when not.

Usage: python dynamic_rescale16_gui_cython.py
"""

import sys
import os
import re
from pathlib import Path

import numpy as np
import cv2
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QGridLayout, QVBoxLayout, QTextEdit, QMessageBox,
    QSpinBox, QHBoxLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator

# -------------------------
# Try importing your compiled Cython extension
# -------------------------
try:
    from warpaffinemaskrescale import warp_affine_mask_rescale
    CYTHON_AVAILABLE = True
    _IMPORT_ERR = None
except Exception as e:
    warp_affine_mask_rescale = None
    CYTHON_AVAILABLE = False
    _IMPORT_ERR = e

# -------------------------
# Pure-Python fallback block rescaler
# -------------------------
def block_rescale_float64(img64, out64, block_size):
    H, W = img64.shape
    bs = max(1, int(block_size))
    for y0 in range(0, H, bs):
        for x0 in range(0, W, bs):
            y1 = min(y0 + bs, H)
            x1 = min(x0 + bs, W)
            block = img64[y0:y1, x0:x1]
            bmin = np.nanmin(block)
            bmax = np.nanmax(block)
            if bmax > bmin:
                norm = (block - bmin) / (bmax - bmin)
            else:
                norm = np.zeros_like(block)
            out64[y0:y1, x0:x1] = norm * 65535.0

# -------------------------
# Wrapper that prefers Cython but falls back
# -------------------------
def rescale_blocks(img64, out64, block_size):
    """
    img64 and out64 must be 2D numpy arrays of dtype float64.
    If the compiled function is available it will be called; otherwise fallback is used.
    """
    if CYTHON_AVAILABLE and callable(warp_affine_mask_rescale):
        # enforce dtype and contiguity
        if img64.dtype != np.float64:
            img64 = img64.astype(np.float64)
        if out64.dtype != np.float64:
            out64 = out64.astype(np.float64)
        if not img64.flags['C_CONTIGUOUS']:
            img64 = np.ascontiguousarray(img64)
        if not out64.flags['C_CONTIGUOUS']:
            out64 = np.ascontiguousarray(out64)
        # call compiled routine (expected to write into out64)
        warp_affine_mask_rescale(img64, out64, int(block_size))
    else:
        # Python fallback
        block_rescale_float64(img64, out64, block_size)

# -------------------------
# FITS helpers and processing logic
# -------------------------
def load_fits(file_path):
    with fits.open(file_path) as hdul:
        return hdul[0].data, hdul[0].header

def split_image(image, tile_size=(600,600), output_dir="tiles"):
    os.makedirs(output_dir, exist_ok=True)
    h, w = image.shape
    tile_h, tile_w = tile_size
    tiles = []
    for i in range(0, h, tile_h):
        for j in range(0, w, tile_w):
            sub_image = image[i:i+tile_h, j:j+tile_w]
            sub_h, sub_w = sub_image.shape
            padded_tile = np.zeros(tile_size, dtype=image.dtype)
            padded_tile[:sub_h, :sub_w] = sub_image
            tile_file = os.path.join(output_dir, f"tile_{i}_{j}_{sub_h}_{sub_w}.fits")
            fits.writeto(tile_file, padded_tile, overwrite=True)
            tiles.append(tile_file)
    return tiles

def reassemble_image(tiles, original_shape):
    final_image = np.zeros(original_shape, dtype=np.float64)
    pattern = r"tile_(\d+)_(\d+)_(\d+)_(\d+)"
    for tile_file in tiles:
        base = os.path.basename(tile_file)
        match = re.search(pattern, base)
        if not match:
            continue
        i_str, j_str, sub_h_str, sub_w_str = match.groups()
        i, j, sub_h, sub_w = map(int, [i_str, j_str, sub_h_str, sub_w_str])
        with fits.open(tile_file) as hdul:
            data = hdul[0].data
            final_image[i:i+sub_h, j:j+sub_w] = data[:sub_h, :sub_w]
    return final_image

def process_tile(tile_file, width_of_square, bin_value, gamma_value, resize_factor, resize_div, log_fn=print):
    log_fn(f"Processing tile: {tile_file}")
    with fits.open(tile_file) as hdul:
        header = hdul[0].header
        image_data = hdul[0].data
    if image_data is None:
        log_fn("Tile contains no data, skipping.")
        return
    if image_data.ndim != 2:
        raise ValueError("Tile data must be 2D")

    # normalize to 0..65535
    minv = np.nanmin(image_data)
    maxv = np.nanmax(image_data)
    if maxv == minv:
        norm_image = np.zeros_like(image_data, dtype=np.uint16)
    else:
        norm_image = ((image_data - minv) / (maxv - minv) * 65535.0).astype(np.uint16)

    fx = (resize_factor / resize_div) if resize_div != 0 else 1.0
    if fx <= 0:
        fx = 1.0
    resized = cv2.resize(norm_image, None, fx=fx, fy=fx, interpolation=cv2.INTER_LANCZOS4)

    img64 = resized.astype(np.float64)
    out64 = np.empty_like(img64, dtype=np.float64)

    # perform block rescale via Cython or fallback
    rescale_blocks(img64, out64, block_size=int(width_of_square))

    # gamma correction
    gamma_corrected = np.round(65535.0 * (out64 / 65535.0) ** float(gamma_value)).astype(np.float64)
    img_array = (gamma_corrected / 6553500.0).astype(np.float64)

    # binning via reshape trick
    bin_factor = max(1, int(bin_value))
    h_img, w_img = img_array.shape
    new_height = h_img // bin_factor
    new_width = w_img // bin_factor
    if new_height == 0 or new_width == 0:
        raise ValueError("Bin factor too large for resized image size.")
    trimmed = img_array[:new_height*bin_factor, :new_width*bin_factor]
    binned = trimmed.reshape(new_height, bin_factor, new_width, bin_factor).sum(axis=(1,3))

    out_filename = tile_file + '_binned_gamma_corrected_drs.fits'
    fits.writeto(out_filename, binned.astype(np.float32), header, overwrite=True)
    log_fn(f"Saved: {out_filename}")

# -------------------------
# PyQt6 GUI
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DynamicRescale16 GUI (Cython integrated)")
        self._build_ui()
        self.resize(820, 620)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Mode
        grid.addWidget(QLabel("Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Split Tiles", "Process Tiles", "Reassemble"])
        self.mode_combo.currentIndexChanged.connect(self._update_mode)
        grid.addWidget(self.mode_combo, 0, 1, 1, 3)

        # Split inputs
        grid.addWidget(QLabel("Input FITS (split):"), 1, 0)
        self.split_input = QLineEdit()
        grid.addWidget(self.split_input, 1, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.split_input, mode='open', filter="FITS Files (*.fits *.fit)"))
        grid.addWidget(btn, 1, 3)

        grid.addWidget(QLabel("Tile height:"), 2, 0)
        self.tile_h = QSpinBox(); self.tile_h.setRange(16, 32768); self.tile_h.setValue(600)
        grid.addWidget(self.tile_h, 2, 1)
        grid.addWidget(QLabel("Tile width:"), 2, 2)
        self.tile_w = QSpinBox(); self.tile_w.setRange(16, 32768); self.tile_w.setValue(600)
        grid.addWidget(self.tile_w, 2, 3)

        grid.addWidget(QLabel("Output tiles dir:"), 3, 0)
        self.tiles_dir = QLineEdit("tiles")
        grid.addWidget(self.tiles_dir, 3, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.tiles_dir, mode='dir'))
        grid.addWidget(btn, 3, 3)

        # Process inputs
        grid.addWidget(QLabel("Tiles dir (process):"), 4, 0)
        self.proc_tiles_dir = QLineEdit("tiles")
        grid.addWidget(self.proc_tiles_dir, 4, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.proc_tiles_dir, mode='dir'))
        grid.addWidget(btn, 4, 3)

        grid.addWidget(QLabel("Width of square:"), 5, 0)
        self.width_square = QSpinBox(); self.width_square.setRange(1,1024); self.width_square.setValue(5)
        grid.addWidget(self.width_square, 5, 1)
        grid.addWidget(QLabel("Bin value:"), 5, 2)
        self.bin_value = QSpinBox(); self.bin_value.setRange(1,1024); self.bin_value.setValue(25)
        grid.addWidget(self.bin_value, 5, 3)

        grid.addWidget(QLabel("Gamma:"), 6, 0)
        self.gamma_edit = QLineEdit("0.3981")
        # set validator for gamma
        self.gamma_edit.setValidator(QDoubleValidator(-10.0, 10.0, 6))
        grid.addWidget(self.gamma_edit, 6, 1)
        grid.addWidget(QLabel("Resize factor:"), 6, 2)
        self.resize_factor = QSpinBox(); self.resize_factor.setRange(1,1024); self.resize_factor.setValue(25)
        grid.addWidget(self.resize_factor, 6, 3)
        grid.addWidget(QLabel("Resize div:"), 7, 2)
        self.resize_div = QSpinBox(); self.resize_div.setRange(1,1024); self.resize_div.setValue(1)
        grid.addWidget(self.resize_div, 7, 3)

        # Reassemble inputs
        grid.addWidget(QLabel("Processed tiles dir (reassemble):"), 8, 0)
        self.rea_tiles_dir = QLineEdit("tiles")
        grid.addWidget(self.rea_tiles_dir, 8, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.rea_tiles_dir, mode='dir'))
        grid.addWidget(btn, 8, 3)

        grid.addWidget(QLabel("Original FITS (for shape/header):"), 9, 0)
        self.orig_fits = QLineEdit()
        grid.addWidget(self.orig_fits, 9, 1, 1, 2)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._pick_file(self.orig_fits, mode='open', filter="FITS Files (*.fits *.fit)"))
        grid.addWidget(btn, 9, 3)

        # Run / clear / status
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run)
        grid.addWidget(self.run_button, 10, 0)

        self.clear_button = QPushButton("Clear Log")
        self.clear_button.clicked.connect(self._clear_log)
        grid.addWidget(self.clear_button, 10, 1)

        # Show whether Cython import succeeded
        self.cython_label = QLabel(f"Cython available: {CYTHON_AVAILABLE}")
        grid.addWidget(self.cython_label, 10, 2, 1, 2)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        grid.addWidget(self.status, 11, 0, 8, 4)

        if not CYTHON_AVAILABLE and _IMPORT_ERR is not None:
            self._log(f"Import error for warpaffinemaskrescale: {_IMPORT_ERR}")

        self._update_mode(0)

    def _log(self, *args):
        text = " ".join(str(a) for a in args)
        self.status.append(text)
        QApplication.processEvents()

    def _clear_log(self):
        self.status.clear()

    def _pick_file(self, line_edit, mode='open', filter=None):
        if mode == 'open':
            fn, _ = QFileDialog.getOpenFileName(self, "Select file", "", filter or "All Files (*)")
            if fn:
                line_edit.setText(fn)
        elif mode == 'save':
            fn, _ = QFileDialog.getSaveFileName(self, "Select file", "", filter or "All Files (*)")
            if fn:
                line_edit.setText(fn)
        elif mode == 'dir':
            dn = QFileDialog.getExistingDirectory(self, "Select directory", "")
            if dn:
                line_edit.setText(dn)

    def _update_mode(self, idx):
        mode = self.mode_combo.currentText()
        split_enabled = (mode == "Split Tiles")
        proc_enabled = (mode == "Process Tiles")
        rea_enabled  = (mode == "Reassemble")
        for w in (self.split_input, self.tile_h, self.tile_w, self.tiles_dir):
            w.setEnabled(split_enabled)
        for w in (self.proc_tiles_dir, self.width_square, self.bin_value, self.gamma_edit, self.resize_factor, self.resize_div):
            w.setEnabled(proc_enabled)
        for w in (self.rea_tiles_dir, self.orig_fits):
            w.setEnabled(rea_enabled)

    def _on_run(self):
        mode = self.mode_combo.currentText()
        try:
            if mode == "Split Tiles":
                inp = self.split_input.text().strip()
                if not inp:
                    raise ValueError("Select input FITS to split")
                tile_h = int(self.tile_h.value()); tile_w = int(self.tile_w.value())
                outdir = self.tiles_dir.text().strip() or "tiles"
                image, hdr = load_fits(inp)
                if image is None or image.ndim != 2:
                    raise ValueError("Input FITS must contain a 2D image in primary HDU")
                self._log(f"Splitting {inp} into tiles {tile_h}x{tile_w} -> {outdir}")
                tiles = split_image(image, tile_size=(tile_h, tile_w), output_dir=outdir)
                self._log(f"Created {len(tiles)} tiles")
            elif mode == "Process Tiles":
                tiles_dir = self.proc_tiles_dir.text().strip() or "tiles"
                if not os.path.isdir(tiles_dir):
                    raise ValueError("Tiles directory not found")
                width_square = int(self.width_square.value())
                binv = int(self.bin_value.value())
                gamma = float(self.gamma_edit.text().strip())
                rf = int(self.resize_factor.value()); rd = int(self.resize_div.value())
                files = sorted([os.path.join(tiles_dir, f) for f in os.listdir(tiles_dir) if f.lower().endswith(".fits")])
                if not files:
                    raise ValueError("No .fits tiles found in directory")
                self._log(f"Processing {len(files)} tiles in {tiles_dir}")
                for f in files:
                    try:
                        process_tile(f, width_square, binv, gamma, rf, rd, log_fn=self._log)
                    except Exception as e:
                        self._log(f"Failed {f}: {e}")
                self._log("Processing complete")
            else:
                tiles_dir = self.rea_tiles_dir.text().strip() or "tiles"
                orig = self.orig_fits.text().strip()
                if not os.path.isdir(tiles_dir):
                    raise ValueError("Tiles directory not found")
                if not orig:
                    raise ValueError("Original FITS required to know final shape/header")
                image, hdr = load_fits(orig)
                if image is None:
                    raise ValueError("Original FITS has no data")
                files = sorted([os.path.join(tiles_dir, f) for f in os.listdir(tiles_dir) if f.endswith("_binned_gamma_corrected_drs.fits")])
                if not files:
                    raise ValueError("No processed tiles found with suffix '_binned_gamma_corrected_drs.fits'")
                self._log(f"Reassembling {len(files)} tiles into shape {image.shape}")
                final = reassemble_image(files, image.shape)
                outname = f"output_{os.path.basename(tiles_dir)}.fits"
                fits.writeto(outname, final.astype(np.float32), hdr, overwrite=True)
                self._log(f"Saved reassembled image to {outname}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            self._log("Error:", exc)

# -------------------------
# Entry
# -------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()