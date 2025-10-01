#!/usr/bin/env python3
"""
mask_tool_gui.py
Combine Apply Mask and Invert Mask operations, with PyQt6 GUI.

Features:
- Mode dropdown: Apply Mask or Invert Mask
- File-type selector: FITS or Image (non-FITS)
- File pickers for input image, mask (for Apply Mask), and output filename
- Run button, status/log area
- FITS operations use astropy.io.fits; image ops use OpenCV (cv2)
"""
import sys
import os
import traceback
from pathlib import Path

import numpy as np
import cv2
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QRadioButton, QButtonGroup,
    QTextEdit, QMessageBox, QHBoxLayout
)
from PyQt6.QtCore import Qt

# ---------- Core operations ----------
def apply_mask_image_cv2(image_path, mask_path, out_path):
    """Apply mask to a non-FITS image using OpenCV bitwise_and."""
    img = cv2.imread(str(image_path), -1)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    m = cv2.imread(str(mask_path), -1)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask image: {mask_path}")
    # ensure mask is single channel if necessary
    if len(m.shape) == 3 and m.shape[2] > 1:
        # convert mask to grayscale (nonzero will be treated as mask)
        m_gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        # convert mask back to same channels as image if image has multiple channels
        if len(img.shape) == 3 and img.shape[2] > 1:
            mask3 = cv2.merge([m_gray] * img.shape[2])
            masked = cv2.bitwise_and(img, mask3)
        else:
            masked = cv2.bitwise_and(img, m_gray)
    else:
        masked = cv2.bitwise_and(img, m)
    ok = cv2.imwrite(str(out_path), masked)
    if not ok:
        raise IOError(f"Failed to write output image: {out_path}")
    return out_path

def invert_image_cv2(image_path, out_path):
    """Invert a non-FITS image with OpenCV."""
    img = cv2.imread(str(image_path), -1)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    inv = cv2.bitwise_not(img)
    ok = cv2.imwrite(str(out_path), inv)
    if not ok:
        raise IOError(f"Failed to write output image: {out_path}")
    return out_path

def apply_mask_fits(image_path, mask_path, out_path):
    """Apply mask to FITS image: elementwise multiplication (with broadcasting safety)."""
    with fits.open(str(image_path)) as h1:
        data1 = h1[0].data.astype(np.float64)
        hdr1 = h1[0].header
    with fits.open(str(mask_path)) as h2:
        data2 = h2[0].data.astype(np.float64)

    if data1.shape != data2.shape:
        # try broadcasting if mask is single channel or slice
        if data2.ndim == 2 and data1.ndim == 3 and data1.shape[0] <= 4:
            # channel-first case: broadcast mask across channels
            data2b = np.broadcast_to(data2, data1.shape)
            result = np.where(np.isfinite(data1), data1 * data2b, data1)
        else:
            raise ValueError("FITS shapes do not match and cannot be broadcasted.")
    else:
        result = np.where(np.isfinite(data1), data1 * data2, data1)

    hdu = fits.PrimaryHDU(result, header=hdr1)
    hdu.writeto(str(out_path), overwrite=True)
    return out_path

def invert_fits(image_path, out_path):
    """Invert FITS data by normalizing to [0,1] then 1 - normalized. Preserves header."""
    with fits.open(str(image_path)) as hdul:
        data = hdul[0].data.astype(np.float64)
        hdr = hdul[0].header

    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("FITS image contains no finite pixels to invert.")

    dmin = np.nanmin(data)
    dmax = np.nanmax(data)
    if dmax == dmin:
        inverted = np.zeros_like(data)
    else:
        norm = (data - dmin) / (dmax - dmin)
        inverted = 1.0 - norm
        # restore masked/non-finite pixels to NaN
        inverted[~finite] = data[~finite]

    hdu = fits.PrimaryHDU(inverted, header=hdr)
    hdu.writeto(str(out_path), overwrite=True)
    return out_path

# ---------- GUI ----------
class MaskToolWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mask Tool (Apply / Invert)")
        self._build_ui()
        self.resize(760, 380)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Mode dropdown
        grid.addWidget(QLabel("Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Apply Mask", "Invert Mask"])
        self.mode_combo.currentIndexChanged.connect(self._update_mode_ui)
        grid.addWidget(self.mode_combo, 0, 1, 1, 3)

        # File type radio
        grid.addWidget(QLabel("File type:"), 1, 0)
        self.fits_radio = QRadioButton("FITS")
        self.img_radio = QRadioButton("Image")
        self.fits_radio.setChecked(True)
        rg = QButtonGroup(self)
        rg.addButton(self.fits_radio)
        rg.addButton(self.img_radio)
        h = QHBoxLayout()
        h.addWidget(self.fits_radio); h.addWidget(self.img_radio)
        grid.addLayout(h, 1, 1, 1, 3)

        # Input image
        grid.addWidget(QLabel("Input Image / FITS:"), 2, 0)
        self.input_edit = QLineEdit()
        grid.addWidget(self.input_edit, 2, 1, 1, 2)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(lambda: self._pick_file(self.input_edit, for_fits=self.fits_radio.isChecked()))
        grid.addWidget(btn_in, 2, 3)

        # Mask (only for Apply Mask)
        grid.addWidget(QLabel("Mask (for Apply Mask):"), 3, 0)
        self.mask_edit = QLineEdit()
        grid.addWidget(self.mask_edit, 3, 1, 1, 2)
        btn_mask = QPushButton("Browse")
        btn_mask.clicked.connect(lambda: self._pick_file(self.mask_edit, for_fits=self.fits_radio.isChecked()))
        grid.addWidget(btn_mask, 3, 3)

        # Output file
        grid.addWidget(QLabel("Output filename:"), 4, 0)
        self.output_edit = QLineEdit("masked_output.fits")
        grid.addWidget(self.output_edit, 4, 1, 1, 2)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._pick_save)
        grid.addWidget(btn_out, 4, 3)

        # Run / Clear
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 5, 0)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(self._clear_log)
        grid.addWidget(self.clear_btn, 5, 1)

        # Status log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 6, 0, 4, 4)

        # initialize mode UI
        self._update_mode_ui()

    def _pick_file(self, line_edit, for_fits=True):
        if for_fits:
            fn, _ = QFileDialog.getOpenFileName(self, "Select FITS file", "", "FITS Files (*.fits *.fit);;All Files (*)")
        else:
            fn, _ = QFileDialog.getOpenFileName(self, "Select image file", "", "Images (*.png *.jpg *.tif *.tiff *.bmp);;All Files (*)")
        if fn:
            line_edit.setText(fn)

    def _pick_save(self):
        # choose appropriate default filter based on filetype radio
        if self.fits_radio.isChecked():
            fn, _ = QFileDialog.getSaveFileName(self, "Save output FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        else:
            fn, _ = QFileDialog.getSaveFileName(self, "Save output image", "", "Images (*.png *.jpg *.tif *.tiff *.bmp);;All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _clear_log(self):
        self.log.clear()

    def _update_mode_ui(self):
        mode = self.mode_combo.currentText()
        if mode == "Apply Mask":
            self.mask_edit.setEnabled(True)
        else:
            self.mask_edit.setEnabled(False)
        # update file dialog behavior for browses
        # (No need to change widgets themselves here; browse callbacks check radio buttons)

    def _on_run(self):
        mode = self.mode_combo.currentText()
        use_fits = self.fits_radio.isChecked()
        in_path = self.input_edit.text().strip()
        out_path = self.output_edit.text().strip()
        mask_path = self.mask_edit.text().strip()

        try:
            if not in_path:
                raise ValueError("Input file required")
            if not out_path:
                raise ValueError("Output filename required")

            if mode == "Apply Mask":
                if not mask_path:
                    raise ValueError("Mask file required for Apply Mask mode")
                self._log(f"Running Apply Mask (fits={use_fits}) on {in_path} with mask {mask_path} -> {out_path}")
                if use_fits:
                    res = apply_mask_fits(in_path, mask_path, out_path)
                else:
                    res = apply_mask_image_cv2(in_path, mask_path, out_path)
                self._log(f"Saved masked output: {res}")

            else:  # Invert Mask
                self._log(f"Running Invert (fits={use_fits}) on {in_path} -> {out_path}")
                if use_fits:
                    res = invert_fits(in_path, out_path)
                else:
                    res = invert_image_cv2(in_path, out_path)
                self._log(f"Saved inverted output: {res}")

            QMessageBox.information(self, "Done", f"Operation completed successfully.\nSaved: {res}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

# ---------- Entry ----------
def main():
    app = QApplication(sys.argv)
    win = MaskToolWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()