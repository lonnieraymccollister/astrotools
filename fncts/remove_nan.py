#!/usr/bin/env python3
"""
replace_black_with_white_with_nodata.py

PyQt6 GUI:
- Open a FITS float image (primary HDU).
- Replace NaN / no-data / Inf pixels with 1.0.
- Replace black pixels (exactly zero or within tolerance) with 1.0.
- Preserve original channel layout (channel-first or channel-last).
- Save modified image preserving header and layout.
"""
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QDoubleSpinBox, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt


def load_fits_primary(path):
    """Load primary HDU data and header. Return (arr, header, layout_info).
    arr is float64 and in shape (Y,X) or (Y,X,3).
    layout_info: {'channel_first': bool, 'orig_shape': tuple}
    """
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    if data is None:
        raise ValueError("FITS contains no primary data")

    arr = np.array(data)  # keep original layout for detection
    layout_info = {"orig_shape": arr.shape, "channel_first": False}

    # Detect and normalize to (Y, X) or (Y, X, 3)
    if arr.ndim == 3 and arr.shape[0] == 3:
        # channel-first -> transpose to channel-last
        arr = np.transpose(arr, (1, 2, 0))
        layout_info["channel_first"] = True
    elif arr.ndim == 3 and arr.shape[2] == 3:
        layout_info["channel_first"] = False
    elif arr.ndim == 2:
        # single plane
        pass
    else:
        # unsupported shape
        raise ValueError(f"Unsupported FITS data shape: {arr.shape}")

    # Work in float64 for safety
    arr = arr.astype(np.float64, copy=False)
    return arr, header, layout_info


def write_fits_preserve_layout(path, arr_yxc, header, layout_info):
    """Write arr_yxc back to FITS preserving original layout.
    arr_yxc is expected in (Y,X) or (Y,X,3) layout (channel-last).
    If original was channel-first, transpose to (3,Y,X) before writing.
    """
    out = np.array(arr_yxc, copy=False)

    # Preserve original layout
    if layout_info.get("channel_first", False):
        if out.ndim == 3 and out.shape[2] == 3:
            out_write = np.transpose(out, (2, 0, 1))
        elif out.ndim == 2:
            out_write = out[np.newaxis, :, :]
        else:
            out_write = out
    else:
        out_write = out

    # Prepare header
    hdr = header.copy() if header is not None else fits.Header()
    hdr['BITPIX'] = -32
    fits.writeto(path, out_write.astype(np.float32), hdr, overwrite=True)


class ReplaceBlackWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Replace Black Pixels with White and Fix No Data")
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QGridLayout(central)

        layout.addWidget(QLabel("Input FITS:"), 0, 0)
        self.input_edit = QLineEdit()
        layout.addWidget(self.input_edit, 0, 1)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self.browse_input)
        layout.addWidget(btn_in, 0, 2)

        layout.addWidget(QLabel("Output FITS:"), 1, 0)
        self.output_edit = QLineEdit()
        layout.addWidget(self.output_edit, 1, 1)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self.browse_output)
        layout.addWidget(btn_out, 1, 2)

        layout.addWidget(QLabel("Tolerance treat |value| <= tol as black:"), 2, 0)
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setDecimals(12)
        self.tol_spin.setRange(0.0, 1.0)
        self.tol_spin.setSingleStep(1e-6)
        self.tol_spin.setValue(0.0)
        layout.addWidget(self.tol_spin, 2, 1)

        self.apply_rgb_checkbox = QCheckBox("For RGB treat pixel as black only if all channels are <= tol")
        self.apply_rgb_checkbox.setChecked(True)
        layout.addWidget(self.apply_rgb_checkbox, 3, 0, 1, 3)

        self.save_button = QPushButton("Fix No Data, Replace Black and Save")
        self.save_button.clicked.connect(self.replace_and_save)
        layout.addWidget(self.save_button, 4, 0, 1, 3)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label, 5, 0, 1, 3)

        self.setMinimumSize(700, 240)

    def browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.input_edit.setText(fn)
            p = Path(fn)
            out = str(p.with_name(p.stem + "_fixed.fits"))
            self.output_edit.setText(out)

    def browse_output(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def replace_and_save(self):
        self.status_label.setText("")
        in_path = self.input_edit.text().strip()
        out_path = self.output_edit.text().strip()
        if not in_path or not out_path:
            self.status_label.setText("Provide both input and output file paths.")
            return

        try:
            arr, hdr, info = load_fits_primary(in_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load FITS: {e}")
            return

        tol = float(self.tol_spin.value())
        is_rgb = (arr.ndim == 3 and arr.shape[2] == 3)

        try:
            # 1) Fix no-data / NaN / Inf first
            if is_rgb:
                # mask where any channel is non-finite
                nodata_mask = np.any(~np.isfinite(arr), axis=-1)
                if np.any(nodata_mask):
                    arr[nodata_mask, :] = 1.0
            else:
                # single plane: set non-finite entries to 1.0
                bad_mask = ~np.isfinite(arr)
                if np.any(bad_mask):
                    arr[bad_mask] = 1.0

            # 2) Replace black pixels according to tolerance and RGB option
            if is_rgb:
                if self.apply_rgb_checkbox.isChecked():
                    # black if all channels <= tol
                    mask = np.all(np.abs(arr) <= tol, axis=-1)
                    arr[mask, :] = 1.0
                else:
                    # treat each channel independently
                    ch_mask = np.abs(arr) <= tol
                    arr[ch_mask] = 1.0
            else:
                mask = np.abs(arr) <= tol
                arr[mask] = 1.0

        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"Failed to modify pixels: {e}")
            return

        # Write preserving layout
        try:
            write_fits_preserve_layout(out_path, arr, hdr, info)
            self.status_label.setText(f"Saved modified FITS to: {out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save FITS: {e}")
            return


def main():
    app = QApplication(sys.argv)
    w = ReplaceBlackWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()