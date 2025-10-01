#!/usr/bin/env python3
"""
fits_splitter.py
Simple PyQt6 GUI to split 3-plane FITS cubes into blue/green/red single-plane FITS files.
Assumes input FITS primary HDU contains data shaped (3, Y, X) or (Y, X, 3).
"""

import sys
import os
import glob

import numpy as np
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt

def read_fits(file_path):
    """Read a FITS primary HDU and return (data, header)."""
    hdul = fits.open(file_path)
    header = hdul[0].header
    data = hdul[0].data
    hdul.close()
    if data is None:
        raise ValueError(f"No data found in FITS file: {file_path}")
    # Normalize shapes: prefer (3, Y, X)
    if data.ndim == 3 and data.shape[0] == 3:
        arr = data
    elif data.ndim == 3 and data.shape[2] == 3:
        # (Y, X, 3) -> (3, Y, X)
        arr = np.transpose(data, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected FITS data shape {data.shape} in {file_path}; expected 3 planes.")
    # Ensure header contains SIP orders to avoid downstream code that expects them
    if 'A_ORDER' not in header:
        header['A_ORDER'] = (0, 'dummy SIP order - no distortion')
    if 'B_ORDER' not in header:
        header['B_ORDER'] = (0, 'dummy SIP order - no distortion')
    return arr, header

class FitsSplitter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS Color-Channel Splitter")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Input directory row
        row_dir = QHBoxLayout()
        lbl_dir = QLabel("Input Directory:")
        self.dir_edit = QLineEdit()
        btn_dir = QPushButton("Browse…")
        btn_dir.clicked.connect(self._browse_dir)
        row_dir.addWidget(lbl_dir)
        row_dir.addWidget(self.dir_edit, stretch=1)
        row_dir.addWidget(btn_dir)
        layout.addLayout(row_dir)

        # Filename pattern row
        row_pat = QHBoxLayout()
        lbl_pat = QLabel("Filename Pattern:")
        self.pat_edit = QLineEdit("*.fit")
        row_pat.addWidget(lbl_pat)
        row_pat.addWidget(self.pat_edit, stretch=1)
        layout.addLayout(row_pat)

        # Split button
        btn_split = QPushButton("Split Channels")
        btn_split.clicked.connect(self._split_channels)
        btn_split.setFixedHeight(36)
        layout.addWidget(btn_split, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)
        self.resize(600, 140)

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory", "", QFileDialog.Option.ShowDirsOnly)
        if path:
            self.dir_edit.setText(path)

    def _split_channels(self):
        input_dir = self.dir_edit.text().strip() or "."
        pattern = self.pat_edit.text().strip() or "*.fit"

        files = glob.glob(os.path.join(input_dir, pattern))
        if not files:
            QMessageBox.warning(self, "No Files Found", f"No FITS files matching:\n{os.path.join(input_dir, pattern)}")
            return

        # Create subdirectories
        for sub in ("blue", "green", "red"):
            os.makedirs(os.path.join(input_dir, sub), exist_ok=True)

        processed = 0
        errors = []
        for fn in files:
            try:
                arr, header = read_fits(fn)
                arr = arr.astype(np.float32)

                # arr expected shape (3, Y, X): [0]=R, [1]=G, [2]=B
                b_plane = arr[2, ...]
                g_plane = arr[1, ...]
                r_plane = arr[0, ...]

                base = os.path.splitext(os.path.basename(fn))[0]
                mapping = [
                    (b_plane, "b", "blue"),
                    (g_plane, "g", "green"),
                    (r_plane, "r", "red"),
                ]

                for arr_plane, suffix, folder in mapping:
                    out_path = os.path.join(input_dir, folder, f"{base}_{suffix}.fits")
                    fits.writeto(out_path, arr_plane, header, overwrite=True)

                processed += 1
            except Exception as e:
                errors.append((fn, str(e)))

        msg = f"Split {processed} file(s) into blue/, green/, red/ subdirs."
        if errors:
            msg += f"\n\nErrors for {len(errors)} file(s). See console for details."
            for fn, err in errors:
                print(f"[ERROR] {fn}: {err}")

        QMessageBox.information(self, "Done", msg)

def main():
    app = QApplication(sys.argv)
    splitter = FitsSplitter()
    splitter.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
    win = FitsSplitter()
    win.show()