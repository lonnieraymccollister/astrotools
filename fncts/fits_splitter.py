#!/usr/bin/env python3
"""
fits_splitter_autodetect_final.py
Splits 3‑plane FITS cubes into blue/green/red single‑plane FITS files.
Auto‑detects RGB channel order using Siril/MaxIm‑style heuristics.
"""

import sys
import os
import glob
import numpy as np
from astropy.io import fits
from scipy.stats import skew

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt


# ---------------------------------------------------------
# FITS helpers
# ---------------------------------------------------------
def clean_header_for_single_plane(hdr):
    """Clean header for a 2D float32 single-plane FITS that Siril will display correctly."""
    hdr = hdr.copy()

    # Force 2D image
    hdr["BITPIX"] = -32
    hdr["NAXIS"] = 2
    for key in ("NAXIS3",):
        if key in hdr:
            del hdr[key]

    # Remove scaling and display/stretch hints
    for key in (
        "BZERO", "BSCALE",
        "DATAMIN", "DATAMAX",
        "MIPS-HI", "MIPS-LO",
        "PEDESTAL", "CSTRETCH",
        "CALSTAT",
    ):
        if key in hdr:
            del hdr[key]

    # Remove SIP / distortion keywords
    sip_prefixes = ("A_", "B_", "AP_", "BP_")
    sip_orders = ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER")
    for key in list(hdr.keys()):
        if key in sip_orders:
            del hdr[key]
        else:
            for p in sip_prefixes:
                if key.startswith(p):
                    del hdr[key]
                    break

    # Remove software metadata that confuses Siril
    for key in ("SWMODIFY", "SWOWNER", "PROGRAM"):
        if key in hdr:
            del hdr[key]

    # Remove HISTORY/COMMENT lines that imply scaling or processing
    clean_history = []
    for h in hdr.get("HISTORY", []):
        if not any(term in h.upper() for term in ("SCALE", "STRETCH", "PEDESTAL", "MIPS")):
            clean_history.append(h)
    if clean_history:
        hdr["HISTORY"] = clean_history
    else:
        if "HISTORY" in hdr:
            del hdr["HISTORY"]

    return hdr


def read_fits(file_path):
    """Read FITS and return normalized (3, Y, X) array + cleaned header."""
    with fits.open(file_path, memmap=False) as hdul:
        header = hdul[0].header.copy()
        data = hdul[0].data

    if data is None:
        raise ValueError(f"No data in FITS file: {file_path}")

    # Normalize shape to (3, Y, X)
    if data.ndim == 3 and data.shape[0] == 3:
        arr = data.astype(np.float64)
    elif data.ndim == 3 and data.shape[2] == 3:
        arr = np.transpose(data, (2, 0, 1)).astype(np.float64)
    else:
        raise ValueError(f"Unexpected FITS shape {data.shape}; expected 3‑plane RGB.")

    # Apply BSCALE/BZERO if present, then remove them
    if "BSCALE" in header or "BZERO" in header:
        bscale = header.get("BSCALE", 1.0)
        bzero = header.get("BZERO", 0.0)
        arr = arr * bscale + bzero

    header = clean_header_for_single_plane(header)
    return arr, header


# ---------------------------------------------------------
# Auto‑detect RGB order (Siril / MaxIm‑style)
# ---------------------------------------------------------
def autodetect_rgb_order(arr):
    """Auto‑detect channel order using mean, median, std, skew."""
    stats = []
    for i in range(3):
        plane = arr[i]
        finite = plane[np.isfinite(plane)]
        if finite.size == 0:
            stats.append({"mean": 0, "median": 0, "std": 0, "skew": 0})
            continue
        stats.append({
            "mean": float(finite.mean()),
            "median": float(np.median(finite)),
            "std": float(finite.std()),
            "skew": float(skew(finite, bias=False))
        })

    means = np.array([s["mean"] for s in stats])
    medians = np.array([s["median"] for s in stats])
    stds = np.array([s["std"] for s in stats])
    skews = np.array([s["skew"] for s in stats])

    # Red: high mean/median + positive skew
    red_score = means + medians + skews * 0.5

    # Blue: low mean/median + high noise
    blue_score = -means + stds

    r_idx = int(np.argmax(red_score))
    b_idx = int(np.argmax(blue_score))
    g_idx = [i for i in range(3) if i not in (r_idx, b_idx)][0]

    return r_idx, g_idx, b_idx


# ---------------------------------------------------------
# GUI
# ---------------------------------------------------------
class FitsSplitter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS Color‑Channel Splitter (Auto‑Detect RGB)")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Input directory
        row_dir = QHBoxLayout()
        row_dir.addWidget(QLabel("Input Directory:"))
        self.dir_edit = QLineEdit()
        row_dir.addWidget(self.dir_edit, stretch=1)
        btn_dir = QPushButton("Browse…")
        btn_dir.clicked.connect(self._browse_dir)
        row_dir.addWidget(btn_dir)
        layout.addLayout(row_dir)

        # Pattern
        row_pat = QHBoxLayout()
        row_pat.addWidget(QLabel("Filename Pattern:"))
        self.pat_edit = QLineEdit("*.fit*")
        row_pat.addWidget(self.pat_edit, stretch=1)
        layout.addLayout(row_pat)

        # Split button
        btn_split = QPushButton("Split Channels")
        btn_split.clicked.connect(self._split_channels)
        btn_split.setFixedHeight(36)
        layout.addWidget(btn_split, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)
        self.resize(600, 150)

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory", "")
        if path:
            self.dir_edit.setText(path)

    def _split_channels(self):
        input_dir = self.dir_edit.text().strip() or "."
        pattern = self.pat_edit.text().strip() or "*.fit*"

        files = glob.glob(os.path.join(input_dir, pattern))
        if not files:
            QMessageBox.warning(self, "No Files Found",
                                f"No FITS files matching:\n{os.path.join(input_dir, pattern)}")
            return

        # Create output directories
        for sub in ("blue", "green", "red"):
            os.makedirs(os.path.join(input_dir, sub), exist_ok=True)

        processed = 0
        errors = []

        for fn in files:
            try:
                arr, header = read_fits(fn)  # arr = (3, Y, X)

                # Auto‑detect channel order
                r_idx, g_idx, b_idx = autodetect_rgb_order(arr)

                r_plane = arr[r_idx].astype(np.float32)
                g_plane = arr[g_idx].astype(np.float32)
                b_plane = arr[b_idx].astype(np.float32)

                base = os.path.splitext(os.path.basename(fn))[0]

                mapping = [
                    (b_plane, "b", "blue"),
                    (g_plane, "g", "green"),
                    (r_plane, "r", "red"),
                ]

                for plane, suffix, folder in mapping:
                    out_path = os.path.join(input_dir, folder, f"{base}_{suffix}.fits")
                    fits.writeto(out_path, plane, header, overwrite=True)

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
