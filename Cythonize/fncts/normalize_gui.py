#!/usr/bin/env python3
"""
normalize_gui.py
PyQt6 GUI to linearly rescale FITS image data from [old_min, old_max] -> [new_min, new_max].
Provides histogram preview and optional Siril launch.
"""

import sys
import os
import shutil
import traceback
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QHBoxLayout, QVBoxLayout, QMessageBox, QDoubleSpinBox,
    QTextEdit, QCheckBox
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import stats

# ---------- Helpers ----------
def load_fits(path):
    with fits.open(path) as hdul:
        return hdul[0].data, hdul[0].header

def write_fits(path, data, header=None):
    if header is None:
        fits.writeto(path, data, overwrite=True)
    else:
        fits.writeto(path, data, header=header, overwrite=True)

def compute_stats(arr):
    a = np.asarray(arr).ravel()
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"min": np.nan, "max": np.nan, "median": np.nan, "mean": np.nan, "std": np.nan}
    return {"min": float(a.min()), "max": float(a.max()),
            "median": float(np.median(a)), "mean": float(a.mean()),
            "std": float(a.std())}

def normalize_array(data, old_min, old_max, new_min, new_max):
    arr = data.astype(np.float64)
    # avoid divide by zero
    if old_max == old_min:
        scaled = np.full_like(arr, new_min, dtype=np.float64)
    else:
        scaled = (arr - old_min) / (old_max - old_min)
        scaled = scaled * (new_max - new_min) + new_min
    return np.clip(scaled, min(new_min, new_max), max(new_min, new_max))

# ---------- Matplotlib canvas ----------
class HistCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6,3))
        super().__init__(fig)
        self.orig_ax = fig.add_subplot(1,2,1)
        self.norm_ax = fig.add_subplot(1,2,2)
        fig.tight_layout()

    def plot(self, orig, norm=None, bins=256):
        self.orig_ax.clear()
        self.norm_ax.clear()
        if orig is not None:
            self.orig_ax.hist(orig.ravel(), bins=bins, color='blue', histtype='step')
            self.orig_ax.set_title("Original histogram")
        else:
            self.orig_ax.set_title("Original: none")
        if norm is not None:
            self.norm_ax.hist(norm.ravel(), bins=bins, color='red', histtype='step')
            self.norm_ax.set_title("Normalized histogram")
        else:
            self.norm_ax.set_title("Normalized: none")
        self.draw()

# ---------- Main window ----------


class NormalizeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Normalize FITS")
        self.image = None
        self.header = None
        self._build_ui()
        self.resize(1000, 620)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Input file row
        grid.addWidget(QLabel("Input FITS:"), 0, 0)
        self.input_edit = QLineEdit()
        grid.addWidget(self.input_edit, 0, 1, 1, 3)
        btn_in = QPushButton("Browse...")
        btn_in.clicked.connect(self._browse_input)
        grid.addWidget(btn_in, 0, 4)

        # Output file row
        grid.addWidget(QLabel("Output FITS:"), 1, 0)
        self.output_edit = QLineEdit("normalized_output.fits")
        grid.addWidget(self.output_edit, 1, 1, 1, 3)
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(self._browse_output)
        grid.addWidget(btn_out, 1, 4)

        # Old/new ranges
        grid.addWidget(QLabel("Old min:"), 2, 0)
        self.old_min = QDoubleSpinBoxWithLargeRange()
        grid.addWidget(self.old_min, 2, 1)
        grid.addWidget(QLabel("Old max:"), 2, 2)
        self.old_max = QDoubleSpinBoxWithLargeRange()
        grid.addWidget(self.old_max, 2, 3)

        grid.addWidget(QLabel("New min:"), 3, 0)
        self.new_min = QDoubleSpinBoxWithLargeRange()
        grid.addWidget(self.new_min, 3, 1)
        grid.addWidget(QLabel("New max:"), 3, 2)
        self.new_max = QDoubleSpinBoxWithLargeRange()
        grid.addWidget(self.new_max, 3, 3)

        # Buttons
        self.preview_btn = QPushButton("Preview Stats/Histogram")
        self.preview_btn.clicked.connect(self._preview)
        grid.addWidget(self.preview_btn, 4, 0)

        self.normalize_btn = QPushButton("Run Normalize & Save")
        self.normalize_btn.clicked.connect(self._run_normalize)
        grid.addWidget(self.normalize_btn, 4, 1)

        # NEW: Plot output file button
        self.plot_output_btn = QPushButton("Plot Output File")
        self.plot_output_btn.clicked.connect(self._plot_output_file)
        grid.addWidget(self.plot_output_btn, 4, 2)

        self.siril_chk = QCheckBox("Offer to open in Siril after save")
        self.siril_chk.setChecked(False)
        grid.addWidget(self.siril_chk, 4, 3, 1, 2)

        # Info text
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setFixedHeight(120)
        grid.addWidget(self.info, 5, 0, 1, 5)

        # Histogram canvas
        self.canvas = HistCanvas()
        grid.addWidget(self.canvas, 6, 0, 6, 5)

    def _browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.input_edit.setText(fn)
            try:
                data, hdr = load_fits(fn)
                if data is None:
                    raise ValueError("No data in primary HDU")
                if data.ndim not in (2,3):
                    raise ValueError(f"Unsupported data ndim {data.ndim} — expected 2D or 3D")
                self.image = data.astype(np.float64)
                self.header = hdr
                st = compute_stats(self.image)
                self.info.append(f"Loaded {os.path.basename(fn)} shape={self.image.shape}")
                self.info.append(f"stats: min={st['min']:.6g} max={st['max']:.6g} mean={st['mean']:.6g} std={st['std']:.6g}")
                # set defaults for old/new from data range
                self.old_min.setValue(st['min'])
                self.old_max.setValue(st['max'])
                # default new range 0..1 scaled to original
                self.new_min.setValue(0.0)
                self.new_max.setValue(1.0)
                # preview original histogram
                self.canvas.plot(self.image, None)
            except Exception as e:
                QMessageBox.critical(self, "Load error", f"Failed to load FITS: {e}")

    def _browse_output(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save FITS as", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def _preview(self):
        if self.image is None:
            QMessageBox.warning(self, "No image", "Load an input FITS first")
            return
        try:
            old_min = float(self.old_min.value())
            old_max = float(self.old_max.value())
            new_min = float(self.new_min.value())
            new_max = float(self.new_max.value())
            preview = normalize_array(self.image, old_min, old_max, new_min, new_max)
            pst = compute_stats(preview)
            self.info.append(f"Preview stats: min={pst['min']:.6g} max={pst['max']:.6g} mean={pst['mean']:.6g} std={pst['std']:.6g}")
            self.canvas.plot(self.image, preview)
        except Exception as e:
            QMessageBox.critical(self, "Preview error", str(e))

    def _run_normalize(self):
        if self.image is None:
            QMessageBox.warning(self, "No image", "Load an input FITS first")
            return
        outpath = self.output_edit.text().strip()
        if not outpath:
            QMessageBox.warning(self, "No output", "Specify output FITS filename")
            return
        try:
            old_min = float(self.old_min.value())
            old_max = float(self.old_max.value())
            new_min = float(self.new_min.value())
            new_max = float(self.new_max.value())
            result = normalize_array(self.image, old_min, old_max, new_min, new_max)
            write_fits(outpath, result.astype(np.float64), header=self.header)
            self.info.append(f"Saved normalized FITS to {outpath}")
            self.canvas.plot(self.image, result)
            if self.siril_chk.isChecked():
                self._maybe_open_siril(outpath)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

    def _plot_output_file(self):
        """
        Load the output FITS file (from output_edit text or ask if empty)
        and plot it in the right-hand histogram (normalized panel).
        """
        outpath = self.output_edit.text().strip() or "normalized_output.fits"
        if not os.path.exists(outpath):
            # ask user to select file if default doesn't exist
            fn, _ = QFileDialog.getOpenFileName(self, "Select output FITS to plot", "", "FITS Files (*.fits *.fit);;All Files (*)")
            if not fn:
                return
            outpath = fn

        try:
            data, hdr = load_fits(outpath)
            if data is None:
                raise ValueError("No data in primary HDU")
            # If 3D, attempt to select first 2D plane
            if data.ndim > 2:
                # common behavior: take first plane along axis 0
                data2 = data[0].astype(np.float64)
            else:
                data2 = data.astype(np.float64)

            # Plot original (if loaded) and output data. If original not loaded, plot None on left.
            self.canvas.plot(self.image if self.image is not None else None, data2)
            st = compute_stats(data2)
            self.info.append(f"Plotted output FITS: {os.path.basename(outpath)} stats: min={st['min']:.6g} max={st['max']:.6g} mean={st['mean']:.6g} std={st['std']:.6g}")
        except Exception as e:
            QMessageBox.critical(self, "Plot error", f"Failed to load/plot output FITS: {e}")

    def _maybe_open_siril(self, fits_path):
        if shutil.which('siril') is None:
            QMessageBox.information(self, "Siril not found", "siril not found on PATH; cannot open")
            return
        try:
            # non-blocking launch
            if sys.platform.startswith('win'):
                os.startfile(fits_path)
            else:
                import subprocess
                subprocess.Popen(['siril', fits_path])
            self.info.append("Launched Siril (or opened file with default) for " + fits_path)
        except Exception as e:
            QMessageBox.warning(self, "Failed to open", f"Failed to open in Siril: {e}")

# convenience QDoubleSpinBox with wide range
class QDoubleSpinBoxWithLargeRange(QDoubleSpinBox):
    def __init__(self, default=0.0):
        super().__init__()
        self.setRange(-1e12, 1e12)
        self.setDecimals(9)
        self.setSingleStep(0.1)
        self.setValue(float(default))

# ---------- Entry ----------
def main():
    app = QApplication(sys.argv)
    w = NormalizeWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()