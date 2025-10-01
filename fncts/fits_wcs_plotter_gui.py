#!/usr/bin/env python3
"""
fits_wcs_plotter_gui.py
PyQt6 GUI to load a 3-plane FITS cube (RGB), preview it with WCS axes and basic stretch,
and save the plotted figure to an image file.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QMessageBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt

# Matplotlib with QtAgg backend via PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ---------- helper functions ----------
def read_fits_rgb(path):
    with fits.open(path) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header
    if data is None:
        raise ValueError("No data in primary HDU")
    arr = np.asarray(data)
    # Accept (3, Y, X) or (Y, X, 3)
    if arr.ndim == 3:
        if arr.shape[0] == 3:
            rgb = np.transpose(arr, (1, 2, 0))
        elif arr.shape[2] == 3:
            rgb = arr
        else:
            # if more planes, try first three
            if arr.shape[0] >= 3:
                rgb = np.transpose(arr[:3], (1,2,0))
            elif arr.shape[2] >= 3:
                rgb = arr[..., :3]
            else:
                raise ValueError(f"Unsupported 3D shape for RGB: {arr.shape}")
    elif arr.ndim == 2:
        # grayscale -> replicate
        rgb = np.stack([arr]*3, axis=-1)
    else:
        raise ValueError(f"Unsupported data ndim: {arr.ndim}")
    return rgb.astype(np.float64), hdr

def stretch_rgb(rgb, mode="Linear", param=1.0):
    """
    rgb: HxWx3 float array
    mode: "Linear", "Log", or "Asinh"
    param: stretch parameter (for asinh scaling intensity)
    returns scaled in the range 0..1 (float)
    """
    out = np.array(rgb, dtype=np.float64)
    # compute channel-wise normalization using 99.5 percentile to ignore outliers
    scaled = np.zeros_like(out)
    for c in range(3):
        ch = out[..., c]
        finite = np.isfinite(ch)
        if not finite.any():
            scaled[..., c] = 0.0
            continue
        pmin = np.nanpercentile(ch[finite], 0.5)
        pmax = np.nanpercentile(ch[finite], 99.5)
        if pmax <= pmin:
            norm = np.clip(ch - pmin, 0.0, 1.0)
        else:
            norm = (ch - pmin) / (pmax - pmin)
        if mode == "Linear":
            s = np.clip(norm, 0.0, 1.0)
        elif mode == "Log":
            s = np.log1p(param * norm) / np.log1p(param)
        elif mode == "Asinh":
            s = np.arcsinh(param * norm) / np.arcsinh(param)
        else:
            s = np.clip(norm, 0.0, 1.0)
        scaled[..., c] = s
    # clip to 0..1
    return np.clip(scaled, 0.0, 1.0)

# ---------- Matplotlib canvas ----------
class WcsCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=(6,6), dpi=100):
        fig = Figure(figsize=figsize, dpi=dpi)
        super().__init__(fig)
        self.fig = fig
        self.ax = None

    def plot_rgb_with_wcs(self, rgb01, header, title=""):
        self.fig.clf()
        try:
            wcs = WCS(header, naxis=2)
            ax = self.fig.add_subplot(1,1,1, projection=wcs)
            ax.imshow(rgb01, origin='lower', aspect='equal')
            ax.coords.grid(True, color="white", ls="dotted")
            ax.set_xlabel("Right Ascension")
            ax.set_ylabel("Declination")
            ax.set_title(title)
            self.ax = ax
        except Exception:
            # fallback without WCS
            ax = self.fig.add_subplot(1,1,1)
            ax.imshow(rgb01, origin='lower', aspect='equal')
            ax.set_title(title + " (no valid WCS)")
            self.ax = ax
        self.draw()

    def save_figure(self, outpath, dpi=150):
        self.fig.savefig(outpath, dpi=dpi, bbox_inches="tight")

# ---------- GUI ----------
class FitsWcsPlotterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS WCS RGB Plotter")
        self._build_ui()
        self.resize(900, 620)
        self.rgb = None
        self.header = None
        self.input_path = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Row 0: FITS pick
        grid.addWidget(QLabel("FITS WCS file:"), 0, 0)
        self.fits_edit = QLineEdit()
        grid.addWidget(self.fits_edit, 0, 1, 1, 3)
        btn_fits = QPushButton("Browse")
        btn_fits.clicked.connect(self._browse_fits)
        grid.addWidget(btn_fits, 0, 4)

        # Row 1: Title
        grid.addWidget(QLabel("Plot title:"), 1, 0)
        self.title_edit = QLineEdit()
        grid.addWidget(self.title_edit, 1, 1, 1, 3)

        # Row 2: Output file
        grid.addWidget(QLabel("Save plot as:"), 2, 0)
        self.out_edit = QLineEdit()
        grid.addWidget(self.out_edit, 2, 1, 1, 3)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_output)
        grid.addWidget(btn_out, 2, 4)

        # Row 3: Stretch options
        grid.addWidget(QLabel("Stretch:"), 3, 0)
        self.stretch_combo = QComboBox()
        self.stretch_combo.addItems(["Linear", "Log", "Asinh"])
        grid.addWidget(self.stretch_combo, 3, 1)

        grid.addWidget(QLabel("Parameter:"), 3, 2)
        self.param_spin = QDoubleSpinBox()
        self.param_spin.setRange(0.01, 100.0)
        self.param_spin.setDecimals(3)
        self.param_spin.setValue(1.0)
        grid.addWidget(self.param_spin, 3, 3)

        # Row 4: Buttons
        self.load_btn = QPushButton("Load FITS & Preview")
        self.load_btn.clicked.connect(self._load_and_preview)
        grid.addWidget(self.load_btn, 4, 0, 1, 2)

        self.plot_btn = QPushButton("Plot & Save")
        self.plot_btn.clicked.connect(self._plot_and_save)
        grid.addWidget(self.plot_btn, 4, 2, 1, 2)

        # Canvas (large)
        self.canvas = WcsCanvas(figsize=(6,6), dpi=110)
        grid.addWidget(self.canvas, 5, 0, 1, 5)

        # Log area
        grid.addWidget(QLabel("Log:"), 6, 0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(120)
        grid.addWidget(self.log, 7, 0, 1, 5)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_fits(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS WCS file", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.fits_edit.setText(fn)

    def _browse_output(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save plot as", "", "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)")
        if fn:
            self.out_edit.setText(fn)

    def _load_and_preview(self):
        path = self.fits_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Input required", "Select a FITS file first")
            return
        try:
            rgb, hdr = read_fits_rgb(path)
            self.rgb = rgb
            self.header = hdr
            self.input_path = path
            self._log(f"Loaded {Path(path).name} shape={rgb.shape} dtype={rgb.dtype}")
            # apply stretch and preview
            mode = self.stretch_combo.currentText()
            param = float(self.param_spin.value())
            rgb01 = stretch_rgb(self.rgb, mode=mode, param=param)
            self.canvas.plot_rgb_with_wcs(rgb01, self.header, title=self.title_edit.text().strip() or Path(path).name)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"{e}\n\n{tb}")

    def _plot_and_save(self):
        if self.rgb is None or self.header is None:
            # try load automatically
            self._load_and_preview()
            if self.rgb is None:
                return
        outpath = self.out_edit.text().strip()
        if not outpath:
            # auto derive from input
            if self.input_path:
                base = Path(self.input_path)
                outpath = str(base.with_suffix("")) + "_wcs.png"
            else:
                QMessageBox.warning(self, "Output required", "Choose an output filename")
                return
        try:
            title = self.title_edit.text().strip() or Path(self.input_path).name
            mode = self.stretch_combo.currentText()
            param = float(self.param_spin.value())
            rgb01 = stretch_rgb(self.rgb, mode=mode, param=param)
            # re-plot to canvas before saving (ensures current settings)
            self.canvas.plot_rgb_with_wcs(rgb01, self.header, title=title)
            self.canvas.save_figure(outpath, dpi=150)
            self._log(f"Saved plot to {outpath}")
            QMessageBox.information(self, "Saved", f"Plot saved to:\n{outpath}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Save error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = FitsWcsPlotterWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()