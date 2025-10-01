#!/usr/bin/env python3
"""
gamma_gui.py
PyQt6 GUI to apply gamma correction to a FITS image and save the result.
"""
import sys
import os
import traceback
import numpy as np
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QDoubleSpinBox, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------- Matplotlib canvas for histogram ----------
class HistCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6,3))
        super().__init__(fig)
        self.ax = fig.add_subplot(1,1,1)
        fig.tight_layout()

    def plot(self, data, bins=256):
        self.ax.clear()
        if data is None:
            self.ax.set_title("No data")
        else:
            arr = np.asarray(data).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                self.ax.hist(arr, bins=bins, color='C0', histtype='stepfilled', alpha=0.6)
                self.ax.set_title("Histogram")
            else:
                self.ax.set_title("No finite data")
        self.draw()

# ---------- Core gamma operation ----------
def load_fits(path):
    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    return data, header

def save_fits(path, data, header=None):
    if header is None:
        fits.writeto(path, data, overwrite=True)
    else:
        fits.writeto(path, data, header=header, overwrite=True)

def apply_gamma(data, gamma):
    # preserve NaNs and infs; operate on finite pixels only
    arr = np.array(data, dtype=np.float64, copy=True)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("No finite pixels to process.")
    # normalize finite pixels to 0..1 using min/max, apply gamma, then restore scale
    vmin = np.nanmin(arr[finite])
    vmax = np.nanmax(arr[finite])
    if vmax == vmin:
        # constant image -> result remains constant (same value)
        out = np.full_like(arr, arr[0] if arr.size else 0.0)
        out[~finite] = arr[~finite]
        return out
    norm = (arr[finite] - vmin) / (vmax - vmin)
    # apply gamma: out = norm ** gamma
    corrected = norm ** gamma
    # map back to original data range
    out = np.array(arr, dtype=np.float64)
    out[finite] = corrected * (vmax - vmin) + vmin
    return out

# ---------- GUI ----------
class GammaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS Gamma Correction")
        self._build_ui()
        self.resize(900, 520)
        self.data = None
        self.header = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        grid.addWidget(QLabel("Input FITS:"), 0, 0)
        self.input_edit = QLineEdit()
        grid.addWidget(self.input_edit, 0, 1, 1, 3)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self._browse_input)
        grid.addWidget(btn_in, 0, 4)

        grid.addWidget(QLabel("Output base name (no extension):"), 1, 0)
        self.output_edit = QLineEdit("output_gamma")
        grid.addWidget(self.output_edit, 1, 1, 1, 3)
        btn_out = QPushButton("Browse Save Location")
        btn_out.clicked.connect(self._browse_save)
        grid.addWidget(btn_out, 1, 4)

        grid.addWidget(QLabel("Gamma value:"), 2, 0)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 10.0)
        self.gamma_spin.setDecimals(6)
        self.gamma_spin.setSingleStep(0.01)
        self.gamma_spin.setValue(0.3981)
        grid.addWidget(self.gamma_spin, 2, 1)

        self.preview_btn = QPushButton("Preview Stats/Histogram")
        self.preview_btn.clicked.connect(self._preview)
        grid.addWidget(self.preview_btn, 2, 2)

        self.run_btn = QPushButton("Apply Gamma & Save FITS")
        self.run_btn.clicked.connect(self._run)
        grid.addWidget(self.run_btn, 2, 3, 1, 2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(140)
        grid.addWidget(self.log, 3, 0, 1, 5)

        self.canvas = HistCanvas()
        grid.addWidget(self.canvas, 4, 0, 6, 5)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.input_edit.setText(fn)
            try:
                data, hdr = load_fits(fn)
                if data is None:
                    raise ValueError("No data in primary HDU")
                self.data = data.astype(np.float64)
                self.header = hdr
                st = self._compute_stats(self.data)
                self._log(f"Loaded {os.path.basename(fn)} shape={self.data.shape} dtype={self.data.dtype}")
                self._log(f"stats: min={st['min']:.6g} max={st['max']:.6g} mean={st['mean']:.6g} std={st['std']:.6g}")
                self.canvas.plot(self.data)
            except Exception as e:
                QMessageBox.critical(self, "Load error", f"Failed to load FITS: {e}")

    def _browse_save(self):
        # choose directory and base name
        dn = QFileDialog.getExistingDirectory(self, "Select directory to save output", "")
        if dn:
            base = self.output_edit.text().strip() or "output_gamma"
            self.output_edit.setText(os.path.join(dn, base))

    def _compute_stats(self, arr):
        a = np.asarray(arr).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
        return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}

    def _preview(self):
        if self.data is None:
            QMessageBox.warning(self, "No image", "Load an input FITS first")
            return
        try:
            gamma = float(self.gamma_spin.value())
            preview = apply_gamma(self.data, gamma)
            st = self._compute_stats(preview)
            self._log(f"Preview stats (gamma={gamma}): min={st['min']:.6g} max={st['max']:.6g} mean={st['mean']:.6g} std={st['std']:.6g}")
            self.canvas.plot(preview)
        except Exception as e:
            QMessageBox.critical(self, "Preview error", str(e))

    def _run(self):
        if self.data is None:
            QMessageBox.warning(self, "No image", "Load an input FITS first")
            return
        outbase = self.output_edit.text().strip()
        if not outbase:
            QMessageBox.warning(self, "No output", "Specify an output base name or use Browse Save Location")
            return
        try:
            gamma = float(self.gamma_spin.value())
            corrected = apply_gamma(self.data, gamma)
            # build output filename: outbase + _gamma.fit (preserve directory if provided)
            if outbase.lower().endswith((".fits", ".fit")):
                outpath = outbase
            else:
                outpath = outbase + "_gamma.fit"
            # preserve header if present
            save_fits(outpath, corrected, header=self.header)
            st = self._compute_stats(corrected)
            self._log(f"Wrote {outpath}  stats: min={st['min']:.6g} max={st['max']:.6g} mean={st['mean']:.6g} std={st['std']:.6g}")
            QMessageBox.information(self, "Saved", f"Wrote gamma-corrected FITS: {outpath}")
            # update canvas to show result
            self.canvas.plot(corrected)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")
            self._log("Error:", e)

def main():
    app = QApplication(sys.argv)
    w = GammaWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()