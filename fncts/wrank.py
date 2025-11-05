#!/usr/bin/env python3
import sys
import numpy as np
from astropy.io import fits
from scipy import ndimage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QSlider
)
from PyQt6.QtCore import Qt
from pathlib import Path

# --- rank filter helpers (same as earlier) ---
def circular_footprint(radius):
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    return (xx**2 + yy**2) <= (radius**2)

def rank_filter_image(img, radius=3.0, percentile=50.0):
    img = img.astype(np.float64)
    nan_mask = np.isnan(img)
    if nan_mask.any():
        # fill NaNs with a finite minimum so rank filter can operate
        finite_min = np.nanmin(img)
        fill_val = finite_min if np.isfinite(finite_min) else 0.0
        tmp = img.copy()
        tmp[nan_mask] = fill_val
        img_work = tmp
    else:
        img_work = img

    fp = circular_footprint(radius)
    N = int(fp.sum())
    if N <= 0:
        raise ValueError("Footprint size is zero; check radius value.")
    # compute order and clamp to valid range
    order = int(round((percentile / 100.0) * (N - 1)))
    order = max(0, min(N - 1, order))
    filtered = ndimage.rank_filter(img_work, rank=order, footprint=fp, mode='mirror')
    if nan_mask.any():
        filtered[nan_mask] = np.nan
    return filtered

def combine_with_weight(orig, filtered, weight=0.6):
    return weight * filtered + (1.0 - weight) * orig

def rank_filter_fits(inpath, outpath, radius=3.0, percentile=50.0, weight=0.6):
    with fits.open(inpath) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()

    if data is None:
        raise ValueError("Input FITS contains no primary data array.")

    if data.ndim == 2:
        filtered = rank_filter_image(data, radius=radius, percentile=percentile)
        out = combine_with_weight(data.astype(np.float64), filtered, weight=weight)
    elif data.ndim == 3:
        out = np.empty_like(data, dtype=np.float64)
        for k in range(data.shape[0]):
            slice_i = data[k].astype(np.float64)
            filtered = rank_filter_image(slice_i, radius=radius, percentile=percentile)
            out[k] = combine_with_weight(slice_i, filtered, weight=weight)
    else:
        raise ValueError("Unsupported data dimensionality: %d" % data.ndim)

    header.add_history(f"Rank filter radius={radius}, pct={percentile}, weight={weight}")
    fits.writeto(outpath, out, header=header, overwrite=True)
    return out

# --- PyQt6 GUI ---
class RankFilterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rank Filter (Maxim DL style)")
        self._inpath = ""
        self._outpath = ""
        self._build_ui()

    def _build_ui(self):
        w = QWidget()
        v = QVBoxLayout()

        # File selection row
        row = QHBoxLayout()
        self.in_label = QLineEdit()
        self.in_label.setPlaceholderText("Input FITS file")
        btn_in = QPushButton("Open Input")
        btn_in.clicked.connect(self._choose_input)
        row.addWidget(self.in_label)
        row.addWidget(btn_in)
        v.addLayout(row)

        # Output selection row
        row2 = QHBoxLayout()
        self.out_label = QLineEdit()
        self.out_label.setPlaceholderText("Output FITS file")
        btn_out = QPushButton("Save As")
        btn_out.clicked.connect(self._choose_output)
        row2.addWidget(self.out_label)
        row2.addWidget(btn_out)
        v.addLayout(row2)

        # Parameters: radius, percentile, weight
        # radius
        rrow = QHBoxLayout()
        rrow.addWidget(QLabel("Radius (px):"))
        self.radius_edit = QLineEdit("8.0")
        rrow.addWidget(self.radius_edit)
        v.addLayout(rrow)

        # percentile
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Percentile (0-100):"))
        self.percentile_edit = QLineEdit("10")
        prow.addWidget(self.percentile_edit)
        v.addLayout(prow)

        # weight (slider + numeric)
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("Blend weight (filtered):"))
        self.weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.weight_slider.setRange(0, 100)
        self.weight_slider.setValue(60)
        self.weight_slider.valueChanged.connect(self._sync_weight_edit)
        self.weight_edit = QLineEdit("0.60")
        self.weight_edit.editingFinished.connect(self._sync_weight_slider)
        wrow.addWidget(self.weight_slider)
        wrow.addWidget(self.weight_edit)
        v.addLayout(wrow)

        # Run button
        self.btn_run = QPushButton("Run Rank Filter")
        self.btn_run.clicked.connect(self._run_filter)
        v.addWidget(self.btn_run)

        w.setLayout(v)
        self.setCentralWidget(w)

        # initialize weight edit from slider value
        self._sync_weight_edit()

    def _choose_input(self):
        start = str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "Select input FITS", start, "FITS Files (*.fits);;All Files (*)")
        if path:
            self._inpath = path
            self.in_label.setText(path)

    def _choose_output(self):
        start = str(Path.cwd())
        path, _ = QFileDialog.getSaveFileName(self, "Save filtered FITS as", start, "FITS Files (*.fits)")
        if path:
            self._outpath = path
            self.out_label.setText(path)

    def _sync_weight_edit(self):
        val = self.weight_slider.value() / 100.0
        self.weight_edit.setText(f"{val:.2f}")

    def _sync_weight_slider(self):
        try:
            v = float(self.weight_edit.text())
            v = max(0.0, min(1.0, v))
            self.weight_slider.setValue(int(round(v * 100)))
        except Exception:
            self.weight_edit.setText(f"{self.weight_slider.value()/100.0:.2f}")

    def _run_filter(self):
        if not self._inpath:
            QMessageBox.warning(self, "Missing input", "Please select an input FITS file.")
            return
        if not self._outpath:
            QMessageBox.warning(self, "Missing output", "Please select an output path.")
            return
        try:
            radius = float(self.radius_edit.text())
            percentile = float(self.percentile_edit.text())
            weight = float(self.weight_edit.text())
            if radius <= 0:
                raise ValueError("Radius must be > 0")
            if not (0.0 <= percentile <= 100.0):
                raise ValueError("Percentile must be 0..100")
            if not (0.0 <= weight <= 1.0):
                raise ValueError("Weight must be 0..1")
        except Exception as e:
            QMessageBox.critical(self, "Parameter error", str(e))
            return

        # disable UI while running to prevent re-entrancy
        self.btn_run.setEnabled(False)
        try:
            rank_filter_fits(self._inpath, self._outpath, radius=radius, percentile=percentile, weight=weight)
            QMessageBox.information(self, "Done", f"Filtered FITS written to:\n{self._outpath}")
        except Exception as e:
            QMessageBox.critical(self, "Processing error", str(e))
        finally:
            self.btn_run.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RankFilterWindow()
    win.show()
    sys.exit(app.exec())