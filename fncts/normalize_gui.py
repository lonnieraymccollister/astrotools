#!/usr/bin/env python3
"""
normalize_gui_fixed.py
PyQt6 GUI to linearly rescale FITS image data from [old_min, old_max] -> [new_min, new_max].
Adds robust 2D/3D detection, channel ordering handling, per-channel normalization,
output options compatible with Siril / PixInsight / Maxim-DL, and an Archive/high-precision option.
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
    QTextEdit, QCheckBox, QComboBox, QGroupBox, QRadioButton
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------- Helpers ----------
def load_fits(path):
    with fits.open(path, memmap=False) as hdul:
        return hdul[0].data, hdul[0].header

def write_fits(path, data, header=None, out_dtype=np.float32):
    out = data.astype(out_dtype, copy=False)
    hdr = header.copy() if header is not None else fits.Header()
    # Ensure BITPIX matches dtype
    hdr['BITPIX'] = -64 if out_dtype == np.float64 else -32
    # NORMED may be set by caller; keep existing or set later
    fits.writeto(path, out, header=hdr, overwrite=True)

def compute_stats(arr):
    a = np.asarray(arr).ravel()
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"min": np.nan, "max": np.nan, "median": np.nan, "mean": np.nan, "std": np.nan}
    return {"min": float(a.min()), "max": float(a.max()),
            "median": float(np.median(a)), "mean": float(a.mean()), "std": float(a.std())}

def normalize_channel(arr, old_min, old_max, new_min, new_max):
    a = arr.astype(np.float64)
    if old_max == old_min:
        scaled = np.full_like(a, new_min, dtype=np.float64)
    else:
        scaled = (a - old_min) / (old_max - old_min)
        scaled = scaled * (new_max - new_min) + new_min
    return np.clip(scaled, min(new_min, new_max), max(new_min, new_max))

def normalize_array(data, old_min, old_max, new_min, new_max, per_channel=False, channel_axis=0):
    arr = data.astype(np.float64)
    if arr.ndim == 2 or not per_channel:
        return normalize_channel(arr, old_min, old_max, new_min, new_max)
    if channel_axis == 0:
        C, H, W = arr.shape
        out = np.empty_like(arr, dtype=np.float64)
        for c in range(C):
            st = compute_stats(arr[c])
            omin = old_min if old_min is not None else st['min']
            omax = old_max if old_max is not None else st['max']
            out[c] = normalize_channel(arr[c], omin, omax, new_min, new_max)
        return out
    else:
        H, W, C = arr.shape
        out = np.empty_like(arr, dtype=np.float64)
        for c in range(C):
            plane = arr[:, :, c]
            st = compute_stats(plane)
            omin = old_min if old_min is not None else st['min']
            omax = old_max if old_max is not None else st['max']
            out[:, :, c] = normalize_channel(plane, omin, omax, new_min, new_max)
        return out

# ---------- Matplotlib canvas ----------
class HistCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(8,3))
        super().__init__(fig)
        self.axes = [fig.add_subplot(1,3,i+1) for i in range(3)]
        fig.tight_layout()

    def plot_channels(self, data, channel_axis=0, bins=256):
        for ax in self.axes:
            ax.clear()
        if data is None:
            for i, ax in enumerate(self.axes):
                ax.set_title(f"Ch {i}: none")
            self.draw()
            return
        arr = np.asarray(data)
        if arr.ndim == 2:
            self.axes[1].hist(arr.ravel(), bins=bins, color='gray', histtype='step')
            self.axes[1].set_title("Gray")
            self.axes[0].set_title("Ch 0")
            self.axes[2].set_title("Ch 2")
        else:
            if channel_axis == 0:
                C = arr.shape[0]
                for i in range(3):
                    if i < C:
                        self.axes[i].hist(arr[i].ravel(), bins=bins, color=['red','green','blue'][i], histtype='step')
                        self.axes[i].set_title(f"Ch {i}")
                    else:
                        self.axes[i].set_title(f"Ch {i}: none")
            else:
                C = arr.shape[2]
                for i in range(3):
                    if i < C:
                        self.axes[i].hist(arr[:, :, i].ravel(), bins=bins, color=['red','green','blue'][i], histtype='step')
                        self.axes[i].set_title(f"Ch {i}")
                    else:
                        self.axes[i].set_title(f"Ch {i}: none")
        self.draw()

# ---------- Main window ----------
class NormalizeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Normalize FITS (fixed)")
        self.image = None
        self.header = None
        self.channel_axis = None
        self._build_ui()
        self.resize(1150, 700)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        grid.addWidget(QLabel("Input FITS:"), 0, 0)
        self.input_edit = QLineEdit()
        grid.addWidget(self.input_edit, 0, 1, 1, 3)
        btn_in = QPushButton("Browse...")
        btn_in.clicked.connect(self._browse_input)
        grid.addWidget(btn_in, 0, 4)

        grid.addWidget(QLabel("Output FITS:"), 1, 0)
        self.output_edit = QLineEdit("normalized_output.fits")
        grid.addWidget(self.output_edit, 1, 1, 1, 3)
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(self._browse_output)
        grid.addWidget(btn_out, 1, 4)

        ch_group = QGroupBox("Channel handling / detection")
        ch_layout = QHBoxLayout()
        ch_group.setLayout(ch_layout)
        self.detect_btn = QRadioButton("Auto-detect (recommended)")
        self.detect_btn.setChecked(True)
        self.cf_btn = QRadioButton("Channel-first (C,H,W)")
        self.cl_btn = QRadioButton("Channel-last (H,W,C)")
        ch_layout.addWidget(self.detect_btn)
        ch_layout.addWidget(self.cf_btn)
        ch_layout.addWidget(self.cl_btn)
        grid.addWidget(ch_group, 2, 0, 1, 5)

        grid.addWidget(QLabel("Old min:"), 3, 0)
        self.old_min = QDoubleSpinBoxWithLargeRange()
        grid.addWidget(self.old_min, 3, 1)
        grid.addWidget(QLabel("Old max:"), 3, 2)
        self.old_max = QDoubleSpinBoxWithLargeRange()
        grid.addWidget(self.old_max, 3, 3)

        grid.addWidget(QLabel("New min:"), 4, 0)
        self.new_min = QDoubleSpinBoxWithLargeRange()
        grid.addWidget(self.new_min, 4, 1)
        grid.addWidget(QLabel("New max:"), 4, 2)
        self.new_max = QDoubleSpinBoxWithLargeRange()
        grid.addWidget(self.new_max, 4, 3)

        self.per_channel_chk = QCheckBox("Normalize per channel (if 3D)")
        self.per_channel_chk.setChecked(True)
        grid.addWidget(self.per_channel_chk, 5, 0, 1, 2)

        grid.addWidget(QLabel("Output convention:"), 5, 2)
        self.output_conv = QComboBox()
        self.output_conv.addItems(["Auto (preserve input layout)", "PixInsight (3,H,W)", "Siril/Maxim-DL (H,W,3)"])
        grid.addWidget(self.output_conv, 5, 3)

        # New: Output dtype selection (float32 default, float64 Archive)
        grid.addWidget(QLabel("Output dtype:"), 6, 0)
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["float32 (default)", "float64 (Archive / high precision)"])
        self.dtype_combo.setCurrentIndex(0)
        grid.addWidget(self.dtype_combo, 6, 1, 1, 2)

        self.preview_btn = QPushButton("Preview Stats/Histogram")
        self.preview_btn.clicked.connect(self._preview)
        grid.addWidget(self.preview_btn, 7, 0)

        self.normalize_btn = QPushButton("Run Normalize & Save")
        self.normalize_btn.clicked.connect(self._run_normalize)
        grid.addWidget(self.normalize_btn, 7, 1)

        self.plot_output_btn = QPushButton("Plot Output File")
        self.plot_output_btn.clicked.connect(self._plot_output_file)
        grid.addWidget(self.plot_output_btn, 7, 2)

        self.siril_chk = QCheckBox("Offer to open in Siril after save")
        self.siril_chk.setChecked(False)
        grid.addWidget(self.siril_chk, 7, 3, 1, 2)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setFixedHeight(140)
        grid.addWidget(self.info, 8, 0, 1, 5)

        self.canvas = HistCanvas()
        grid.addWidget(self.canvas, 9, 0, 9, 5)

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
                self.old_min.setValue(st['min'])
                self.old_max.setValue(st['max'])
                self.new_min.setValue(0.0)
                self.new_max.setValue(1.0)
                self._detect_channel_layout()
                if self.image.ndim == 2:
                    self.canvas.plot_channels(self.image)
                else:
                    self.canvas.plot_channels(self.image, channel_axis=self.channel_axis if self.channel_axis is not None else 0)
            except Exception as e:
                QMessageBox.critical(self, "Load error", f"Failed to load FITS: {e}")

    def _detect_channel_layout(self):
        if self.image is None:
            self.channel_axis = None
            return
        arr = self.image
        if arr.ndim == 2:
            self.channel_axis = None
            self.info.append("Detected 2D grayscale FITS.")
            return
        shape = arr.shape
        if shape[0] == 3:
            self.channel_axis = 0
            self.info.append("Detected channel-first layout (3,H,W).")
            self.cf_btn.setChecked(True)
        elif shape[-1] == 3:
            self.channel_axis = -1
            self.info.append("Detected channel-last layout (H,W,3).")
            self.cl_btn.setChecked(True)
        else:
            self.channel_axis = 0
            self.info.append(f"Ambiguous 3D shape {shape}; defaulting to channel-first (C,H,W). If incorrect, choose Channel-last and re-run preview.")
            self.detect_btn.setChecked(False)
            self.cf_btn.setChecked(True)

    def _browse_output(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save FITS as", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def _get_channel_axis_from_ui(self):
        if self.detect_btn.isChecked():
            return self.channel_axis if self.channel_axis is not None else 0
        if self.cf_btn.isChecked():
            return 0
        if self.cl_btn.isChecked():
            return -1
        return self.channel_axis if self.channel_axis is not None else 0

    def _preview(self):
        if self.image is None:
            QMessageBox.warning(self, "No image", "Load an input FITS first")
            return
        try:
            old_min = float(self.old_min.value())
            old_max = float(self.old_max.value())
            new_min = float(self.new_min.value())
            new_max = float(self.new_max.value())
            per_channel = bool(self.per_channel_chk.isChecked())
            channel_axis = self._get_channel_axis_from_ui()
            preview = normalize_array(self.image, old_min, old_max, new_min, new_max,
                                      per_channel=per_channel and self.image.ndim==3,
                                      channel_axis=channel_axis)
            pst = compute_stats(preview if preview is not None else self.image)
            self.info.append(f"Preview stats: min={pst['min']:.6g} max={pst['max']:.6g} mean={pst['mean']:.6g} std={pst['std']:.6g}")
            if preview.ndim == 2:
                self.canvas.plot_channels(preview)
            else:
                self.canvas.plot_channels(preview, channel_axis=channel_axis)
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
            per_channel = bool(self.per_channel_chk.isChecked())
            channel_axis = self._get_channel_axis_from_ui()
            result = normalize_array(self.image, old_min, old_max, new_min, new_max,
                                     per_channel=per_channel and self.image.ndim==3,
                                     channel_axis=channel_axis)
            conv = self.output_conv.currentText()
            out = result
            if conv.startswith("PixInsight"):
                if out.ndim == 2:
                    out = out[np.newaxis, :, :]
                elif out.ndim == 3:
                    if channel_axis == -1:
                        out = np.transpose(out, (2,0,1))
            elif conv.startswith("Siril"):
                if out.ndim == 2:
                    out = out.astype(np.float32)
                elif out.ndim == 3:
                    if channel_axis == 0:
                        out = np.transpose(out, (1,2,0))
            # Determine dtype from UI
            dtype_text = self.dtype_combo.currentText()
            out_dtype = np.float64 if "float64" in dtype_text else np.float32

            # Prepare header: set NORMED and BITPIX
            hdr = self.header.copy() if self.header is not None else fits.Header()
            hdr['NORMED'] = (1, "Output normalized to [new_min,new_max]")
            hdr['BITPIX'] = -64 if out_dtype == np.float64 else -32

            # Write file
            write_fits(outpath, out, header=hdr, out_dtype=out_dtype)
            self.info.append(f"Saved normalized FITS to {outpath} shape={out.shape} dtype={out_dtype}")
            if out.ndim == 2:
                self.canvas.plot_channels(out)
            else:
                if conv.startswith("PixInsight"):
                    self.canvas.plot_channels(out, channel_axis=0)
                elif conv.startswith("Siril"):
                    self.canvas.plot_channels(out, channel_axis=-1)
                else:
                    self.canvas.plot_channels(out, channel_axis=0 if out.shape[0]==3 else -1)
            if self.siril_chk.isChecked():
                self._maybe_open_siril(outpath)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

    def _plot_output_file(self):
        outpath = self.output_edit.text().strip() or "normalized_output.fits"
        if not os.path.exists(outpath):
            fn, _ = QFileDialog.getOpenFileName(self, "Select output FITS to plot", "", "FITS Files (*.fits *.fit);;All Files (*)")
            if not fn:
                return
            outpath = fn
        try:
            data, hdr = load_fits(outpath)
            if data is None:
                raise ValueError("No data in primary HDU")
            if data.ndim > 2:
                if data.shape[0] == 3:
                    data2 = data
                    axis = 0
                elif data.shape[-1] == 3:
                    data2 = data
                    axis = -1
                else:
                    data2 = data[0].astype(np.float64)
                    axis = 0
            else:
                data2 = data.astype(np.float64)
                axis = None
            self.canvas.plot_channels(data2, channel_axis=axis if axis is not None else 0)
            st = compute_stats(data2 if data2.ndim==2 else data2)
            self.info.append(f"Plotted output FITS: {os.path.basename(outpath)} stats: min={st['min']:.6g} max={st['max']:.6g} mean={st['mean']:.6g} std={st['std']:.6g}")
        except Exception as e:
            QMessageBox.critical(self, "Plot error", f"Failed to load/plot output FITS: {e}")

    def _maybe_open_siril(self, fits_path):
        if shutil.which('siril') is None:
            QMessageBox.information(self, "Siril not found", "siril not found on PATH; cannot open")
            return
        try:
            if sys.platform.startswith('win'):
                os.startfile(fits_path)
            else:
                import subprocess
                subprocess.Popen(['siril', fits_path])
            self.info.append("Launched Siril (or opened file with default) for " + fits_path)
        except Exception as e:
            QMessageBox.warning(self, "Failed to open", f"Failed to open in Siril: {e}")

class QDoubleSpinBoxWithLargeRange(QDoubleSpinBox):
    def __init__(self, default=0.0):
        super().__init__()
        self.setRange(-1e12, 1e12)
        self.setDecimals(9)
        self.setSingleStep(0.1)
        self.setValue(float(default))

def main():
    app = QApplication(sys.argv)
    w = NormalizeWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()