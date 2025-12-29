#!/usr/bin/env python3
"""
fits_wcs_plotter_gui.py
PyQt6 GUI to load a 3-plane FITS cube (RGB), preview it with WCS axes and robust normalization,
and save the plotted figure to an image file.

This version adds editable controls for rescale parameters:
- Percentile and Target (%) for percentile rescale
- Asinh parameter for asinh compression
- Masked-core percentile and Target (%) for masked-core attenuation
"""
import sys
import traceback
from pathlib import Path
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import (
    ImageNormalize, ZScaleInterval, PercentileInterval, AsinhStretch, LogStretch
)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QMessageBox, QDoubleSpinBox,
    QSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import imageio
from PIL import Image

# ---------- helper functions ----------
def read_fits_rgb(path):
    with fits.open(path) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header
    if data is None:
        raise ValueError("No data in primary HDU")
    arr = np.asarray(data)
    if arr.ndim == 3:
        if arr.shape[0] == 3:
            rgb = np.transpose(arr, (1, 2, 0))
        elif arr.shape[2] == 3:
            rgb = arr
        else:
            if arr.shape[0] >= 3:
                rgb = np.transpose(arr[:3], (1,2,0))
            elif arr.shape[2] >= 3:
                rgb = arr[..., :3]
            else:
                raise ValueError(f"Unsupported 3D shape for RGB: {arr.shape}")
    elif arr.ndim == 2:
        rgb = np.stack([arr]*3, axis=-1)
    else:
        raise ValueError(f"Unsupported data ndim: {arr.ndim}")
    return rgb.astype(np.float64), hdr

def _safe_limits(ch, use_zscale=True, perc=99.5):
    if ch.size == 0:
        return 0.0, 1.0
    try:
        if use_zscale:
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(ch)
        else:
            interval = PercentileInterval(perc)
            vmin, vmax = interval.get_limits(ch)
    except Exception:
        vmin = np.nanpercentile(ch, 0.5)
        vmax = np.nanpercentile(ch, perc)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        med = np.nanmedian(ch)
        mad = np.nanmedian(np.abs(ch - med))
        if mad <= 0 or not np.isfinite(mad):
            span = max(1.0, abs(med) * 0.01)
        else:
            span = max(mad * 6.0, 1e-6)
        vmin = med - span/2.0
        vmax = med + span/2.0
    return float(vmin), float(vmax)

def normalize_rgb_for_display(rgb, mode="Linear", param=1.0, use_zscale=True, perc=99.5):
    out = np.array(rgb, dtype=np.float64)
    normed = np.zeros_like(out, dtype=np.float32)
    if mode == "Asinh":
        stretch = AsinhStretch(a=param)
    elif mode == "Log":
        stretch = LogStretch()
    else:
        stretch = None
    for c in range(3):
        ch = out[..., c]
        finite_mask = np.isfinite(ch)
        if not finite_mask.any():
            normed[..., c] = 0.0
            continue
        finite_vals = ch[finite_mask]
        vmin, vmax = _safe_limits(finite_vals, use_zscale=use_zscale, perc=perc)
        if stretch is None:
            normalizer = ImageNormalize(vmin=vmin, vmax=vmax, clip=False)
        else:
            normalizer = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch, clip=False)
        med = float(np.nanmedian(finite_vals))
        ch_safe = np.array(ch, dtype=np.float64)
        ch_safe[~finite_mask] = med
        mapped = normalizer(ch_safe)
        mapped = np.nan_to_num(mapped, nan=0.0, posinf=1.0, neginf=0.0)
        normed[..., c] = np.clip(mapped, 0.0, 1.0)
    return normed

def rgba_to_rgb_uint8(rgba):
    arr = np.asarray(rgba)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    if arr.ndim == 2:
        return np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 3:
        return arr
    alpha = arr[..., 3].astype(np.float32) / 255.0
    rgb = arr[..., :3].astype(np.float32)
    bg = 255.0
    comp = (rgb * alpha[..., None] + bg * (1.0 - alpha[..., None]))
    return np.clip(comp.round(), 0, 255).astype(np.uint8)

# ---------- Rescale methods ----------
def rescale_percentile_to_target(img01, percentile=99.5, target=0.95):
    flat = img01.reshape(-1, img01.shape[-1])
    per_pixel = np.max(flat, axis=1)
    pval = np.percentile(per_pixel, percentile)
    if pval <= 0:
        return img01.copy()
    scale = target / float(pval)
    return np.clip(img01 * scale, 0.0, 1.0)

def rescale_asinh(img01, param=1.0):
    p = max(1e-6, float(param))
    out = np.arcsinh(p * img01) / np.arcsinh(p)
    return np.clip(out, 0.0, 1.0)

def rescale_masked_core(img01, core_percentile=99.9, target=0.95):
    if img01.size == 0:
        return img01.copy()
    per_pixel = np.max(img01.reshape(-1, img01.shape[-1]), axis=1)
    thresh = np.percentile(per_pixel, core_percentile)
    if not np.isfinite(thresh) or thresh <= 0:
        return img01.copy()
    mask = np.max(img01, axis=2) > thresh
    if not np.any(mask):
        return img01.copy()
    mmax = float(np.max(img01[mask]))
    if mmax <= 0:
        return img01.copy()
    scale = target / mmax
    out = img01.copy()
    out[mask] = np.clip(out[mask] * scale, 0.0, 1.0)
    return out

# ---------- Matplotlib canvas ----------
class WcsCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=(6,6), dpi=100):
        fig = Figure(figsize=figsize, dpi=dpi)
        super().__init__(fig)
        self.fig = fig
        self.ax = None

    def plot_rgb_with_wcs(self, rgb01, header, title=""):
        self.fig.clf()
        rgb_disp = np.asarray(rgb01, dtype=np.float32)
        rgb_disp = np.nan_to_num(rgb_disp, nan=0.0, posinf=1.0, neginf=0.0)
        rgb_disp = np.clip(rgb_disp, 0.0, 1.0)
        try:
            wcs = WCS(header, naxis=2)
            ax = self.fig.add_subplot(1, 1, 1, projection=wcs)
            ax.imshow(rgb_disp, origin="lower", aspect="equal", interpolation="nearest", vmin=0.0, vmax=1.0)
            ax.coords.grid(True, color="white", ls="dotted")
            ax.set_xlabel("Right Ascension")
            ax.set_ylabel("Declination")
            ax.set_title(title)
            self.ax = ax
        except Exception:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.imshow(rgb_disp, origin="lower", aspect="equal", interpolation="nearest", vmin=0.0, vmax=1.0)
            ax.set_title(title + " (no valid WCS)")
            self.ax = ax
        self.draw()

# ---------- GUI ----------
class FitsWcsPlotterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS WCS RGB Plotter")
        self._build_ui()
        self.resize(960, 700)
        self.rgb = None
        self.header = None
        self.input_path = None
        self.last_preview = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        grid.addWidget(QLabel("FITS WCS file:"), 0, 0)
        self.fits_edit = QLineEdit()
        grid.addWidget(self.fits_edit, 0, 1, 1, 3)
        btn_fits = QPushButton("Browse")
        btn_fits.clicked.connect(self._browse_fits)
        grid.addWidget(btn_fits, 0, 4)

        grid.addWidget(QLabel("Plot title:"), 1, 0)
        self.title_edit = QLineEdit()
        grid.addWidget(self.title_edit, 1, 1, 1, 3)

        grid.addWidget(QLabel("Save plot as:"), 2, 0)
        self.out_edit = QLineEdit()
        grid.addWidget(self.out_edit, 2, 1, 1, 3)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_output)
        grid.addWidget(btn_out, 2, 4)

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

        grid.addWidget(QLabel("Max pixel dimension (px):"), 4, 0)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(100, 20000)
        self.size_spin.setSingleStep(100)
        self.size_spin.setValue(5000)
        grid.addWidget(self.size_spin, 4, 1)

        grid.addWidget(QLabel("Export DPI (0 = auto):"), 4, 2)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(0, 2400)
        self.dpi_spin.setValue(0)
        grid.addWidget(self.dpi_spin, 4, 3)

        grid.addWidget(QLabel("High-res preview:"), 5, 0)
        self.preview_check = QCheckBox("Enable")
        self.preview_check.stateChanged.connect(self._toggle_preview_dpi)
        grid.addWidget(self.preview_check, 5, 1)

        grid.addWidget(QLabel("Save high-bit image:"), 5, 2)
        self.highbit_check = QCheckBox("Save 16-bit TIFF / FITS (when enabled)")
        self.highbit_check.setChecked(True)
        grid.addWidget(self.highbit_check, 5, 3)

        # Rescale controls
        grid.addWidget(QLabel("Enable preview rescale:"), 6, 0)
        self.enable_rescale_check = QCheckBox("Enable")
        self.enable_rescale_check.setChecked(False)
        grid.addWidget(self.enable_rescale_check, 6, 1)

        grid.addWidget(QLabel("Rescale method:"), 6, 2)
        self.rescale_method_combo = QComboBox()
        self.rescale_method_combo.addItems([
            "Percentile 99.5 -> 0.95",
            "Asinh compression",
            "Masked core attenuation"
        ])
        grid.addWidget(self.rescale_method_combo, 6, 3)

        # Percentile controls
        grid.addWidget(QLabel("Percentile (%):"), 7, 0)
        self.percentile_spin = QDoubleSpinBox()
        self.percentile_spin.setRange(50.0, 100.0)
        self.percentile_spin.setDecimals(2)
        self.percentile_spin.setValue(99.5)
        grid.addWidget(self.percentile_spin, 7, 1)

        grid.addWidget(QLabel("Percentile target (%):"), 7, 2)
        self.percentile_target_spin = QDoubleSpinBox()
        self.percentile_target_spin.setRange(1.0, 100.0)
        self.percentile_target_spin.setDecimals(2)
        self.percentile_target_spin.setValue(95.0)
        grid.addWidget(self.percentile_target_spin, 7, 3)

        # Asinh parameter
        grid.addWidget(QLabel("Asinh param:"), 8, 0)
        self.asinh_param_spin = QDoubleSpinBox()
        self.asinh_param_spin.setRange(0.01, 100.0)
        self.asinh_param_spin.setDecimals(3)
        self.asinh_param_spin.setValue(1.0)
        grid.addWidget(self.asinh_param_spin, 8, 1)

        # Masked core controls
        grid.addWidget(QLabel("Masked core percentile (%):"), 8, 2)
        self.masked_percentile_spin = QDoubleSpinBox()
        self.masked_percentile_spin.setRange(90.0, 100.0)
        self.masked_percentile_spin.setDecimals(2)
        self.masked_percentile_spin.setValue(99.9)
        grid.addWidget(self.masked_percentile_spin, 8, 3)

        grid.addWidget(QLabel("Masked target (%):"), 9, 0)
        self.masked_target_spin = QDoubleSpinBox()
        self.masked_target_spin.setRange(1.0, 100.0)
        self.masked_target_spin.setDecimals(2)
        self.masked_target_spin.setValue(95.0)
        grid.addWidget(self.masked_target_spin, 9, 1)

        self.rescale_btn = QPushButton("Rescale preview now")
        self.rescale_btn.clicked.connect(self._rescale_preview_button)
        grid.addWidget(self.rescale_btn, 9, 2, 1, 2)

        # Buttons
        self.load_btn = QPushButton("Load FITS & Preview")
        self.load_btn.clicked.connect(self._load_and_preview)
        grid.addWidget(self.load_btn, 10, 0, 1, 2)

        self.plot_btn = QPushButton("Plot & Save")
        self.plot_btn.clicked.connect(self._plot_and_save)
        grid.addWidget(self.plot_btn, 10, 2, 1, 2)

        # Canvas
        self.canvas = WcsCanvas(figsize=(6,6), dpi=110)
        grid.addWidget(self.canvas, 11, 0, 1, 5)

        # Log
        grid.addWidget(QLabel("Log:"), 12, 0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(120)
        grid.addWidget(self.log, 13, 0, 1, 5)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_fits(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS WCS file", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.fits_edit.setText(fn)

    def _browse_output(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save plot as", "", "PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tif *.tiff);;FITS (*.fits);;All Files (*)")
        if fn:
            self.out_edit.setText(fn)

    def _prepare_input_array(self, arr):
        a = np.asarray(arr)
        if a.ndim == 2:
            a = np.stack([a]*3, axis=-1)
        elif a.ndim == 3 and a.shape[2] != 3:
            if a.shape[0] == 3:
                a = np.transpose(a, (1,2,0))
            elif a.shape[2] >= 3:
                a = a[..., :3]
            else:
                a = np.stack([a[..., 0]]*3, axis=-1)
        a = a.astype(np.float64)
        for c in range(3):
            ch = a[..., c]
            finite = np.isfinite(ch)
            if not finite.any():
                a[..., c] = 0.0
            else:
                med = float(np.nanmedian(ch[finite]))
                ch[~finite] = med
                a[..., c] = ch
        return a

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

            mode = self.stretch_combo.currentText()
            param = float(self.param_spin.value())

            arr = self._prepare_input_array(self.rgb)
            rgb01 = normalize_rgb_for_display(arr, mode=mode, param=param, use_zscale=True, perc=99.9)
            if np.nanmax(rgb01) - np.nanmin(rgb01) < 1e-6:
                rgb01 = normalize_rgb_for_display(arr, mode=mode, param=param, use_zscale=False, perc=99.5)

            span = np.nanmax(rgb01) - np.nanmin(rgb01)
            if span < 1e-6:
                for c in range(3):
                    ch = rgb01[..., c]
                    mn = np.nanmin(ch)
                    mx = np.nanmax(ch)
                    if mx > mn:
                        rgb01[..., c] = (ch - mn) / (mx - mn)
                    else:
                        rgb01[..., c] = 0.0

            rgb01 = np.asarray(rgb01, dtype=np.float32)
            rgb01 = np.nan_to_num(rgb01, nan=0.0, posinf=1.0, neginf=0.0)
            rgb01 = np.clip(rgb01, 0.0, 1.0)

            self.last_preview = rgb01.copy()
            self.canvas.plot_rgb_with_wcs(self.last_preview, self.header, title=self.title_edit.text().strip() or Path(path).name)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"{e}\n\n{tb}")

    def _toggle_preview_dpi(self, state):
        try:
            fig = self.canvas.fig
            if state == Qt.CheckState.Checked.value:
                self._preview_orig_dpi = fig.get_dpi()
                new_dpi = min(600, int(self._preview_orig_dpi * 2))
                fig.set_dpi(new_dpi)
                self._log(f"High-res preview enabled (dpi={new_dpi})")
            else:
                if hasattr(self, "_preview_orig_dpi"):
                    fig.set_dpi(self._preview_orig_dpi)
                    self._log(f"High-res preview disabled (dpi restored={self._preview_orig_dpi})")
            self.canvas.draw()
        except Exception as e:
            self._log("Preview DPI toggle error:", e)

    def _apply_rescale_preview(self, method=None):
        if self.last_preview is None:
            self._log("No preview available to rescale.")
            return
        if method is None:
            method = self.rescale_method_combo.currentText()
        img = self.last_preview.copy()
        try:
            if method.startswith("Percentile"):
                perc = float(self.percentile_spin.value())
                target_pct = float(self.percentile_target_spin.value())
                target = max(0.0, min(1.0, target_pct / 100.0))
                img = rescale_percentile_to_target(img, percentile=perc, target=target)
                self._log(f"Applied percentile rescale ({perc}% -> {target_pct}%).")
            elif method.startswith("Asinh"):
                param = float(self.asinh_param_spin.value())
                img = rescale_asinh(img, param=param)
                self._log(f"Applied asinh compression (param={param}).")
            elif method.startswith("Masked"):
                core_perc = float(self.masked_percentile_spin.value())
                target_pct = float(self.masked_target_spin.value())
                target = max(0.0, min(1.0, target_pct / 100.0))
                img = rescale_masked_core(img, core_percentile=core_perc, target=target)
                self._log(f"Applied masked core attenuation ({core_perc}% -> {target_pct}%).")
            else:
                self._log("Unknown rescale method; no change.")
                return
            self.last_preview = np.clip(img, 0.0, 1.0).astype(np.float32)
            title = self.title_edit.text().strip() or (Path(self.input_path).name if self.input_path else "")
            self.canvas.plot_rgb_with_wcs(self.last_preview, self.header, title=title)
        except Exception as e:
            self._log("Error applying rescale:", e)

    def _rescale_preview_button(self):
        try:
            if not self.enable_rescale_check.isChecked():
                self.enable_rescale_check.setChecked(True)
            self._apply_rescale_preview()
        except Exception as e:
            self._log("Rescale button error:", e)

    def _plot_and_save(self):
        if self.rgb is None or self.header is None:
            self._load_and_preview()
            if self.rgb is None:
                return
        outpath = self.out_edit.text().strip()
        if not outpath:
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

            arr = self._prepare_input_array(self.rgb)
            rgb01 = normalize_rgb_for_display(arr, mode=mode, param=param, use_zscale=True, perc=99.9)
            if np.nanmax(rgb01) - np.nanmin(rgb01) < 1e-6:
                rgb01 = normalize_rgb_for_display(arr, mode=mode, param=param, use_zscale=False, perc=99.5)
            span = np.nanmax(rgb01) - np.nanmin(rgb01)
            if span < 1e-6:
                for c in range(3):
                    ch = rgb01[..., c]
                    mn = np.nanmin(ch)
                    mx = np.nanmax(ch)
                    if mx > mn:
                        rgb01[..., c] = (ch - mn) / (mx - mn)
                    else:
                        rgb01[..., c] = 0.0

            rgb01 = np.asarray(rgb01, dtype=np.float32)
            rgb01 = np.nan_to_num(rgb01, nan=0.0, posinf=1.0, neginf=0.0)
            rgb01 = np.clip(rgb01, 0.0, 1.0)

            preview_for_figure = rgb01
            if self.enable_rescale_check.isChecked():
                if self.last_preview is not None:
                    preview_for_figure = self.last_preview
                else:
                    method = self.rescale_method_combo.currentText()
                    if method.startswith("Percentile"):
                        perc = float(self.percentile_spin.value())
                        target_pct = float(self.percentile_target_spin.value())
                        target = max(0.0, min(1.0, target_pct / 100.0))
                        preview_for_figure = rescale_percentile_to_target(rgb01, percentile=perc, target=target)
                    elif method.startswith("Asinh"):
                        param_asinh = float(self.asinh_param_spin.value())
                        preview_for_figure = rescale_asinh(rgb01, param=param_asinh)
                    elif method.startswith("Masked"):
                        core_perc = float(self.masked_percentile_spin.value())
                        target_pct = float(self.masked_target_spin.value())
                        target = max(0.0, min(1.0, target_pct / 100.0))
                        preview_for_figure = rescale_masked_core(rgb01, core_percentile=core_perc, target=target)
                    preview_for_figure = np.clip(preview_for_figure, 0.0, 1.0).astype(np.float32)

            self.canvas.plot_rgb_with_wcs(preview_for_figure, self.header, title=title)

            # Save logic (unchanged from previous version)...
            # For brevity, reuse the same saving approach as before:
            fig = self.canvas.fig
            orig_size = fig.get_size_inches().copy()
            orig_dpi = fig.get_dpi()
            max_pixels = int(self.size_spin.value())
            MAX_SAFE = 20000
            if max_pixels > MAX_SAFE:
                self._log(f"Requested max_pixels {max_pixels} exceeds safe limit {MAX_SAFE}; clamping.")
                max_pixels = MAX_SAFE
                self.size_spin.setValue(MAX_SAFE)
            user_dpi = int(self.dpi_spin.value())
            if user_dpi > 0:
                dpi = user_dpi
            else:
                max_inches = max(orig_size[0], orig_size[1])
                dpi = 150 if max_inches <= 0 else max(1, int(round(max_pixels / max_inches)))
            target_inches = max_pixels / dpi
            scale = target_inches / max(orig_size[0], orig_size[1])
            new_size = orig_size * scale
            fig.set_size_inches(new_size, forward=True)
            fig.set_dpi(dpi)
            self.canvas.draw()

            out_lower = outpath.lower()
            if out_lower.endswith(".png"):
                preview_path = outpath
                fig.savefig(preview_path, dpi=dpi, bbox_inches="tight")
                self._log(f"Saved preview PNG to {preview_path} (dpi={dpi})")
            else:
                preview_path = str(Path(outpath).with_suffix(".preview.png"))
                fig.savefig(preview_path, dpi=dpi, bbox_inches="tight")
                self._log(f"Saved preview PNG to {preview_path} (dpi={dpi})")

            # Convert preview to requested formats (TIFF/JPEG/FITS) as before
            if out_lower.endswith((".tif", ".tiff")):
                try:
                    img8 = imageio.imread(preview_path)
                    if img8.ndim == 3 and img8.shape[2] == 4:
                        img8 = rgba_to_rgb_uint8(img8)
                    imageio.imwrite(outpath, img8)
                    self._log(f"Wrote TIFF with axes/title to {outpath}")
                except Exception as e:
                    self._log("Failed to write TIFF preview:", e)

            if out_lower.endswith(".fits"):
                try:
                    fits_data = np.transpose((rgb01).astype(np.float32), (2, 0, 1))
                    fits.writeto(outpath, fits_data, header=self.header, overwrite=True)
                    self._log(f"Saved FITS (float32) to {outpath}")
                except Exception as e:
                    self._log("Failed to write FITS:", e)

            if out_lower.endswith((".jpg", ".jpeg")):
                try:
                    img8 = imageio.imread(preview_path)
                    if img8.ndim == 3 and img8.shape[2] == 4:
                        img8 = rgba_to_rgb_uint8(img8)
                    pil = Image.fromarray(img8)
                    if pil.mode == 'RGBA':
                        pil = pil.convert('RGB')
                    pil.save(outpath, format='JPEG', quality=95)
                    self._log(f"Saved JPEG to {outpath}")
                except Exception as e:
                    self._log("Failed to write JPEG:", e)

            if out_lower not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".fits"):
                try:
                    img8 = imageio.imread(preview_path)
                    if img8.ndim == 3 and img8.shape[2] == 4:
                        img8 = rgba_to_rgb_uint8(img8)
                    imageio.imwrite(outpath, img8)
                    self._log(f"Saved raster plot to {outpath} (dpi={dpi})")
                except Exception as e:
                    self._log("Failed to write default raster output:", e)

            # High-bit saving (image data only)
            if self.highbit_check.isChecked():
                try:
                    base = Path(outpath)
                    hb_norm = normalize_rgb_for_display(arr, mode=mode, param=param, use_zscale=True, perc=99.9)
                    if np.nanmax(hb_norm) - np.nanmin(hb_norm) < 1e-6:
                        hb_norm = normalize_rgb_for_display(arr, mode=mode, param=param, use_zscale=False, perc=99.5)
                    hb_norm = np.asarray(hb_norm, dtype=np.float32)
                    hb_norm = np.nan_to_num(hb_norm, nan=0.0, posinf=1.0, neginf=0.0)
                    hb_norm = np.clip(hb_norm, 0.0, 1.0)
                    if out_lower.endswith((".tif", ".tiff")):
                        hb_path = str(base.with_suffix("")) + "_16bit.tif"
                        img16 = (hb_norm * 65535.0).round().astype(np.uint16)
                        imageio.imwrite(hb_path, img16)
                        self._log(f"Saved sibling 16-bit TIFF (image data only) to {hb_path}")
                    elif out_lower.endswith(".fits"):
                        hb_path = str(base.with_suffix("")) + "_image.fits"
                        fits_data = np.transpose((hb_norm).astype(np.float32), (2, 0, 1))
                        fits.writeto(hb_path, fits_data, header=self.header, overwrite=True)
                        self._log(f"Saved FITS (float32 image data) to {hb_path}")
                    else:
                        hb_path = str(base.with_suffix("")) + "_16bit.tif"
                        img16 = (hb_norm * 65535.0).round().astype(np.uint16)
                        imageio.imwrite(hb_path, img16)
                        self._log(f"Saved sibling 16-bit TIFF to {hb_path}")
                    QMessageBox.information(self, "Saved", f"Saved preview and high-bit image(s).\nPreview: {preview_path}\nHigh-bit: {hb_path}")
                except Exception as e:
                    self._log("High-bit save failed:", e)
                    QMessageBox.warning(self, "High-bit save failed", f"High-bit save failed:\n{e}")
            else:
                QMessageBox.information(self, "Saved", f"Saved preview: {preview_path}")

            fig.set_size_inches(orig_size, forward=True)
            fig.set_dpi(orig_dpi)
            self.canvas.draw()
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