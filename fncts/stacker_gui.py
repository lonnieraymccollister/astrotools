#!/usr/bin/env python3
"""
stacker_gui_fixed.py

Corrected PyQt6 GUI for stacking FITS images with optional reprojection,
fast sigma-clipped mean, robust shape handling, header sanitization,
background worker, and an optional weighted combine (per-pixel counts + sqrt scaling).
"""

import sys
import os
import glob
import traceback
import math
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from reproject import reproject_interp

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QCheckBox, QTextEdit, QMessageBox,
    QHBoxLayout, QVBoxLayout, QDoubleSpinBox, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# -------------------------
# FITS helpers (robust)
# -------------------------
def load_fits(path):
    """Return (data, header) for the first HDU that contains data."""
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.array(hdu.data), hdu.header
    raise ValueError(f"No image data in {path}")

def sanitize_header_for_float32(header):
    """Return a copy of header with BITPIX set to -32 and scaling keys removed."""
    hdr = header.copy()
    hdr['BITPIX'] = -32
    for k in ('BZERO', 'BSCALE'):
        if k in hdr:
            hdr.pop(k)
    return hdr

def normalize_fits_shape(arr):
    """
    Normalize FITS data to either:
      - 2D array: (H, W) for mono
      - 3D array: (3, H, W) for RGB (channels-first)
    Accepts common shapes: (H,W), (H,W,1), (1,H,W), (H,W,3), (3,H,W).
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        # (H,W,1) or (H,W,3)
        if a.shape[2] == 1:
            return a[:, :, 0]
        if a.shape[2] == 3:
            return np.transpose(a, (2, 0, 1))
        # (1,H,W) or (3,H,W)
        if a.shape[0] == 1:
            return a[0]
        if a.shape[0] == 3:
            return a
    # fallback: squeeze and re-evaluate
    s = np.squeeze(a)
    if s.ndim == 2:
        return s
    if s.ndim == 3 and s.shape[0] in (1, 3):
        if s.shape[0] == 1:
            return s[0]
        return s
    raise ValueError(f"Unsupported FITS shape: {a.shape}")

def to_luma(arr):
    """Return 2D luma for alignment/reprojection decisions."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return a.astype(np.float64)
    if a.ndim == 3 and a.shape[0] == 3:
        return np.mean(a.astype(np.float64), axis=0)
    raise ValueError(f"Unsupported shape for luma: {a.shape}")

# -------------------------
# Stacking helpers
# -------------------------
def compute_stack_mean(cube):
    return np.nanmean(cube, axis=0)

def compute_stack_median(cube):
    return np.nanmedian(cube, axis=0)

def compute_stack_sigma_clip(cube, sigma=3.0, maxiters=5):
    """
    Vectorized sigma-clipped mean along axis 0.
    Uses astropy.stats.sigma_clip which supports axis parameter.
    """
    # sigma_clip returns a MaskedArray; compute mean over unmasked values
    clipped = sigma_clip(cube, sigma=sigma, maxiters=maxiters, axis=0, masked=True)
    # Convert masked array to float with NaNs where masked
    arr = clipped.filled(np.nan)
    return np.nanmean(arr, axis=0)

def weighted_combine_from_cube(cube, mask_threshold=0.0):
    """
    Implements the combine behavior from the second program:
      - For each pixel, count contributions where pixel > mask_threshold
      - Compute per-pixel average = sum / count
      - Multiply per-pixel average by sqrt(max_count) where max_count is the
        maximum contributions across the canvas
    Returns final_image (float64), weight_map (float64)
    """
    # cube shape: (N, Y, X) or (N, 3, Y, X) - we only support mono cube here
    if cube.ndim != 3:
        raise ValueError("weighted_combine_from_cube expects a mono cube with shape (N, Y, X)")
    # mask contributions
    finite_mask = np.isfinite(cube)
    contrib_mask = finite_mask & (cube > mask_threshold)
    weights = contrib_mask.sum(axis=0).astype(np.float64)  # shape (Y, X)
    # sum contributions (ignore non-contributors)
    sum_vals = np.where(contrib_mask, cube, 0.0).sum(axis=0)
    # avoid divide by zero
    valid = weights > 0
    avg = np.full_like(sum_vals, np.nan, dtype=np.float64)
    avg[valid] = sum_vals[valid] / weights[valid]
    max_weight = float(np.max(weights)) if weights.size else 0.0
    if max_weight <= 0:
        raise RuntimeError("No valid contributions found across inputs (all weights zero)")
    scaled = np.full_like(avg, np.nan, dtype=np.float64)
    scaled[valid] = avg[valid] * math.sqrt(max_weight)
    return scaled, weights

def save_as_float32(data64, header, output_file, overwrite=True):
    data32 = data64.astype(np.float32)
    hdr = sanitize_header_for_float32(header) if header is not None else fits.Header()
    hdu = fits.PrimaryHDU(data=data32, header=hdr)
    hdu.writeto(output_file, overwrite=overwrite)

# -------------------------
# Preview canvas (fixed)
# -------------------------
class PreviewCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6,4))
        super().__init__(fig)
        # create axes once and reuse
        self.ax_img = fig.add_subplot(1,2,1)
        self.ax_hist = fig.add_subplot(1,2,2)
        fig.tight_layout()
        self._img_artist = None
        self._cbar = None

    def plot(self, image, header=None, hist_bins=256):
        # clear existing axes content but keep axes objects
        self.ax_img.clear()
        self.ax_hist.clear()

        if image is None:
            self.ax_img.set_title("No image")
        else:
            # If WCS header provided, attempt WCS projection; fallback to normal imshow
            try:
                if header is not None:
                    wcs = WCS(header)
                    # replace ax_img with a WCSAxes if possible
                    self.figure.delaxes(self.ax_img)
                    self.ax_img = self.figure.add_subplot(1,2,1, projection=wcs)
                else:
                    # ensure ax_img is a normal Axes
                    self.figure.delaxes(self.ax_img)
                    self.ax_img = self.figure.add_subplot(1,2,1)
            except Exception:
                # fallback: ensure a normal axes
                try:
                    self.figure.delaxes(self.ax_img)
                except Exception:
                    pass
                self.ax_img = self.figure.add_subplot(1,2,1)

            im = self.ax_img.imshow(image, origin="lower", cmap="gray")
            # remove old colorbar if present
            if self._cbar:
                try:
                    self._cbar.remove()
                except Exception:
                    pass
            self._cbar = self.figure.colorbar(im, ax=self.ax_img, orientation="vertical")
            self.ax_img.set_title("Stacked image")

        if image is not None:
            clean = image[np.isfinite(image)]
            if clean.size:
                self.ax_hist.hist(clean.ravel(), bins=hist_bins, color="red", histtype="step")
                self.ax_hist.set_title("Histogram (stacked)")
        self.draw()

# -------------------------
# Worker thread for reprojection/stacking
# -------------------------
class StackWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)  # percent
    finished = pyqtSignal(bool, str, object, object)  # success, message, stack_result, header

    def __init__(self, files, reproject_flag, method, sigma, weighted_combine_flag, mask_threshold, outpath, overwrite):
        super().__init__()
        self.files = files
        self.reproject_flag = reproject_flag
        self.method = method
        self.sigma = sigma
        self.weighted_combine_flag = weighted_combine_flag
        self.mask_threshold = float(mask_threshold)
        self.outpath = outpath
        self.overwrite = overwrite
        self._abort = False

    def abort(self):
        self._abort = True

    def _emit(self, *args):
        self.log.emit(" ".join(str(a) for a in args))

    def run(self):
        try:
            N = len(self.files)
            self._emit(f"Processing {N} files...")
            # Determine reference WCS and shape if reprojection requested
            if self.reproject_flag:
                # Validate WCS on first file
                try:
                    data0, hdr0 = load_fits(self.files[0])
                    wcs0 = WCS(hdr0)
                    ny, nx = int(hdr0.get("NAXIS2")), int(hdr0.get("NAXIS1"))
                except Exception as e:
                    raise RuntimeError(f"Failed to read reference WCS from {self.files[0]}: {e}")
                # Reproject each file into ref grid and build cube
                cube_list = []
                for i, fn in enumerate(self.files, start=1):
                    if self._abort:
                        self._emit("Aborted by user.")
                        self.finished.emit(False, "Aborted", None, None)
                        return
                    self._emit(f"[{i}/{N}] Reprojecting: {os.path.basename(fn)}")
                    try:
                        data, hdr = load_fits(fn)
                        # normalize shape and convert to mono if needed
                        data = normalize_fits_shape(data)
                        # require 2D mono for weighted combine; if 3-channel, convert to luma for stacking
                        if data.ndim == 3 and data.shape[0] == 3:
                            data_for_reproj = np.mean(data.astype(np.float64), axis=0)
                        else:
                            data_for_reproj = data.astype(np.float64)
                        wcs_in = WCS(hdr)
                        reprojected, footprint = reproject_interp((data_for_reproj, wcs_in), wcs0, shape_out=(ny, nx))
                        # replace non-finite with NaN
                        reprojected = np.asarray(reprojected, dtype=np.float64)
                        reprojected[~np.isfinite(reprojected)] = np.nan
                        cube_list.append(reprojected)
                    except Exception as e:
                        self._emit(f"Failed to reproject {fn}: {e}; skipping")
                    self.progress.emit(int(100.0 * i / N))
                if not cube_list:
                    raise RuntimeError("No files successfully reprojected.")
                cube = np.stack(cube_list, axis=0)  # shape (M, Y, X)
                header_out = hdr0  # use reference header for output
            else:
                # load all without reprojection; ensure shapes match
                shapes = set()
                arrs = []
                header_out = None
                for i, fn in enumerate(self.files, start=1):
                    if self._abort:
                        self._emit("Aborted by user.")
                        self.finished.emit(False, "Aborted", None, None)
                        return
                    self._emit(f"[{i}/{N}] Loading: {os.path.basename(fn)}")
                    try:
                        data, hdr = load_fits(fn)
                        data = normalize_fits_shape(data)
                        # convert 3-channel to luma for stacking (this GUI focuses on mono stacks)
                        if data.ndim == 3 and data.shape[0] == 3:
                            data = np.mean(data.astype(np.float64), axis=0)
                        else:
                            data = data.astype(np.float64)
                        arrs.append(data)
                        shapes.add(data.shape)
                        if header_out is None:
                            header_out = hdr
                    except Exception as e:
                        self._emit(f"Failed to read {fn}: {e}; skipping")
                    self.progress.emit(int(100.0 * i / N))
                if not arrs:
                    raise RuntimeError("No valid input files loaded.")
                if len(shapes) != 1:
                    raise RuntimeError("Input FITS files have differing shapes. Enable reprojection or make shapes equal.")
                cube = np.stack(arrs, axis=0)
            # At this point we have a mono cube (N, Y, X)
            self._emit(f"Cube shape: {cube.shape}, dtype={cube.dtype}")

            # Choose stacking method or weighted combine
            if self.weighted_combine_flag:
                self._emit("Performing weighted combine (per-pixel contribution count + sqrt scaling)...")
                final, weight_map = weighted_combine_from_cube(cube, mask_threshold=self.mask_threshold)
                # Save both final and weight map as separate HDUs (primary=final, extension=weights)
                hdr_out = sanitize_header_for_float32(header_out) if header_out is not None else fits.Header()
                primary = fits.PrimaryHDU(final.astype(np.float32), header=hdr_out)
                hdu_weights = fits.ImageHDU(weight_map.astype(np.float32), name="WEIGHTS")
                hdul = fits.HDUList([primary, hdu_weights])
                if os.path.exists(self.outpath) and not self.overwrite:
                    raise FileExistsError(f"Output exists and overwrite disabled: {self.outpath}")
                hdul.writeto(self.outpath, overwrite=self.overwrite)
                self._emit(f"Wrote weighted composite and weight map to {self.outpath}")
                self.finished.emit(True, f"Wrote {self.outpath}", final, header_out)
                return

            # Standard stacking
            self._emit(f"Computing stack with method: {self.method}")
            if self.method == "mean":
                result = compute_stack_mean(cube)
            elif self.method == "median":
                result = compute_stack_median(cube)
            elif self.method == "sigma":
                result = compute_stack_sigma_clip(cube, sigma=self.sigma)
            else:
                raise ValueError(f"Unknown stacking method: {self.method}")

            # Save result
            if os.path.exists(self.outpath) and not self.overwrite:
                raise FileExistsError(f"Output exists and overwrite disabled: {self.outpath}")
            save_as_float32(result, header_out, self.outpath, overwrite=self.overwrite)
            self._emit(f"Saved stacked image to {self.outpath}")
            self.finished.emit(True, f"Wrote {self.outpath}", result, header_out)

        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit(f"Error: {e}\n{tb}")
            self.finished.emit(False, str(e), None, None)

# -------------------------
# GUI
# -------------------------
class StackerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS Stacker (fixed + weighted combine)")
        self._build_ui()
        self.resize(1100, 760)
        self.stack_result = None
        self.stack_header = None
        self.worker = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Input directory
        grid.addWidget(QLabel("Input FITS directory:"), 0, 0)
        self.input_dir_edit = QLineEdit()
        grid.addWidget(self.input_dir_edit, 0, 1, 1, 3)
        btn_dir = QPushButton("Browse...")
        btn_dir.clicked.connect(self._browse_dir)
        grid.addWidget(btn_dir, 0, 4)

        # Output filename
        grid.addWidget(QLabel("Output FITS file:"), 1, 0)
        self.output_edit = QLineEdit("stacked_output.fits")
        grid.addWidget(self.output_edit, 1, 1, 1, 3)
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(self._browse_save)
        grid.addWidget(btn_out, 1, 4)

        # Method dropdown
        grid.addWidget(QLabel("Stack method:"), 2, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["mean", "median", "sigma-clipped mean"])
        grid.addWidget(self.method_combo, 2, 1)

        # Sigma parameter (for sigma-clipped mean)
        grid.addWidget(QLabel("Sigma (for sigma-clipped):"), 2, 2)
        self.sigma_edit = QLineEdit("3.0")
        grid.addWidget(self.sigma_edit, 2, 3)

        # Reproject toggle
        self.reproject_chk = QCheckBox("Reproject to first FITS WCS")
        grid.addWidget(self.reproject_chk, 3, 0, 1, 3)

        # Weighted combine option (per-pixel counts + sqrt scaling)
        self.weighted_chk = QCheckBox("Weighted combine (per-pixel count + sqrt(max_count) scaling)")
        grid.addWidget(self.weighted_chk, 4, 0, 1, 4)

        # Mask threshold for weighted combine
        grid.addWidget(QLabel("Mask threshold (for weighted combine):"), 5, 0)
        self.mask_spin = QDoubleSpinBox()
        self.mask_spin.setRange(-1e12, 1e12)
        self.mask_spin.setDecimals(6)
        self.mask_spin.setValue(0.0)
        grid.addWidget(self.mask_spin, 5, 1)

        # Run / preview / log
        self.run_btn = QPushButton("Run Stack")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 6, 0)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)
        grid.addWidget(self.cancel_btn, 6, 1)

        self.preview_btn = QPushButton("Preview Last Result")
        self.preview_btn.clicked.connect(self._on_preview)
        grid.addWidget(self.preview_btn, 6, 2)

        self.load_output_btn = QPushButton("Plot Output File")
        self.load_output_btn.clicked.connect(self._plot_output_file)
        grid.addWidget(self.load_output_btn, 6, 3)

        # Progress bar and log
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        grid.addWidget(self.progress, 7, 0, 1, 5)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        grid.addWidget(self.log_box, 8, 0, 1, 5)

        # Preview canvas
        self.canvas = PreviewCanvas()
        grid.addWidget(self.canvas, 9, 0, 6, 5)

    def _log(self, *args):
        self.log_box.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_dir(self):
        dn = QFileDialog.getExistingDirectory(self, "Select input directory", "")
        if dn:
            self.input_dir_edit.setText(dn)

    def _browse_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Select output FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def _on_run(self):
        try:
            input_dir = self.input_dir_edit.text().strip()
            if not input_dir:
                raise ValueError("Select input directory")
            pattern = os.path.join(input_dir, "*.fit*")
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No FITS files found in {input_dir}")
            self._log(f"Found {len(files)} FITS files")

            reproject_flag = self.reproject_chk.isChecked()
            weighted_flag = self.weighted_chk.isChecked()
            method_text = self.method_combo.currentText()
            if method_text == "mean":
                method = "mean"
            elif method_text == "median":
                method = "median"
            else:
                method = "sigma"
            sigma = float(self.sigma_edit.text().strip() or 3.0)
            mask_thresh = float(self.mask_spin.value())
            outpath = self.output_edit.text().strip() or "stacked_output.fits"
            overwrite = True  # always allow overwrite via save dialog; could add checkbox

            # disable UI while running
            self.run_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self._log("Starting worker thread for reprojection/stacking...")
            self.progress.setValue(0)

            # start worker
            self.worker = StackWorker(files, reproject_flag, method, sigma, weighted_flag, mask_thresh, outpath, overwrite)
            self.worker.log.connect(self._log)
            self.worker.progress.connect(self.progress.setValue)
            self.worker.finished.connect(self._on_finished)
            self.worker.start()

        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")
            self._log("Error:", e)

    def _on_cancel(self):
        if self.worker is not None:
            self.worker.abort()
            self._log("Abort requested; waiting for worker to stop...")
            self.cancel_btn.setEnabled(False)

    def _on_finished(self, success, message, stack_result, header):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(100 if success else 0)
        if success:
            QMessageBox.information(self, "Done", message)
            self._log("Worker finished:", message)
            self.stack_result = stack_result
            self.stack_header = header
            if stack_result is not None:
                self.canvas.plot(stack_result, header)
        else:
            QMessageBox.critical(self, "Failed", message)
            self._log("Worker failed:", message)

    def _on_preview(self):
        if self.stack_result is None:
            QMessageBox.information(self, "No result", "There is no stacked result to preview. Run the stack first.")
            return
        self.canvas.plot(self.stack_result, self.stack_header)
        st = compute_stats(self.stack_result)
        self._log(f"Preview stats: min={st['min']:.6g} max={st['max']:.6g} mean={st['mean']:.6g} std={st['std']:.6g}")

    def _plot_output_file(self):
        outpath = self.output_edit.text().strip() or "stacked_output.fits"
        if not os.path.exists(outpath):
            fn, _ = QFileDialog.getOpenFileName(self, "Select output FITS to plot", "", "FITS Files (*.fits *.fit);;All Files (*)")
            if not fn:
                return
            outpath = fn
        try:
            data, hdr = load_fits(outpath)
            if data is None:
                raise ValueError("No data in primary HDU")
            # normalize and convert to 2D for preview
            data = normalize_fits_shape(data)
            if data.ndim == 3 and data.shape[0] == 3:
                data2 = np.mean(data.astype(np.float64), axis=0)
            elif data.ndim == 3 and data.shape[0] != 3:
                # unexpected cube; take first plane
                data2 = data[0].astype(np.float64)
            else:
                data2 = data.astype(np.float64)
            self.canvas.plot(data2, hdr)
            st = compute_stats(data2)
            self._log(f"Plotted output {os.path.basename(outpath)} stats: min={st['min']:.6g} max={st['max']:.6g} mean={st['mean']:.6g} std={st['std']:.6g}")
        except Exception as e:
            QMessageBox.critical(self, "Plot error", str(e))
            self._log("Plot error:", e)

# ---------- small helper used by preview logging ----------
def compute_stats(arr):
    a = np.asarray(arr).ravel()
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}

# ---------- entry ----------
def main():
    app = QApplication(sys.argv)
    w = StackerWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
