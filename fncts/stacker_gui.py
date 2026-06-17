#!/usr/bin/env python3
"""
stacker_gui.py

Dust-map–oriented FITS stacker with:
 - Optional reprojection to a mosaic WCS using find_optimal_celestial_wcs (oldest API)
 - Fallback to first-frame WCS if mosaicking is unavailable
 - Flux-linear weighted combine (no sqrt scaling)
 - Primary HDU = stacked image, WEIGHTS extension = per-pixel counts
 - Optional masking of low-coverage pixels (weight < 2)
"""

import sys
import os
import glob
import traceback
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from astropy.wcs.utils import proj_plane_pixel_scales

# --- Reproject imports with safe fallback ---
from reproject import reproject_interp
try:
    from reproject.mosaicking import find_optimal_celestial_wcs
except Exception:
    find_optimal_celestial_wcs = None

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QCheckBox, QTextEdit, QMessageBox,
    QDoubleSpinBox, QProgressBar
)
from PyQt6.QtCore import QThread, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# -------------------------
# FITS helpers
# -------------------------
def load_fits(path):
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.array(hdu.data), hdu.header
    raise ValueError(f"No image data in {path}")

def sanitize_header_for_float32(header):
    hdr = header.copy()
    hdr["BITPIX"] = -32
    for k in ("BZERO", "BSCALE"):
        if k in hdr:
            hdr.pop(k)
    return hdr

def normalize_fits_shape(arr):
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[2] == 1:
            return a[:, :, 0]
        if a.shape[2] == 3:
            return np.transpose(a, (2, 0, 1))
        if a.shape[0] == 1:
            return a[0]
        if a.shape[0] == 3:
            return a
    s = np.squeeze(a)
    if s.ndim == 2:
        return s
    if s.ndim == 3 and s.shape[0] in (1, 3):
        return s[0] if s.shape[0] == 1 else s
    raise ValueError(f"Unsupported FITS shape: {a.shape}")

def to_luma(arr):
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
    clipped = sigma_clip(cube, sigma=sigma, maxiters=maxiters, axis=0, masked=True)
    arr = clipped.filled(np.nan)
    return np.nanmean(arr, axis=0)

def weighted_combine_from_cube(cube, mask_threshold=0.0):
    if cube.ndim != 3:
        raise ValueError("weighted_combine_from_cube expects (N, Y, X)")
    finite_mask = np.isfinite(cube)
    contrib_mask = finite_mask & (cube > mask_threshold)
    weights = contrib_mask.sum(axis=0).astype(np.float64)
    sum_vals = np.where(contrib_mask, cube, 0.0).sum(axis=0)
    valid = weights > 0
    avg = np.full_like(sum_vals, np.nan, dtype=np.float64)
    avg[valid] = sum_vals[valid] / weights[valid]
    return avg, weights

def save_stack_and_weights(stack, weights, header, outpath, overwrite=True):
    hdr_out = sanitize_header_for_float32(header) if header is not None else fits.Header()
    primary = fits.PrimaryHDU(stack.astype(np.float32), header=hdr_out)
    hdu_weights = fits.ImageHDU(weights.astype(np.float32), name="WEIGHTS")
    fits.HDUList([primary, hdu_weights]).writeto(outpath, overwrite=overwrite)

# -------------------------
# Preview canvas
# -------------------------
class PreviewCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4))
        super().__init__(fig)
        self.ax_img = fig.add_subplot(1, 2, 1)
        self.ax_hist = fig.add_subplot(1, 2, 2)
        fig.tight_layout()
        self._cbar = None

    def plot(self, image, header=None, hist_bins=256):
        self.ax_img.clear()
        self.ax_hist.clear()

        if image is None:
            self.ax_img.set_title("No image")
        else:
            try:
                if header is not None:
                    wcs = WCS(header)
                    self.figure.delaxes(self.ax_img)
                    self.ax_img = self.figure.add_subplot(1, 2, 1, projection=wcs)
                else:
                    self.figure.delaxes(self.ax_img)
                    self.ax_img = self.figure.add_subplot(1, 2, 1)
            except Exception:
                try:
                    self.figure.delaxes(self.ax_img)
                except Exception:
                    pass
                self.ax_img = self.figure.add_subplot(1, 2, 1)

            im = self.ax_img.imshow(image, origin="lower", cmap="gray")
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
# Worker thread
# -------------------------
class StackWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str, object, object)

    def __init__(self, files, reproject_flag, method, sigma,
                 weighted_combine_flag, mask_threshold,
                 mask_lowcov_flag, lowcov_threshold,
                 outpath, overwrite):
        super().__init__()
        self.files = files
        self.reproject_flag = reproject_flag
        self.method = method
        self.sigma = sigma
        self.weighted_combine_flag = weighted_combine_flag
        self.mask_threshold = float(mask_threshold)
        self.mask_lowcov_flag = mask_lowcov_flag
        self.lowcov_threshold = int(lowcov_threshold)
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

            frames = []
            for i, fn in enumerate(self.files, start=1):
                if self._abort:
                    self._emit("Aborted by user.")
                    self.finished.emit(False, "Aborted", None, None)
                    return
                self._emit(f"[{i}/{N}] Reading: {os.path.basename(fn)}")
                try:
                    data, hdr = load_fits(fn)
                    data = normalize_fits_shape(data)
                    if data.ndim == 3 and data.shape[0] == 3:
                        data_for_wcs = np.mean(data.astype(np.float64), axis=0)
                    else:
                        data_for_wcs = data.astype(np.float64)
                    wcs = WCS(hdr)
                    frames.append((data_for_wcs, hdr, wcs))
                except Exception as e:
                    self._emit(f"Failed to read {fn}: {e}; skipping")
                self.progress.emit(int(100.0 * i / N))

            if not frames:
                raise RuntimeError("No valid input files loaded.")

            # -------------------------
            # WCS selection
            # -------------------------
            if self.reproject_flag:
                if find_optimal_celestial_wcs is None:
                    self._emit("find_optimal_celestial_wcs not available; using first-frame WCS.")
                    data0, hdr0 = load_fits(self.files[0])
                    data0 = normalize_fits_shape(data0)
                    if data0.ndim == 3 and data0.shape[0] == 3:
                        data0 = np.mean(data0.astype(np.float64), axis=0)
                    else:
                        data0 = data0.astype(np.float64)
                    mosaic_wcs = WCS(hdr0)
                    ny, nx = data0.shape
                    shape_out = (ny, nx)
                else:
                    self._emit("Computing optimal mosaic WCS for all inputs...")
                    pairs = [(f[0], f[2]) for f in frames]
                    mosaic_wcs, shape_out = find_optimal_celestial_wcs(pairs)
                    ny, nx = shape_out

                # -------------------------
                # Reproject frames
                # -------------------------
                cube_list = []
                for i, (data_for_wcs, hdr, wcs_in) in enumerate(frames, start=1):
                    if self._abort:
                        self._emit("Aborted by user.")
                        self.finished.emit(False, "Aborted", None, None)
                        return
                    self._emit(f"[{i}/{N}] Reprojecting: {os.path.basename(self.files[i-1])}")
                    try:
                        reprojected, footprint = reproject_interp(
                            (data_for_wcs, wcs_in),
                            mosaic_wcs,
                            shape_out=shape_out
                        )
                        reprojected = np.asarray(reprojected, dtype=np.float64)
                        reprojected[~np.isfinite(reprojected)] = np.nan
                        cube_list.append(reprojected)
                    except Exception as e:
                        self._emit(f"Failed to reproject frame {i}: {e}; skipping")
                    self.progress.emit(int(100.0 * i / N))

                if not cube_list:
                    raise RuntimeError("No frames successfully reprojected.")
                cube = np.stack(cube_list, axis=0)
                header_out = mosaic_wcs.to_header()

            else:
                # -------------------------
                # No reprojection
                # -------------------------
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
                ny, nx = cube.shape[1], cube.shape[2]

            self._emit(f"Cube shape: {cube.shape}, dtype={cube.dtype}")

            # -------------------------
            # Weighted combine
            # -------------------------
            if self.weighted_combine_flag:
                self._emit("Performing flux-linear weighted combine...")
                final, weight_map = weighted_combine_from_cube(cube, mask_threshold=self.mask_threshold)
                if self.mask_lowcov_flag:
                    self._emit(f"Masking pixels with weight < {self.lowcov_threshold}")
                    mask = weight_map < self.lowcov_threshold
                    final[mask] = np.nan
                if os.path.exists(self.outpath) and not self.overwrite:
                    raise FileExistsError(f"Output exists and overwrite disabled: {self.outpath}")
                save_stack_and_weights(final, weight_map, header_out, self.outpath, overwrite=self.overwrite)
                self._emit(f"Wrote weighted composite and weight map to {self.outpath}")
                self.finished.emit(True, f"Wrote {self.outpath}", final, header_out)
                return

            # -------------------------
            # Standard stacking
            # -------------------------
            self._emit(f"Computing stack with method: {self.method}")
            if self.method == "mean":
                result = compute_stack_mean(cube)
            elif self.method == "median":
                result = compute_stack_median(cube)
            elif self.method == "sigma":
                result = compute_stack_sigma_clip(cube, sigma=self.sigma)
            else:
                raise ValueError(f"Unknown stacking method: {self.method}")

            if os.path.exists(self.outpath) and not self.overwrite:
                raise FileExistsError(f"Output exists and overwrite disabled: {self.outpath}")
            save_stack_and_weights(result, np.ones_like(result), header_out, self.outpath, overwrite=self.overwrite)
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
        self.setWindowTitle("FITS Stacker (dust-map optimized)")
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

        grid.addWidget(QLabel("Input FITS directory:"), 0, 0)
        self.input_dir_edit = QLineEdit()
        grid.addWidget(self.input_dir_edit, 0, 1, 1, 3)
        btn_dir = QPushButton("Browse...")
        btn_dir.clicked.connect(self._browse_dir)
        grid.addWidget(btn_dir, 0, 4)

        grid.addWidget(QLabel("Output FITS file:"), 1, 0)
        self.output_edit = QLineEdit("stacked_output.fits")
        grid.addWidget(self.output_edit, 1, 1, 1, 3)
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(self._browse_save)
        grid.addWidget(btn_out, 1, 4)

        grid.addWidget(QLabel("Stack method:"), 2, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["mean", "median", "sigma-clipped mean"])
        grid.addWidget(self.method_combo, 2, 1)

        grid.addWidget(QLabel("Sigma (for sigma-clipped):"), 2, 2)
        self.sigma_edit = QLineEdit("3.0")
        grid.addWidget(self.sigma_edit, 2, 3)

        self.reproject_chk = QCheckBox("Reproject to mosaic WCS (if available)")
        self.reproject_chk.setChecked(True)
        grid.addWidget(self.reproject_chk, 3, 0, 1, 3)

        self.weighted_chk = QCheckBox("Weighted combine (per-pixel average, flux-linear)")
        grid.addWidget(self.weighted_chk, 4, 0, 1, 4)

        grid.addWidget(QLabel("Mask threshold (for weighted combine):"), 5, 0)
        self.mask_spin = QDoubleSpinBox()
        self.mask_spin.setRange(-1e12, 1e12)
        self.mask_spin.setDecimals(6)
        self.mask_spin.setValue(0.0)
        grid.addWidget(self.mask_spin, 5, 1)

        self.lowcov_chk = QCheckBox("Mask low-coverage pixels (weight < 2)")
        self.lowcov_chk.setChecked(True)
        grid.addWidget(self.lowcov_chk, 6, 0, 1, 3)

        self.run_btn = QPushButton("Run Stack")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 7, 0)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)
        grid.addWidget(self.cancel_btn, 7, 1)

        self.preview_btn = QPushButton("Preview Last Result")
        self.preview_btn.clicked.connect(self._on_preview)
        grid.addWidget(self.preview_btn, 7, 2)

        self.load_output_btn = QPushButton("Plot Output File")
        self.load_output_btn.clicked.connect(self._plot_output_file)
        grid.addWidget(self.load_output_btn, 7, 3)

        # Progress bar and log
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        grid.addWidget(self.progress, 8, 0, 1, 5)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        grid.addWidget(self.log_box, 9, 0, 1, 5)

        # Preview canvas
        self.canvas = PreviewCanvas()
        grid.addWidget(self.canvas, 10, 0, 6, 5)

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
            mask_lowcov_flag = self.lowcov_chk.isChecked()

            method_text = self.method_combo.currentText()
            if method_text == "mean":
                method = "mean"
            elif method_text == "median":
                method = "median"
            else:
                method = "sigma"

            sigma = float(self.sigma_edit.text().strip() or 3.0)
            mask_thresh = float(self.mask_spin.value())
            lowcov_threshold = 2

            outpath = self.output_edit.text().strip() or "stacked_output.fits"
            overwrite = True

            self.run_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self._log("Starting worker thread for reprojection/stacking...")
            self.progress.setValue(0)

            self.worker = StackWorker(
                files, reproject_flag, method, sigma,
                weighted_flag, mask_thresh,
                mask_lowcov_flag, lowcov_threshold,
                outpath, overwrite
            )
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
            data = normalize_fits_shape(data)
            if data.ndim == 3 and data.shape[0] == 3:
                data2 = np.mean(data.astype(np.float64), axis=0)
            elif data.ndim == 3:
                data2 = data[0].astype(np.float64)
            else:
                data2 = data.astype(np.float64)
            self.canvas.plot(data2, hdr)
            st = compute_stats(data2)
            self._log(f"Plotted output {os.path.basename(outpath)} stats: min={st['min']:.6g} max={st['max']:.6g} mean={st['mean']:.6g} std={st['std']:.6g}")
        except Exception as e:
            QMessageBox.critical(self, "Plot error", str(e))
            self._log("Plot error:", e)

def compute_stats(arr):
    a = np.asarray(arr).ravel()
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}

def main():
    app = QApplication(sys.argv)
    w = StackerWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
