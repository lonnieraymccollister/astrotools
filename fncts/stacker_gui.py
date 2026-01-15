#!/usr/bin/env python3
"""
st acker_gui.py
PyQt6 GUI for stacking FITS images.

Features:
- Select input directory containing .fits files
- Choose stacking method: Mean, Median, Sigma clipped mean
- Optionally reproject inputs to the WCS of the first FITS
- Set output filename and run; result saved as float32
- Preview stacked image (with WCS) and histogram inside the GUI
"""
import sys
import os
import glob
import traceback

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from reproject import reproject_interp

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QCheckBox, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# -------------------------
# Core stacking helpers
# -------------------------
def get_fits_files(input_dir):
    pattern = os.path.join(input_dir, "*.fit*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No FITS files found in {input_dir}")
    return files

def reproject_all_to_ref(files):
    # Use first file as reference
    ref_hdr = fits.getheader(files[0], ext=0)
    ref_wcs = WCS(ref_hdr)
    ny, nx = int(ref_hdr.get("NAXIS2")), int(ref_hdr.get("NAXIS1"))

    cube = []
    headers = []
    for fn in files:
        with fits.open(fn) as hdul:
            data = hdul[0].data.astype(np.float64)
            hdr = hdul[0].header
            wcs_in = WCS(hdr)
        reprojected, _ = reproject_interp((data, wcs_in), ref_wcs, shape_out=(ny, nx))
        cube.append(reprojected)
        headers.append(hdr)
    return np.stack(cube, axis=0), ref_hdr

def load_all_without_reproject(files):
    arrs = []
    headers = []
    shapes = set()
    for fn in files:
        with fits.open(fn) as hdul:
            data = hdul[0].data.astype(np.float64)
            hdr = hdul[0].header
        arrs.append(data)
        headers.append(hdr)
        shapes.add(data.shape)
    if len(shapes) != 1:
        raise ValueError("Input FITS files have differing shapes. Enable reprojection or make shapes equal.")
    return np.stack(arrs, axis=0), headers[0]

def compute_stack(cube, method="mean", sigma_clip=None):
    if method == "mean":
        return np.nanmean(cube, axis=0)
    if method == "median":
        return np.nanmedian(cube, axis=0)
    if method == "sigma":
        # sigma-clipped mean per-pixel
        # cube shape: (N, Y, X) -> compute along axis 0
        N, Y, X = cube.shape
        out = np.full((Y, X), np.nan, dtype=np.float64)
        for yi in range(Y):
            # process row to reduce Python overhead per pixel
            rowslice = cube[:, yi, :]  # shape (N, X)
            # compute sigma-clipped on each column vector; vectorize using loop per column
            # This keeps memory reasonable; could optimize with astropy.stats.sigma_clipped_stats on 1D arrays
            for xi in range(X):
                col = rowslice[:, xi]
                if np.all(np.isnan(col)):
                    out[yi, xi] = np.nan
                else:
                    mean, med, std = sigma_clipped_stats(col, sigma=sigma_clip, maxiters=5)
                    out[yi, xi] = mean
        return out
    raise ValueError(f"Unknown stacking method: {method}")

def save_as_float32(data64, header, output_file):
    data32 = data64.astype(np.float32)
    hdr = header.copy()
    hdr["BITPIX"] = -32
    hdu = fits.PrimaryHDU(data=data32, header=hdr)
    hdu.writeto(output_file, overwrite=True)

# -------------------------
# Matplotlib canvas
# -------------------------
class PreviewCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6,4))
        super().__init__(fig)
        self.ax_img = fig.add_subplot(1,2,1, projection=None)
        self.ax_hist = fig.add_subplot(1,2,2)
        fig.tight_layout()

    def plot(self, image, header=None, hist_bins=256):
        self.ax_img.clear()
        self.ax_hist.clear()
        if image is None:
            self.ax_img.set_title("No image")
        else:
            # If WCS header provided, attempt WCS projection; fallback to normal imshow
            try:
                from astropy.visualization.wcsaxes import WCSAxes
                if header is not None:
                    wcs = WCS(header)
                    self.ax_img = self.figure.add_subplot(1,2,1, projection=wcs)
                else:
                    self.ax_img = self.figure.add_subplot(1,2,1)
            except Exception:
                self.ax_img = self.figure.add_subplot(1,2,1)
            im = self.ax_img.imshow(image, origin="lower", cmap="gray")
            self.figure.colorbar(im, ax=self.ax_img, orientation="vertical")
            self.ax_img.set_title("Stacked image")
        if image is not None:
            clean = image[np.isfinite(image)]
            if clean.size:
                self.ax_hist.hist(clean.ravel(), bins=hist_bins, color="red", histtype="step")
                self.ax_hist.set_title("Histogram (stacked)")
        self.draw()

# -------------------------
# GUI
# -------------------------
class StackerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS Stacker")
        self._build_ui()
        self.resize(1000, 680)
        self.stack_result = None
        self.stack_header = None

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

        # Run / preview / log
        self.run_btn = QPushButton("Run Stack")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 4, 0)

        self.preview_btn = QPushButton("Preview Last Result")
        self.preview_btn.clicked.connect(self._on_preview)
        grid.addWidget(self.preview_btn, 4, 1)

        self.load_output_btn = QPushButton("Plot Output File")
        self.load_output_btn.clicked.connect(self._plot_output_file)
        grid.addWidget(self.load_output_btn, 4, 2)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        grid.addWidget(self.log_box, 5, 0, 1, 5)

        # Preview canvas
        self.canvas = PreviewCanvas()
        grid.addWidget(self.canvas, 6, 0, 6, 5)

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
            files = get_fits_files(input_dir)
            self._log(f"Found {len(files)} FITS files")

            reproject_flag = self.reproject_chk.isChecked()
            if reproject_flag:
                self._log("Reprojecting all files to first file WCS. This may be slow.")
                cube, ref_hdr = reproject_all_to_ref(files)
                header_out = ref_hdr
            else:
                cube, header_out = load_all_without_reproject(files)

            method = self.method_combo.currentText()
            if method == "mean":
                self._log("Computing mean stack")
                stack = compute_stack(cube, method="mean")
            elif method == "median":
                self._log("Computing median stack")
                stack = compute_stack(cube, method="median")
            else:
                sigma = float(self.sigma_edit.text().strip() or 3.0)
                self._log(f"Computing sigma-clipped mean (sigma={sigma})")
                stack = compute_stack(cube, method="sigma", sigma_clip=sigma)

            # Save
            outpath = self.output_edit.text().strip() or "stacked_output.fits"
            save_as_float32(stack, header_out, outpath)
            self._log(f"Saved stacked image to {outpath}")

            # store result for preview
            self.stack_result = stack
            self.stack_header = header_out

            # show preview
            self.canvas.plot(stack, header_out)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")
            self._log("Error:", e)

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
            if data.ndim > 2:
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