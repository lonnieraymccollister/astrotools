#!/usr/bin/env python3
"""
Simple FITS WCS overlay GUI
Saves a JPEG at 300 DPI with the longest side = 10000 pixels.
"""
import sys
from pathlib import Path
import traceback
from io import BytesIO

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox
)
from PyQt6.QtCore import Qt

import matplotlib
# Use Agg for reliable headless rendering to buffer, then save via Pillow
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ---------- helper utilities ----------
def read_fits_rgb(path):
    """Read FITS primary HDU and return HxWx3 float64 RGB array and header."""
    with fits.open(path) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header
    if data is None:
        raise ValueError("No data in primary HDU")
    arr = np.asarray(data)
    # Accept (3, Y, X) or (Y, X, 3) or grayscale (Y,X)
    if arr.ndim == 3:
        if arr.shape[0] == 3:
            rgb = np.transpose(arr, (1, 2, 0))
        elif arr.shape[2] == 3:
            rgb = arr
        else:
            # try first three planes
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

def safe_normalize_channel(ch):
    """Normalize a single channel to 0..1 safely (handles NaN/inf and constant arrays)."""
    ch = np.asarray(ch, dtype=np.float64)
    finite = np.isfinite(ch)
    if not finite.any():
        return np.zeros_like(ch, dtype=np.float32)
    # replace non-finite with median of finite values
    med = np.nanmedian(ch[finite])
    ch_safe = ch.copy()
    ch_safe[~finite] = med
    mn = np.nanmin(ch_safe)
    mx = np.nanmax(ch_safe)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(ch_safe, dtype=np.float32)
    out = (ch_safe - mn) / (mx - mn)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def normalize_per_channel(rgb):
    """Normalize each channel independently to 0..1 with safe handling."""
    a = np.array(rgb, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("normalize_per_channel expects HxWx3 input")
    out = np.zeros_like(a, dtype=np.float32)
    for c in range(3):
        out[..., c] = safe_normalize_channel(a[..., c])
    return out

# ---------- GUI ----------
class FitsWcsPlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS WCS RGB Plotter")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Row 1: FITS input
        row1 = QHBoxLayout()
        lbl1 = QLabel("FITS WCS File:")
        self.fits_edit = QLineEdit()
        btn1 = QPushButton("Browse…")
        btn1.clicked.connect(self._browse_fits)
        row1.addWidget(lbl1)
        row1.addWidget(self.fits_edit, stretch=1)
        row1.addWidget(btn1)
        layout.addLayout(row1)

        # Row 2: Plot title
        row2 = QHBoxLayout()
        lbl2 = QLabel("Plot Title:")
        self.title_edit = QLineEdit()
        row2.addWidget(lbl2)
        row2.addWidget(self.title_edit, stretch=1)
        layout.addLayout(row2)

        # Row 3: Output filename
        row3 = QHBoxLayout()
        lbl3 = QLabel("Save Plot As:")
        self.out_edit = QLineEdit()
        btn3 = QPushButton("Browse…")
        btn3.clicked.connect(self._browse_output)
        row3.addWidget(lbl3)
        row3.addWidget(self.out_edit, stretch=1)
        row3.addWidget(btn3)
        layout.addLayout(row3)

        # Plot button
        plot_btn = QPushButton("Plot & Save (JPEG 300 DPI, longest=10000 px)")
        plot_btn.clicked.connect(self._plot_and_save)
        plot_btn.setFixedHeight(36)
        layout.addWidget(plot_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)
        self.resize(700, 180)

    def _browse_fits(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FITS WCS File", "", "FITS Files (*.fits *.fit);;All Files (*)"
        )
        if path:
            self.fits_edit.setText(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot As…", "", "JPEG Image (*.jpg *.jpeg);;All Files (*)"
        )
        if path:
            # ensure .jpg extension
            p = Path(path)
            if p.suffix.lower() not in (".jpg", ".jpeg"):
                path = str(p.with_suffix(".jpg"))
            self.out_edit.setText(path)

    def _plot_and_save(self):
        try:
            fits_path = self.fits_edit.text().strip()
            title     = self.title_edit.text().strip() or ""
            out_path  = self.out_edit.text().strip()

            if not fits_path:
                raise ValueError("Please select a FITS WCS file.")
            if not out_path:
                raise ValueError("Please choose an output filename (JPEG).")

            # Load FITS and WCS
            rgb_raw, hdr = read_fits_rgb(fits_path)
            rgb = normalize_per_channel(rgb_raw)

            # compute scaling so longest side == MAX_PIX
            MAX_PIX = 10000
            dpi = 300
            h, w = rgb.shape[:2]
            longest = max(h, w)
            if longest <= 0:
                raise ValueError("Invalid image dimensions from FITS.")
            scale = float(MAX_PIX) / float(longest) if longest > MAX_PIX else 1.0
            out_w = int(round(w * scale))
            out_h = int(round(h * scale))

            # warn if extremely large area (prevent accidental OOM)
            MAX_PIX_AREA = 20000 * 20000  # very conservative cap
            if out_w * out_h > MAX_PIX_AREA:
                raise MemoryError(f"Requested output ({out_w}x{out_h}) is too large.")

            # resize image if scaling required (use simple nearest or bilinear)
            if scale != 1.0:
                # use PIL for resizing
                pil_img = Image.fromarray((rgb * 255.0).astype(np.uint8))
                pil_img = pil_img.resize((out_w, out_h), resample=Image.LANCZOS)
                rgb_disp = np.asarray(pil_img).astype(np.float32) / 255.0
            else:
                rgb_disp = rgb

            # figure size in inches for matplotlib
            fig_w_in = out_w / dpi
            fig_h_in = out_h / dpi

            # Create figure and axis (use WCS if possible)
            try:
                wcs = WCS(hdr, naxis=2)
                fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
                ax = fig.add_subplot(1, 1, 1, projection=wcs)
            except Exception:
                fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
                ax = fig.add_subplot(1, 1, 1)

            # Display image
            ax.imshow(rgb_disp, origin='lower', interpolation='nearest', aspect='equal')
            ax.set_xlabel("Right Ascension")
            ax.set_ylabel("Declination")
            try:
                ax.coords.grid(True, color="white", ls="dotted")
            except Exception:
                pass
            if title:
                ax.set_title(title)

            # Render figure to an in-memory PNG, then save as JPEG via Pillow (robust)
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf).convert("RGB")   # ensure RGB (drop alpha)
            # Save JPEG with explicit quality and dpi
            out_path = str(Path(out_path))
            img.save(out_path, format="JPEG", quality=95, dpi=(dpi, dpi), optimize=True)
            buf.close()
            plt.close(fig)

            QMessageBox.information(self, "Success", f"Plot saved to:\n{out_path}\nSize: {out_w}×{out_h} px @ {dpi} DPI")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

# ---------- run ----------
def WcsOvrlay():
    app = QApplication(sys.argv)
    window = FitsWcsPlotter()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(WcsOvrlay())