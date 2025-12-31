#!/usr/bin/env python3
"""
Simple FITS WCS overlay GUI
Saves a JPEG at user-specified DPI with the longest side limited by Max Pixels.
Adds:
 - High resolution preview (checkbox, default enabled)
 - Display histogram (checkbox, default enabled)
 - Max pixel dimension (spinbox, default 10000)
 - Export DPI (spinbox, default 300; 0 = auto)
 - When preview/histogram are checked, both windows open simultaneously (non-modal)
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
    QVBoxLayout, QHBoxLayout, QMessageBox, QCheckBox, QSpinBox, QDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage

import matplotlib
# Use Qt backend for embedding canvases
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

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

# ---------- small histogram dialog ----------
class HistDialog(QDialog):
    def __init__(self, img_rgb, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Brightness Histogram")
        self.resize(520, 360)
        layout = QVBoxLayout()
        self.setLayout(layout)

        fig = Figure(figsize=(5,3.5), dpi=100)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)
        ax = fig.add_subplot(1,1,1)
        # per-pixel brightness = max across channels
        flat = img_rgb.reshape(-1, img_rgb.shape[-1])
        per_pixel = np.max(flat, axis=1)
        ax.hist(per_pixel, bins=120, range=(0.0,1.0), color="#2c7fb8", alpha=0.8)
        ax.set_xlabel("Brightness")
        ax.set_ylabel("Count")
        ax.set_xlim(0,1)
        self.canvas.draw()

# ---------- preview dialog ----------
class PreviewDialog(QDialog):
    def __init__(self, img_rgb, title="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("High-res Preview")
        self.resize(900, 700)
        self._img_rgb = img_rgb
        self._title = title

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.lbl = QLabel()
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.lbl)

        self._update_pixmap()
        if title:
            self.setWindowTitle(f"High-res Preview — {title}")

    def _update_pixmap(self):
        arr = np.clip(self._img_rgb * 255.0, 0, 255).astype(np.uint8)
        h, w = arr.shape[:2]
        # create QImage from RGB array
        if arr.ndim == 3 and arr.shape[2] == 3:
            # QImage expects bytes in row-major order; ensure contiguous
            arr_c = np.ascontiguousarray(arr)
            qimg = QImage(arr_c.data, w, h, 3*w, QImage.Format.Format_RGB888)
        else:
            pil = Image.fromarray(arr)
            pil = pil.convert("RGB")
            data = pil.tobytes("raw", "RGB")
            qimg = QImage(data, pil.width, pil.height, 3*pil.width, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        # scale pixmap to fit dialog while keeping aspect
        scaled = pix.scaled(self.size() - QSize(40, 80), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self._update_pixmap()
        except Exception:
            pass

# ---------- GUI ----------
class FitsWcsPlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS WCS RGB Plotter")
        self._preview_dialog = None
        self._hist_dialog = None
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

        # Row 4: Options (high-res preview, histogram)
        row4 = QHBoxLayout()
        self.preview_check = QCheckBox("High-res preview")
        self.hist_check = QCheckBox("Display histogram")
        # Default both enabled
        self.preview_check.setChecked(True)
        self.hist_check.setChecked(True)
        row4.addWidget(self.preview_check)
        row4.addWidget(self.hist_check)
        row4.addStretch()
        layout.addLayout(row4)

        # Row 5: Max pixels and DPI
        row5 = QHBoxLayout()
        lbl_max = QLabel("Max pixel dimension:")
        self.maxpix_spin = QSpinBox()
        self.maxpix_spin.setRange(100, 20000)
        self.maxpix_spin.setSingleStep(100)
        self.maxpix_spin.setValue(10000)  # default 10000
        lbl_dpi = QLabel("Export DPI (0 = auto):")
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(0, 2400)
        self.dpi_spin.setValue(300)  # default 300
        row5.addWidget(lbl_max)
        row5.addWidget(self.maxpix_spin)
        row5.addSpacing(20)
        row5.addWidget(lbl_dpi)
        row5.addWidget(self.dpi_spin)
        row5.addStretch()
        layout.addLayout(row5)

        # Plot button
        plot_btn = QPushButton("Plot & Save (JPEG)")
        plot_btn.clicked.connect(self._plot_and_save)
        plot_btn.setFixedHeight(36)
        layout.addWidget(plot_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)
        self.resize(820, 240)

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
            max_pix   = int(self.maxpix_spin.value())
            user_dpi  = int(self.dpi_spin.value())
            show_preview = self.preview_check.isChecked()
            show_hist = self.hist_check.isChecked()

            if not fits_path:
                raise ValueError("Please select a FITS WCS file.")
            if not out_path:
                raise ValueError("Please choose an output filename (JPEG).")

            # Load FITS and WCS
            rgb_raw, hdr = read_fits_rgb(fits_path)
            rgb = normalize_per_channel(rgb_raw)

            # compute scaling so longest side == max_pix
            dpi = user_dpi if user_dpi > 0 else 300  # if auto (0), use 300 for sizing; saved DPI will be 0->auto handled below
            h, w = rgb.shape[:2]
            longest = max(h, w)
            if longest <= 0:
                raise ValueError("Invalid image dimensions from FITS.")
            scale = float(max_pix) / float(longest) if longest > max_pix else 1.0
            out_w = int(round(w * scale))
            out_h = int(round(h * scale))

            # warn if extremely large area (prevent accidental OOM)
            MAX_PIX_AREA = 20000 * 20000  # very conservative cap
            if out_w * out_h > MAX_PIX_AREA:
                raise MemoryError(f"Requested output ({out_w}x{out_h}) is too large.")

            # resize image if scaling required (use PIL for high-quality resize)
            if scale != 1.0:
                pil_img = Image.fromarray((rgb * 255.0).astype(np.uint8))
                pil_img = pil_img.resize((out_w, out_h), resample=Image.LANCZOS)
                rgb_disp = np.asarray(pil_img).astype(np.float32) / 255.0
            else:
                rgb_disp = rgb

            # Show high-res preview and histogram simultaneously if requested (non-modal)
            # Keep references on self to avoid garbage collection
            if show_preview:
                try:
                    self._preview_dialog = PreviewDialog(rgb_disp, title=title, parent=self)
                    self._preview_dialog.show()
                except Exception:
                    self._preview_dialog = None

            if show_hist:
                try:
                    self._hist_dialog = HistDialog(rgb_disp, parent=self)
                    self._hist_dialog.show()
                except Exception:
                    self._hist_dialog = None

            # figure size in inches for matplotlib
            save_dpi = user_dpi if user_dpi > 0 else 300
            fig_w_in = out_w / save_dpi
            fig_h_in = out_h / save_dpi

            # Create figure and axis (use WCS if possible)
            try:
                wcs = WCS(hdr, naxis=2)
                fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=save_dpi)
                ax = fig.add_subplot(1, 1, 1, projection=wcs)
            except Exception:
                fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=save_dpi)
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
            fig.savefig(buf, format="png", dpi=save_dpi, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf).convert("RGB")   # ensure RGB (drop alpha)

            out_path = str(Path(out_path))
            img.save(out_path, format="JPEG", quality=95, dpi=(save_dpi, save_dpi), optimize=True)
            buf.close()
            plt.close(fig)

            QMessageBox.information(self, "Success", f"Plot saved to:\n{out_path}\nSize: {out_w}×{out_h} px @ {save_dpi} DPI")
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