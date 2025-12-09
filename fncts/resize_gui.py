#!/usr/bin/env python3
"""
resize_gui.py
PyQt6 GUI for resizing images with three modes:
- FITS cubic (float64) using scipy.ndimage.zoom (order=3)
- FITS Lanczos4 using OpenCV (channels handled)
- Other image files (tiff/png/...) using OpenCV (LANCZOS4)

Added:
- Checkbox to update astrometric solution to match larger image.
- When checked and scale > 1, WCS-related header keywords are adjusted.
- Mutually exclusive WCS scaling: adjust CD or CDELT (with PC/CROTA) but never both.
- Changes are logged before/after in the GUI log.
"""
import sys
import traceback
import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom
import cv2
import tifffile

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QMessageBox, QSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt

# ---------- WCS adjustment helper (mutually exclusive scaling) ----------
def _adjust_header_for_scale(hdr, old_shape, new_shape, scale, log_fn=None):
    """
    Adjust common astrometric header keywords to account for a pixel scale change.

    Mutually exclusive scaling:
      - If CD matrix present: divide CD by scale; leave CDELT unchanged.
      - Else if PC (or CROTA) present with CDELT: divide CDELT by scale; leave PC/CROTA unchanged.
      - Else if only CDELT present: divide CDELT by scale.
      - Else: no scale adjustment to WCS terms.

    CRPIX is always multiplied by scale.
    CRVAL stays unchanged.
    NAXIS1/2 updated to new shape if keys exist.
    """
    h = hdr

    def _update_key(key, new_val):
        if key in h:
            old_val = h[key]
            h[key] = new_val
            if log_fn:
                log_fn(f"{key}: {old_val} -> {new_val}")

    # Update NAXIS1/NAXIS2 to new integer sizes if known
    if new_shape is not None:
        ny, nx = int(new_shape[0]), int(new_shape[1])
        if 'NAXIS1' in h:
            _update_key('NAXIS1', nx)
        if 'NAXIS2' in h:
            _update_key('NAXIS2', ny)

    # Scale CRPIX (reference pixel in pixel units)
    for k in ('CRPIX1', 'CRPIX2'):
        if k in h:
            try:
                val = float(h[k])
                _update_key(k, val * float(scale))
            except Exception:
                pass

    # Detect presence of WCS representations
    cd_present    = any(k.upper().startswith('CD')    for k in h.keys())
    pc_present    = any(k.upper().startswith('PC')    for k in h.keys())
    crota_present = ('CROTA2' in h) or ('CROTA1' in h)
    cdelt_present = ('CDELT1' in h) or ('CDELT2' in h)

    # Apply mutually exclusive scaling
    if cd_present:
        if log_fn: log_fn("WCS scale path: CD matrix present -> divide CD by scale; leave CDELT unchanged.")
        for key in list(h.keys()):
            if key.upper().startswith('CD'):
                try:
                    val = float(h[key])
                    _update_key(key, val / float(scale))
                except Exception:
                    pass
        # Do NOT touch CDELT here
    elif cdelt_present and (pc_present or crota_present):
        if log_fn: log_fn("WCS scale path: PC/CROTA with CDELT -> divide CDELT by scale; leave PC/CROTA unchanged.")
        for k in ('CDELT1', 'CDELT2'):
            if k in h:
                try:
                    val = float(h[k])
                    _update_key(k, val / float(scale))
                except Exception:
                    pass
    elif cdelt_present:
        if log_fn: log_fn("WCS scale path: CDELT only -> divide CDELT by scale.")
        for k in ('CDELT1', 'CDELT2'):
            if k in h:
                try:
                    val = float(h[k])
                    _update_key(k, val / float(scale))
                except Exception:
                    pass
    else:
        if log_fn: log_fn("WCS scale path: no CD/PC/CROTA/CDELT found; no WCS scale adjustment applied.")

    return h

# ---------- GUI ----------
class ResizeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Resize Tool")
        self._build_ui()
        self.resize(820, 360)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        grid.addWidget(QLabel("Mode:"), 0, 0)
        self.mode = QComboBox()
        self.mode.addItems([
            "FITS cubic float64 (order=3)",
            "FITS Lanczos4 (OpenCV, keep channels)",
            "Other image Lanczos4 (tif/png/jpg)"
        ])
        grid.addWidget(self.mode, 0, 1, 1, 3)

        grid.addWidget(QLabel("Input file:"), 1, 0)
        self.input_edit = QLineEdit()
        grid.addWidget(self.input_edit, 1, 1, 1, 2)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self._browse_input)
        grid.addWidget(btn_in, 1, 3)

        grid.addWidget(QLabel("Scale numerator (int):"), 2, 0)
        self.num_spin = QSpinBox(); self.num_spin.setRange(1, 64); self.num_spin.setValue(1)
        grid.addWidget(self.num_spin, 2, 1)
        grid.addWidget(QLabel("Scale denominator (int):"), 2, 2)
        self.den_spin = QSpinBox(); self.den_spin.setRange(1, 64); self.den_spin.setValue(1)
        grid.addWidget(self.den_spin, 2, 3)

        grid.addWidget(QLabel("Output file:"), 3, 0)
        self.output_edit = QLineEdit("resized_output.fits")
        grid.addWidget(self.output_edit, 3, 1, 1, 2)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_output)
        grid.addWidget(btn_out, 3, 3)

        # Checkbox: update astrometric solution for larger image
        self.update_wcs_chk = QCheckBox("Update astrometric solution to match larger image")
        self.update_wcs_chk.setChecked(False)
        grid.addWidget(self.update_wcs_chk, 4, 0, 1, 3)

        self.run_btn = QPushButton("Run Resize")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 5, 0)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(self._clear_log)
        grid.addWidget(self.clear_btn, 5, 1)

        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self._show_help)
        grid.addWidget(self.help_btn, 5, 2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 6, 0, 1, 4)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _clear_log(self):
        self.log.clear()

    def _show_help(self):
        QMessageBox.information(self, "Help",
            "Modes:\n"
            "1) FITS cubic float64: uses scipy.ndimage.zoom(order=3) on a 2D FITS image.\n"
            "2) FITS Lanczos4: uses OpenCV LANCZOS4 on multi-plane FITS.\n"
            "3) Other image: reads with tifffile or OpenCV and resizes with OpenCV LANCZOS4.\n\n"
            "Scale = numerator / denominator. Example: 2/1 doubles size, 1/2 halves size.\n\n"
            "Option: 'Update astrometric solution to match larger image' — when checked and\n"
            "scale > 1, WCS header keywords are adjusted and each change is logged (before -> after)."
        )

    def _browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select input file", "", "All Files (*)")
        if fn:
            self.input_edit.setText(fn)

    def _browse_output(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Select output file", "", "All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def _on_run(self):
        in_path = self.input_edit.text().strip()
        out_path = self.output_edit.text().strip()
        num = int(self.num_spin.value())
        den = int(self.den_spin.value())
        if not in_path:
            QMessageBox.warning(self, "Input required", "Select an input file.")
            return
        if not out_path:
            QMessageBox.warning(self, "Output required", "Specify an output filename.")
            return
        scale = float(num) / float(den)
        mode_idx = self.mode.currentIndex()
        update_wcs = bool(self.update_wcs_chk.isChecked())

        try:
            if mode_idx == 0:
                self._resize_fits_cubic(in_path, out_path, scale, update_wcs)
            elif mode_idx == 1:
                self._resize_fits_lanczos(in_path, out_path, scale, update_wcs)
            else:
                self._resize_other_image(in_path, out_path, scale)
            QMessageBox.information(self, "Done", f"Saved resized file: {out_path}")
            self._log("Done:", out_path)
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

    def _resize_fits_cubic(self, in_path, out_path, scale, update_wcs=False):
        with fits.open(in_path) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
        if data is None or data.ndim != 2:
            raise ValueError("FITS cubic mode expects a 2D image in primary HDU.")
        self._log(f"Original shape: {data.shape}, dtype={data.dtype}")
        arr = data.astype(np.float64)
        resized = zoom(arr, (scale, scale), order=3).astype(np.float64)

        out_hdr = hdr.copy()
        if update_wcs and scale > 1.0:
            try:
                self._log(f"Updating WCS for scale={scale} (larger image)...")
                _adjust_header_for_scale(out_hdr,
                                         old_shape=data.shape,
                                         new_shape=resized.shape,
                                         scale=scale,
                                         log_fn=self._log)
                self._log("Adjusted header astrometric keywords for larger image.")
            except Exception as e:
                self._log("Warning: failed to adjust header WCS:", e)

        hdu = fits.PrimaryHDU(resized, header=out_hdr)
        hdu.writeto(out_path, overwrite=True)
        self._log(f"Wrote FITS (float64) shape {resized.shape}")

    def _resize_fits_lanczos(self, in_path, out_path, scale, update_wcs=False):
        with fits.open(in_path) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
        if data is None:
            raise ValueError("No data found in FITS primary HDU.")
        self._log(f"Input FITS shape: {data.shape}, dtype={data.dtype}")

        # Normalize to 0..65535 for OpenCV processing, convert back after resize
        if data.ndim == 3:
            if data.shape[0] <= 4:
                arr = np.transpose(data, (1, 2, 0))  # (C,Y,X)->(Y,X,C)
                orig_channel_first = True
            else:
                arr = data  # (Y,X,C)
                orig_channel_first = False
        elif data.ndim == 2:
            arr = data[..., None]  # (Y,X,1)
            orig_channel_first = False
        else:
            raise ValueError("Unsupported FITS data ndim for Lanczos mode.")

        finite = np.isfinite(arr)
        if not finite.any():
            raise ValueError("FITS contains no finite pixels.")
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        if arr_max == arr_min:
            norm = np.zeros_like(arr, dtype=np.float32)
        else:
            norm = ((arr - arr_min) / (arr_max - arr_min) * 65535.0).astype(np.float32)

        h, w = norm.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        self._log(f"Resizing from ({h},{w}) to ({new_h},{new_w}) using LANCZOS4")
        resized = cv2.resize(norm, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # convert back to original numeric range
        resized_back = resized.astype(np.float64) / 65535.0 * (arr_max - arr_min) + arr_min

        # restore dimensionality to FITS conventions
        if data.ndim == 2:
            resized_back = resized_back[..., 0]
            new_shape_for_wcs = resized_back.shape  # (ny, nx)
        else:
            if orig_channel_first:
                resized_back = np.transpose(resized_back, (2, 0, 1))  # (Y,X,C)->(C,Y,X)
                new_shape_for_wcs = (resized_back.shape[1], resized_back.shape[2])  # (ny, nx)
            else:
                new_shape_for_wcs = (resized_back.shape[0], resized_back.shape[1])  # (ny, nx)

        out_hdr = hdr.copy()
        if update_wcs and scale > 1.0:
            try:
                self._log(f"Updating WCS for scale={scale} (larger image)...")
                _adjust_header_for_scale(out_hdr,
                                         old_shape=data.shape,
                                         new_shape=new_shape_for_wcs,
                                         scale=scale,
                                         log_fn=self._log)
                self._log("Adjusted header astrometric keywords for larger image.")
            except Exception as e:
                self._log("Warning: failed to adjust header WCS:", e)

        hdu = fits.PrimaryHDU(resized_back, header=out_hdr)
        hdu.writeto(out_path, overwrite=True)
        self._log(f"Wrote FITS (Lanczos) shape {resized_back.shape}")

    def _resize_other_image(self, in_path, out_path, scale):
        # Try tifffile first; fallback to cv2
        try:
            img = tifffile.imread(in_path)
            self._log("Read image with tifffile")
        except Exception:
            img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {in_path}")
            self._log("Read image with OpenCV")
        self._log(f"Original image shape: {img.shape}, dtype={img.dtype}")

        h = img.shape[0]; w = img.shape[1]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        self._log(f"Resizing to ({new_h},{new_w}) with LANCZOS4")
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Save with tifffile if possible, else cv2.imwrite
        try:
            tifffile.imwrite(out_path, resized)
            self._log("Saved with tifffile")
        except Exception:
            ok = cv2.imwrite(out_path, resized)
            if not ok:
                raise IOError("Failed to write output image with cv2.imwrite")
            self._log("Saved with OpenCV imwrite")

def main():
    app = QApplication(sys.argv)
    w = ResizeWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()