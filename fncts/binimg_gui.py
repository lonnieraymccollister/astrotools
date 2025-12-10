#!/usr/bin/env python3
"""
binimg_gui.py
PyQt6 GUI to bin color FITS images (supports (C,Y,X) or (Y,X,C) layouts).
Writes output as channel-first (C,Y,X) by default and copies header.
"""#!/usr/bin/env python3
"""
binimg_gui.py
PyQt6 GUI to bin color FITS images (supports (C,Y,X) or (Y,X,C) layouts).
Writes output as channel-first (C,Y,X) by default and copies header.
"""
import sys
import os
import traceback
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import tifffile

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QSpinBox, QTextEdit, QMessageBox, QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

# ---------- helpers ----------
def read_fits_primary(path):
    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()
    return data, header

def write_fits_primary(path, data, header=None):
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(path, overwrite=True)

def qpix_from_rgb(arr):
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 2:
        arr8 = np.clip(a, 0, 255).astype(np.uint8)
        h, w = arr8.shape
        qimg = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    if a.ndim == 3:
        # expect RGB for preview; if C,Y,X convert first
        if a.shape[0] == 3:
            rgb = np.transpose(a, (1,2,0))
        else:
            rgb = a
        # scale floats / large ints -> uint8 for preview
        if rgb.dtype != np.uint8:
            b = rgb.astype(np.float64)
            mn, mx = np.nanmin(b), np.nanmax(b)
            if mx > mn:
                b8 = ((b - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                b8 = np.zeros_like(b, dtype=np.uint8)
        else:
            b8 = rgb
        if b8.shape[2] >= 3:
            rgb8 = b8[:, :, :3]
            h, w, ch = rgb8.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg)
    return None

def downsample_for_preview(arr, maxdim=(320,320)):
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[0] == 3:
        img = np.transpose(arr, (1,2,0))
    else:
        img = arr
    h, w = img.shape[:2]
    mw, mh = maxdim
    scale = min(1.0, mw / w, mh / h)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        try:
            import cv2
            if img.ndim == 2:
                disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            disp = img
        return disp
    return img

def bin_array_blockwise(img, bin_size):
    # img is 2D numpy array
    h, w = img.shape
    nh = h // bin_size
    nw = w // bin_size
    # trim edges
    img_trim = img[:nh*bin_size, :nw*bin_size]
    if bin_size == 1:
        return img_trim
    reshaped = img_trim.reshape(nh, bin_size, nw, bin_size)
    return reshaped.mean(axis=(1,3))

# ---------- GUI ----------
class BinImgWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bin Color FITS")
        self._build_ui()
        self.resize(920, 520)
        self.input_path = None
        self.input_header = None
        self.input_data = None  # as loaded (could be C,Y,X or Y,X,C)

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

        grid.addWidget(QLabel("Output FITS:"), 1, 0)
        self.output_edit = QLineEdit("binned_output.fits")
        grid.addWidget(self.output_edit, 1, 1, 1, 3)
        btn_out = QPushButton("Browse Save")
        btn_out.clicked.connect(self._browse_save)
        grid.addWidget(btn_out, 1, 4)

        grid.addWidget(QLabel("Binning factor (int):"), 2, 0)
        self.bin_spin = QSpinBox(); self.bin_spin.setRange(1, 1000); self.bin_spin.setValue(25)
        grid.addWidget(self.bin_spin, 2, 1)

        grid.addWidget(QLabel("Input layout:"), 2, 2)
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Auto detect", "C,Y,X (channels first)", "Y,X,C (channels last)"])
        grid.addWidget(self.layout_combo, 2, 3)

        self.patch_wcs_chk = QCheckBox("Patch WCS CRPIX/CDELT for integer bin")
        grid.addWidget(self.patch_wcs_chk, 3, 0, 1, 3)

        self.load_btn = QPushButton("Load & Preview")
        self.load_btn.clicked.connect(self._load_input)
        grid.addWidget(self.load_btn, 3, 3)

        self.run_btn = QPushButton("Run Binning")
        self.run_btn.clicked.connect(self._run_binning)
        grid.addWidget(self.run_btn, 4, 0)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(lambda: self.log.clear())
        grid.addWidget(self.clear_btn, 4, 1)

        # Previews and log
        self.in_preview = QLabel("Input preview")
        self.in_preview.setFixedSize(360, 260)
        self.in_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.in_preview, 5, 0, 6, 2)

        self.out_preview = QLabel("Binned preview")
        self.out_preview.setFixedSize(360, 260)
        self.out_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.out_preview, 5, 2, 6, 2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 11, 0, 3, 4)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open input FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.input_edit.setText(fn)

    def _browse_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save binned FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def _load_input(self):
        path = self.input_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Missing file", "Select an existing input FITS first")
            return
        try:
            data, header = read_fits_primary(path)
            if data is None:
                raise ValueError("No data in primary HDU")
            self.input_path = path
            self.input_header = header
            self.input_data = np.array(data)
            self._log(f"Loaded {Path(path).name} shape={self.input_data.shape} dtype={self.input_data.dtype}")
            # Try to form an RGB preview
            arr = self._to_preview_rgb(self.input_data)
            disp = downsample_for_preview(arr)
            pix = qpix_from_rgb(disp)
            if pix:
                self.in_preview.setPixmap(pix.scaled(self.in_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"{e}\n\n{tb}")

    def _to_preview_rgb(self, data):
        arr = np.array(data)
        if arr.ndim == 3:
            # detect layout
            mode = self.layout_combo.currentIndex()
            if mode == 0:
                # auto
                if arr.shape[0] == 3:
                    rgb = np.transpose(arr, (1,2,0))
                elif arr.shape[2] == 3:
                    rgb = arr
                else:
                    # try to select first three planes
                    if arr.shape[0] >= 3:
                        rgb = np.transpose(arr[:3], (1,2,0))
                    elif arr.shape[2] >= 3:
                        rgb = arr[..., :3]
                    else:
                        # collapse channels
                        rgb = np.stack([arr[...,0]]*3, axis=-1)
            elif mode == 1:
                rgb = np.transpose(arr, (1,2,0))
            else:
                rgb = arr
        else:
            # grayscale -> replicate
            rgb = np.stack([arr]*3, axis=-1)
        # scale to 0..255
        b = rgb.astype(np.float64)
        mn, mx = np.nanmin(b), np.nanmax(b)
        if mx > mn:
            out = ((b - mn) / (mx - mn) * 255.0).astype(np.uint8)
        else:
            out = np.zeros_like(b, dtype=np.uint8)
        return out

    def _patch_wcs_header(self, header, bin_factor):
        """
        Adjust WCS for integer binning (smaller image). This function:
        - Logs before/after values for CRPIX, CDELT, CD*, NAXIS1/2 (if present).
        - Uses mutually exclusive scaling: if CD matrix present, scale CD by bin_factor;
          else scale CDELT by bin_factor (leaving PC/CROTA unchanged).
        - Adjusts CRPIX using the standard pixel-centering formula:
            new_crpix = (orig_crpix - 0.5) / bin_factor + 0.5
        - Returns a new header (copy) with updated WCS keywords. Falls back to
          manual header edits if WCS conversion fails.
        """
        hdr = header.copy()
        try:
            # Log existing relevant keys (if present)
            def _log_key_if_present(k):
                if k in hdr:
                    self._log(f"{k} (before): {hdr[k]}")

            for k in ('NAXIS1','NAXIS2','CRPIX1','CRPIX2','CDELT1','CDELT2'):
                _log_key_if_present(k)
            for key in list(hdr.keys()):
                if key.upper().startswith('CD'):
                    self._log(f"{key} (before): {hdr[key]}")

            # Build WCS and read current values
            w = WCS(hdr)
            orig_crpix = np.array(w.wcs.crpix, dtype=float)  # 1-based FITS convention inside WCS object
            # cdelt may be None if CD matrix used
            orig_cdelt = None
            try:
                orig_cdelt = np.array(w.wcs.cdelt, dtype=float)
            except Exception:
                orig_cdelt = None
            orig_cd = None
            try:
                if hasattr(w.wcs, 'cd') and w.wcs.cd is not None:
                    orig_cd = np.array(w.wcs.cd, dtype=float)
            except Exception:
                orig_cd = None

            # compute new CRPIX
            new_crpix = (orig_crpix - 0.5) / float(bin_factor) + 0.5

            # detect presence of CD / PC / CROTA / CDELT in header
            cd_present = any(k.upper().startswith('CD') for k in hdr.keys())
            pc_present = any(k.upper().startswith('PC') for k in hdr.keys())
            crota_present = ('CROTA2' in hdr) or ('CROTA1' in hdr)
            cdelt_present = ('CDELT1' in hdr) or ('CDELT2' in hdr)

            # Mutually exclusive scaling: for binning (smaller image), pixel size increases by bin_factor
            if cd_present and orig_cd is not None:
                # scale CD matrix by bin_factor
                new_cd = orig_cd * float(bin_factor)
                # apply to WCS
                w.wcs.cd = new_cd
                w.wcs.crpix = new_crpix
                hdr_new = w.to_header(relax=True)
                # log changes
                for key in list(hdr.keys()):
                    if key.upper().startswith('CD'):
                        old = hdr[key]
                        new = hdr_new.get(key, hdr[key])
                        self._log(f"{key}: {old} -> {new}")
                if 'CRPIX1' in hdr:
                    self._log(f"CRPIX1: {hdr.get('CRPIX1')} -> {hdr_new.get('CRPIX1')}")
                if 'CRPIX2' in hdr:
                    self._log(f"CRPIX2: {hdr.get('CRPIX2')} -> {hdr_new.get('CRPIX2')}")
                # merge new WCS keys into header copy
                for k in hdr_new:
                    hdr[k] = hdr_new[k]
                # ensure CDELT keys remain unchanged (do not double-scale)
                # done by using CD path only
                self._log("WCS scale path: CD matrix present -> scaled CD by bin factor; left CDELT unchanged.")
            elif cdelt_present and (pc_present or crota_present):
                # scale CDELT by bin_factor, leave PC/CROTA unchanged
                # compute new cdelt
                if orig_cdelt is None:
                    # fallback: read from header directly
                    orig_cdelt = np.array([float(hdr.get('CDELT1', 0.0)), float(hdr.get('CDELT2', 0.0))], dtype=float)
                new_cdelt = orig_cdelt * float(bin_factor)
                # apply to WCS
                w.wcs.cdelt = new_cdelt
                w.wcs.crpix = new_crpix
                hdr_new = w.to_header(relax=True)
                # log changes
                if 'CDELT1' in hdr:
                    self._log(f"CDELT1: {hdr.get('CDELT1')} -> {hdr_new.get('CDELT1')}")
                if 'CDELT2' in hdr:
                    self._log(f"CDELT2: {hdr.get('CDELT2')} -> {hdr_new.get('CDELT2')}")
                if 'CRPIX1' in hdr:
                    self._log(f"CRPIX1: {hdr.get('CRPIX1')} -> {hdr_new.get('CRPIX1')}")
                if 'CRPIX2' in hdr:
                    self._log(f"CRPIX2: {hdr.get('CRPIX2')} -> {hdr_new.get('CRPIX2')}")
                for k in hdr_new:
                    hdr[k] = hdr_new[k]
                self._log("WCS scale path: PC/CROTA with CDELT -> scaled CDELT by bin factor; left PC/CROTA unchanged.")
            elif cdelt_present:
                # only CDELT present -> scale CDELT by bin_factor
                if orig_cdelt is None:
                    orig_cdelt = np.array([float(hdr.get('CDELT1', 0.0)), float(hdr.get('CDELT2', 0.0))], dtype=float)
                new_cdelt = orig_cdelt * float(bin_factor)
                w.wcs.cdelt = new_cdelt
                w.wcs.crpix = new_crpix
                hdr_new = w.to_header(relax=True)
                if 'CDELT1' in hdr:
                    self._log(f"CDELT1: {hdr.get('CDELT1')} -> {hdr_new.get('CDELT1')}")
                if 'CDELT2' in hdr:
                    self._log(f"CDELT2: {hdr.get('CDELT2')} -> {hdr_new.get('CDELT2')}")
                if 'CRPIX1' in hdr:
                    self._log(f"CRPIX1: {hdr.get('CRPIX1')} -> {hdr_new.get('CRPIX1')}")
                if 'CRPIX2' in hdr:
                    self._log(f"CRPIX2: {hdr.get('CRPIX2')} -> {hdr_new.get('CRPIX2')}")
                for k in hdr_new:
                    hdr[k] = hdr_new[k]
                self._log("WCS scale path: CDELT only -> scaled CDELT by bin factor.")
            else:
                # No CD/CDELT/PC/CROTA found; attempt minimal manual CRPIX update and log
                self._log("WCS scale path: no CD/PC/CROTA/CDELT found; applying minimal CRPIX update if present.")
                if 'CRPIX1' in hdr:
                    old = hdr['CRPIX1']
                    new = float(old - 0.5) / float(bin_factor) + 0.5
                    hdr['CRPIX1'] = new
                    self._log(f"CRPIX1: {old} -> {new}")
                if 'CRPIX2' in hdr:
                    old = hdr['CRPIX2']
                    new = float(old - 0.5) / float(bin_factor) + 0.5
                    hdr['CRPIX2'] = new
                    self._log(f"CRPIX2: {old} -> {new}")

            # Update NAXIS1/NAXIS2 to reflect trimmed binned image if present
            # new sizes are floor(original / bin_factor)
            try:
                if 'NAXIS1' in hdr and 'NAXIS2' in hdr:
                    old_nx = int(header.get('NAXIS1', 0))
                    old_ny = int(header.get('NAXIS2', 0))
                    new_nx = old_nx // int(bin_factor)
                    new_ny = old_ny // int(bin_factor)
                    self._log(f"NAXIS1: {old_nx} -> {new_nx}")
                    self._log(f"NAXIS2: {old_ny} -> {new_ny}")
                    hdr['NAXIS1'] = new_nx
                    hdr['NAXIS2'] = new_ny
            except Exception:
                pass

            # Ensure explicit CDELT keys exist if we updated cdelt path
            try:
                if 'CDELT1' in hdr and 'CDELT2' in hdr:
                    self._log(f"CDELT1 (after): {hdr.get('CDELT1')}")
                    self._log(f"CDELT2 (after): {hdr.get('CDELT2')}")
            except Exception:
                pass

            return hdr
        except Exception as e:
            # fallback: try minimal manual edits and log them
            self._log("WCS patch via astropy.wcs failed, falling back to manual header edits:", e)
            hdr2 = header.copy()
            try:
                if 'CRPIX1' in hdr2:
                    old = hdr2['CRPIX1']
                    new = float(old - 0.5) / float(bin_factor) + 0.5
                    hdr2['CRPIX1'] = new
                    self._log(f"CRPIX1: {old} -> {new}")
                if 'CRPIX2' in hdr2:
                    old = hdr2['CRPIX2']
                    new = float(old - 0.5) / float(bin_factor) + 0.5
                    hdr2['CRPIX2'] = new
                    self._log(f"CRPIX2: {old} -> {new}")
                if 'CDELT1' in hdr2:
                    old = float(hdr2['CDELT1'])
                    new = old * float(bin_factor)
                    hdr2['CDELT1'] = (new, "deg/pix")
                    self._log(f"CDELT1: {old} -> {new}")
                if 'CDELT2' in hdr2:
                    old = float(hdr2['CDELT2'])
                    new = old * float(bin_factor)
                    hdr2['CDELT2'] = (new, "deg/pix")
                    self._log(f"CDELT2: {old} -> {new}")
            except Exception as ex:
                self._log("Fallback header edits failed:", ex)
            return hdr2

    def _run_binning(self):
        if self.input_data is None:
            self._load_input()
            if self.input_data is None:
                return
        outpath = self.output_edit.text().strip()
        if not outpath:
            QMessageBox.warning(self, "Output required", "Specify an output filename")
            return
        bin_size = int(self.bin_spin.value())
        layout_choice = self.layout_combo.currentIndex()
        try:
            arr = self.input_data
            # interpret layout
            if arr.ndim != 3:
                raise ValueError("Input FITS must be 3D (channels + Y + X) for color binning")
            if layout_choice == 0:
                # auto detect
                if arr.shape[0] == 3:
                    # channels first (C,Y,X)
                    cfirst = True
                elif arr.shape[2] == 3:
                    cfirst = False
                else:
                    # assume channels-first if first dim small
                    cfirst = (arr.shape[0] <= 4)
            else:
                cfirst = (layout_choice == 1)

            if cfirst:
                C, H, W = arr.shape
                # compute trimmed sizes
                new_h = H // bin_size
                new_w = W // bin_size
                if new_h < 1 or new_w < 1:
                    raise ValueError("Binning factor too large for image dimensions")
                binned = np.zeros((C, new_h, new_w), dtype=np.float64)
                for ci in range(C):
                    bchan = bin_array_blockwise(arr[ci], bin_size)
                    binned[ci] = bchan
            else:
                H, W, C = arr.shape
                new_h = H // bin_size
                new_w = W // bin_size
                if new_h < 1 or new_w < 1:
                    raise ValueError("Binning factor too large for image dimensions")
                binned_ch = []
                for ci in range(C):
                    ch = arr[..., ci]
                    bchan = bin_array_blockwise(ch, bin_size)
                    binned_ch.append(bchan)
                binned = np.stack(binned_ch, axis=0)  # C, new_h, new_w

            # preserve dtype meaning: cast back to reasonable dtype
            # if original integer, round and cast; else keep float64
            if np.issubdtype(self.input_data.dtype, np.integer):
                binned_out = np.rint(binned).astype(self.input_data.dtype)
            else:
                binned_out = binned.astype(np.float32)

            # header handling
            hdr_out = self.input_header.copy() if self.input_header is not None else None
            if self.patch_wcs_chk.isChecked() and hdr_out is not None:
                try:
                    hdr_out = self._patch_wcs_header(hdr_out, bin_size)
                    self._log("WCS patched for binning factor:", bin_size)
                except Exception as ex:
                    self._log("WCS patch failed, continuing with original header:", ex)

            # write output as C,Y,X (channel-first) to match your original pipeline
            write_fits_primary(outpath, binned_out, header=hdr_out)
            self._log(f"Wrote binned FITS: {outpath} shape={binned_out.shape} dtype={binned_out.dtype}")
            # preview binned result (convert to previewable RGB)
            preview_rgb = None
            if binned_out.ndim == 3:
                if binned_out.shape[0] == 3:
                    preview_rgb = np.transpose(binned_out, (1,2,0))
                else:
                    preview_rgb = np.transpose(binned_out[:3], (1,2,0))
            else:
                preview_rgb = binned_out
            disp = downsample_for_preview(preview_rgb)
            pix = qpix_from_rgb(disp)
            if pix:
                self.out_preview.setPixmap(pix.scaled(self.out_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            QMessageBox.information(self, "Done", f"Wrote binned FITS: {outpath}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = BinImgWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
import sys
import os
import traceback
from pathlib import Path

import numpy as np
from astropy.io import fits
import tifffile

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QSpinBox, QTextEdit, QMessageBox, QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

# ---------- helpers ----------
def read_fits_primary(path):
    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()
    return data, header

def write_fits_primary(path, data, header=None):
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(path, overwrite=True)

def qpix_from_rgb(arr):
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 2:
        arr8 = np.clip(a, 0, 255).astype(np.uint8)
        h, w = arr8.shape
        qimg = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    if a.ndim == 3:
        # expect RGB for preview; if C,Y,X convert first
        if a.shape[0] == 3:
            rgb = np.transpose(a, (1,2,0))
        else:
            rgb = a
        # scale floats / large ints -> uint8 for preview
        if rgb.dtype != np.uint8:
            b = rgb.astype(np.float64)
            mn, mx = np.nanmin(b), np.nanmax(b)
            if mx > mn:
                b8 = ((b - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                b8 = np.zeros_like(b, dtype=np.uint8)
        else:
            b8 = rgb
        if b8.shape[2] >= 3:
            rgb8 = b8[:, :, :3]
            h, w, ch = rgb8.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg)
    return None

def downsample_for_preview(arr, maxdim=(320,320)):
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[0] == 3:
        img = np.transpose(arr, (1,2,0))
    else:
        img = arr
    h, w = img.shape[:2]
    mw, mh = maxdim
    scale = min(1.0, mw / w, mh / h)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        try:
            import cv2
            if img.ndim == 2:
                disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            disp = img
        return disp
    return img

def bin_array_blockwise(img, bin_size):
    # img is 2D numpy array
    h, w = img.shape
    nh = h // bin_size
    nw = w // bin_size
    # trim edges
    img_trim = img[:nh*bin_size, :nw*bin_size]
    if bin_size == 1:
        return img_trim
    reshaped = img_trim.reshape(nh, bin_size, nw, bin_size)
    return reshaped.mean(axis=(1,3))

# ---------- GUI ----------
class BinImgWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bin Color FITS")
        self._build_ui()
        self.resize(920, 520)
        self.input_path = None
        self.input_header = None
        self.input_data = None  # as loaded (could be C,Y,X or Y,X,C)

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

        grid.addWidget(QLabel("Output FITS:"), 1, 0)
        self.output_edit = QLineEdit("binned_output.fits")
        grid.addWidget(self.output_edit, 1, 1, 1, 3)
        btn_out = QPushButton("Browse Save")
        btn_out.clicked.connect(self._browse_save)
        grid.addWidget(btn_out, 1, 4)

        grid.addWidget(QLabel("Binning factor (int):"), 2, 0)
        self.bin_spin = QSpinBox(); self.bin_spin.setRange(1, 1000); self.bin_spin.setValue(25)
        grid.addWidget(self.bin_spin, 2, 1)

        grid.addWidget(QLabel("Input layout:"), 2, 2)
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Auto detect", "C,Y,X (channels first)", "Y,X,C (channels last)"])
        grid.addWidget(self.layout_combo, 2, 3)

        self.patch_wcs_chk = QCheckBox("Patch WCS CRPIX/CDELT for integer bin")
        grid.addWidget(self.patch_wcs_chk, 3, 0, 1, 3)

        self.load_btn = QPushButton("Load & Preview")
        self.load_btn.clicked.connect(self._load_input)
        grid.addWidget(self.load_btn, 3, 3)

        self.run_btn = QPushButton("Run Binning")
        self.run_btn.clicked.connect(self._run_binning)
        grid.addWidget(self.run_btn, 4, 0)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(lambda: self.log.clear())
        grid.addWidget(self.clear_btn, 4, 1)

        # Previews and log
        self.in_preview = QLabel("Input preview")
        self.in_preview.setFixedSize(360, 260)
        self.in_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.in_preview, 5, 0, 6, 2)

        self.out_preview = QLabel("Binned preview")
        self.out_preview.setFixedSize(360, 260)
        self.out_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.out_preview, 5, 2, 6, 2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 11, 0, 3, 4)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open input FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.input_edit.setText(fn)

    def _browse_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save binned FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def _load_input(self):
        path = self.input_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Missing file", "Select an existing input FITS first")
            return
        try:
            data, header = read_fits_primary(path)
            if data is None:
                raise ValueError("No data in primary HDU")
            self.input_path = path
            self.input_header = header
            self.input_data = np.array(data)
            self._log(f"Loaded {Path(path).name} shape={self.input_data.shape} dtype={self.input_data.dtype}")
            # Try to form an RGB preview
            arr = self._to_preview_rgb(self.input_data)
            disp = downsample_for_preview(arr)
            pix = qpix_from_rgb(disp)
            if pix:
                self.in_preview.setPixmap(pix.scaled(self.in_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"{e}\n\n{tb}")

    def _to_preview_rgb(self, data):
        arr = np.array(data)
        if arr.ndim == 3:
            # detect layout
            mode = self.layout_combo.currentIndex()
            if mode == 0:
                # auto
                if arr.shape[0] == 3:
                    rgb = np.transpose(arr, (1,2,0))
                elif arr.shape[2] == 3:
                    rgb = arr
                else:
                    # try to select first three planes
                    if arr.shape[0] >= 3:
                        rgb = np.transpose(arr[:3], (1,2,0))
                    elif arr.shape[2] >= 3:
                        rgb = arr[..., :3]
                    else:
                        # collapse channels
                        rgb = np.stack([arr[...,0]]*3, axis=-1)
            elif mode == 1:
                rgb = np.transpose(arr, (1,2,0))
            else:
                rgb = arr
        else:
            # grayscale -> replicate
            rgb = np.stack([arr]*3, axis=-1)
        # scale to 0..255
        b = rgb.astype(np.float64)
        mn, mx = np.nanmin(b), np.nanmax(b)
        if mx > mn:
            out = ((b - mn) / (mx - mn) * 255.0).astype(np.uint8)
        else:
            out = np.zeros_like(b, dtype=np.uint8)
        return out

    def _patch_wcs_header(self, header, bin_factor):
        # minimal safe attempt: adjust CRPIX and CDELT if present
        from astropy.wcs import WCS
        try:
            w = WCS(header)
            orig_crpix = w.wcs.crpix.copy()
            orig_cdelt = w.wcs.cdelt.copy()
            new_crpix = (orig_crpix - 0.5) / bin_factor + 0.5
            new_cdelt = orig_cdelt * bin_factor
            w.wcs.crpix = new_crpix
            w.wcs.cdelt = new_cdelt
            hdr_cd = w.to_header(relax=True)
            hdr = header.copy()
            for k in hdr_cd:
                hdr[k] = hdr_cd[k]
            # explicit CDELT/CUNIT if available
            hdr['CDELT1'] = (new_cdelt[0], "deg/pix")
            hdr['CDELT2'] = (new_cdelt[1], "deg/pix")
            return hdr
        except Exception:
            # fallback: return original header unchanged
            return header.copy()

    def _run_binning(self):
        if self.input_data is None:
            self._load_input()
            if self.input_data is None:
                return
        outpath = self.output_edit.text().strip()
        if not outpath:
            QMessageBox.warning(self, "Output required", "Specify an output filename")
            return
        bin_size = int(self.bin_spin.value())
        layout_choice = self.layout_combo.currentIndex()
        try:
            arr = self.input_data
            # interpret layout
            if arr.ndim != 3:
                raise ValueError("Input FITS must be 3D (channels + Y + X) for color binning")
            if layout_choice == 0:
                # auto detect
                if arr.shape[0] == 3:
                    # channels first (C,Y,X)
                    cfirst = True
                elif arr.shape[2] == 3:
                    cfirst = False
                else:
                    # assume channels-first if first dim small
                    cfirst = (arr.shape[0] <= 4)
            else:
                cfirst = (layout_choice == 1)

            if cfirst:
                C, H, W = arr.shape
                # compute trimmed sizes
                new_h = H // bin_size
                new_w = W // bin_size
                if new_h < 1 or new_w < 1:
                    raise ValueError("Binning factor too large for image dimensions")
                binned = np.zeros((C, new_h, new_w), dtype=np.float64)
                for ci in range(C):
                    bchan = bin_array_blockwise(arr[ci], bin_size)
                    binned[ci] = bchan
            else:
                H, W, C = arr.shape
                new_h = H // bin_size
                new_w = W // bin_size
                if new_h < 1 or new_w < 1:
                    raise ValueError("Binning factor too large for image dimensions")
                binned_ch = []
                for ci in range(C):
                    ch = arr[..., ci]
                    bchan = bin_array_blockwise(ch, bin_size)
                    binned_ch.append(bchan)
                binned = np.stack(binned_ch, axis=0)  # C, new_h, new_w

            # preserve dtype meaning: cast back to reasonable dtype
            # if original integer, round and cast; else keep float64
            if np.issubdtype(self.input_data.dtype, np.integer):
                binned_out = np.rint(binned).astype(self.input_data.dtype)
            else:
                binned_out = binned.astype(np.float32)

            # header handling
            hdr_out = self.input_header.copy() if self.input_header is not None else None
            if self.patch_wcs_chk.isChecked() and hdr_out is not None:
                try:
                    hdr_out = self._patch_wcs_header(hdr_out, bin_size)
                    self._log("WCS patched for binning factor:", bin_size)
                except Exception as ex:
                    self._log("WCS patch failed, continuing with original header:", ex)

            # write output as C,Y,X (channel-first) to match your original pipeline
            write_fits_primary(outpath, binned_out, header=hdr_out)
            self._log(f"Wrote binned FITS: {outpath} shape={binned_out.shape} dtype={binned_out.dtype}")
            # preview binned result (convert to previewable RGB)
            preview_rgb = None
            if binned_out.ndim == 3:
                if binned_out.shape[0] == 3:
                    preview_rgb = np.transpose(binned_out, (1,2,0))
                else:
                    preview_rgb = np.transpose(binned_out[:3], (1,2,0))
            else:
                preview_rgb = binned_out
            disp = downsample_for_preview(preview_rgb)
            pix = qpix_from_rgb(disp)
            if pix:
                self.out_preview.setPixmap(pix.scaled(self.out_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            QMessageBox.information(self, "Done", f"Wrote binned FITS: {outpath}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = BinImgWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()