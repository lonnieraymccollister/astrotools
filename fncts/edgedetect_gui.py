#!/usr/bin/env python3
"""
edgedetect_gui.py
PyQt6 GUI to run Sobel and/or Canny edge detection on a color image and save results.

Features added:
- Load a normalized 32-bit FITS image, convert to 8-bit for processing/preview.
- If a FITS is not normalized and the "Check FITS normalized" checkbox is checked,
  the GUI will launch normalize_gui.py and abort the current load so the user can normalize first.
- After processing, save the processed result as a normalized 32-bit FITS file:
  * If "Normalize output FITS to [0,1]" is checked, the output is normalized to [0,1].
  * Otherwise, if the input was a FITS and we recorded its original min/max, the processed 8-bit
    values are remapped back into the original float range.
- Checkbox to offer to open the saved FITS in Siril (default checked).
"""

import sys
import os
import traceback
import subprocess
import shutil
from pathlib import Path

import numpy as np
import cv2
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QMessageBox, QSpinBox,
    QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage


def ndarray_to_qpix(arr):
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 2:
        # grayscale -> convert to uint8 for preview
        if a.dtype != np.uint8:
            mn, mx = np.nanmin(a), np.nanmax(a)
            if mx > mn:
                a8 = ((a - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                a8 = np.zeros_like(a, dtype=np.uint8)
        else:
            a8 = a
        h, w = a8.shape
        qimg = QImage(a8.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    else:
        # color: OpenCV uses BGR, convert to RGB for display
        if a.shape[2] >= 3:
            if a.dtype != np.uint8:
                b = a.astype(np.float64)
                mn, mx = np.nanmin(b), np.nanmax(b)
                if mx > mn:
                    b8 = ((b - mn) / (mx - mn) * 255.0).astype(np.uint8)
                else:
                    b8 = np.zeros_like(b, dtype=np.uint8)
            else:
                b8 = a
            rgb = cv2.cvtColor(b8[:, :, :3], cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg)
        else:
            # fallback: collapse to grayscale preview
            gray = a[:, :, 0]
            return ndarray_to_qpix(gray)


def scale_float_to_uint8(img):
    arr = np.asarray(img, dtype=np.float64)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    out = ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
    return out


class EdgeDetectWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge Detection (Sobel / Canny)")
        self._build_ui()
        self.resize(920, 520)
        self.loaded_image = None  # raw image as read by OpenCV or converted uint8 preview
        self.loaded_from_fits = False
        self.orig_min = None
        self.orig_max = None
        self.last_output_image = None  # store last processed uint8 image for saving

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        grid.addWidget(QLabel("Input image:"), 0, 0)
        self.input_edit = QLineEdit()
        grid.addWidget(self.input_edit, 0, 1, 1, 3)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self._browse_input)
        grid.addWidget(btn_in, 0, 4)

        # Checkbox: if checked, attempt to load FITS as normalized 32-bit and verify normalization
        self.check_fits_normalized = QCheckBox("Check FITS normalized to [0,1] on load")
        self.check_fits_normalized.setChecked(True)
        grid.addWidget(self.check_fits_normalized, 0, 5, 1, 1)

        grid.addWidget(QLabel("Method:"), 1, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Sobel", "Canny", "Both"])
        grid.addWidget(self.method_combo, 1, 1)

        grid.addWidget(QLabel("Lower threshold (Canny):"), 1, 2)
        self.lower_spin = QSpinBox(); self.lower_spin.setRange(0, 1000); self.lower_spin.setValue(100)
        grid.addWidget(self.lower_spin, 1, 3)

        grid.addWidget(QLabel("Upper threshold (Canny):"), 1, 4)
        self.upper_spin = QSpinBox(); self.upper_spin.setRange(0, 1000); self.upper_spin.setValue(200)
        grid.addWidget(self.upper_spin, 1, 5)

        grid.addWidget(QLabel("Output basename (no suffix):"), 2, 0)
        self.output_edit = QLineEdit("edges_output")
        grid.addWidget(self.output_edit, 2, 1, 1, 3)
        btn_out = QPushButton("Browse Folder")
        btn_out.clicked.connect(self._browse_output_folder)
        grid.addWidget(btn_out, 2, 4)

        self.preview_btn = QPushButton("Load & Preview")
        self.preview_btn.clicked.connect(self._load_preview)
        grid.addWidget(self.preview_btn, 3, 0)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 3, 1)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(lambda: self.log.clear())
        grid.addWidget(self.clear_btn, 3, 2)

        # Save buttons
        self.savefits_btn = QPushButton("Save Output as FITS (8-bit values)")
        self.savefits_btn.clicked.connect(self._save_output_as_fits)
        grid.addWidget(self.savefits_btn, 3, 3)

        self.save_norm_fits_btn = QPushButton("Save Processed as Normalized FITS (32-bit)")
        self.save_norm_fits_btn.clicked.connect(self._save_processed_as_normalized_fits)
        grid.addWidget(self.save_norm_fits_btn, 3, 4)

        # Checkbox: offer to open result in Siril after save
        self.check_open_in_siril = QCheckBox("Offer to open result in Siril after save")
        self.check_open_in_siril.setChecked(True)
        grid.addWidget(self.check_open_in_siril, 3, 5, 1, 1)

        # Checkbox: normalize output FITS to [0,1]
        self.check_normalize_output = QCheckBox("Normalize output FITS to [0,1]")
        self.check_normalize_output.setChecked(True)
        grid.addWidget(self.check_normalize_output, 4, 5, 1, 1)

        # Preview label and log
        self.preview_label = QLabel("Preview")
        self.preview_label.setFixedSize(560, 360)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.preview_label, 4, 0, 6, 3)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 4, 3, 6, 3)

    # -------------------------
    # Helpers for FITS normalization check and launching normalizer
    # -------------------------
    def _check_normalized_0_1(self, arr, tol=1e-8):
        """
        Return True if arr min is approximately 0 and max approximately 1 within tol.
        Works for 2D and 3D arrays.
        """
        if arr is None or arr.size == 0:
            return False
        amin = float(np.nanmin(arr))
        amax = float(np.nanmax(arr))
        if np.isnan(amin) or np.isnan(amax):
            return False
        return (abs(amin - 0.0) <= tol) and (abs(amax - 1.0) <= tol)

    def _warn_and_launch_normalize(self, filepath, which="FITS"):
        """Show warning and attempt to launch normalize_gui.py (non-blocking)."""
        QMessageBox.warning(
            self,
            f"{which} Not Normalized",
            f"{which} is not normalized to [0,1].\n"
            "normalize_gui.py will be launched so you can normalize the file."
        )
        launched = self._launch_normalize_gui(filepath)
        if launched:
            self._log(f"Launched normalize_gui.py for {filepath}")
        else:
            self._log("normalize_gui.py not found; please normalize the file manually.")
        return launched

    def _launch_normalize_gui(self, filepath):
        """
        Launch normalize_gui.py with the given filepath using the same Python interpreter.
        Non-blocking. Returns True if launch succeeded, False otherwise.
        """
        try:
            script = Path("normalize_gui.py")
            if not script.exists():
                script = Path(__file__).resolve().parent / "normalize_gui.py"
            if not script.exists():
                return False
            subprocess.Popen([sys.executable, str(script), str(filepath)])
            return True
        except Exception:
            return False

    # -------------------------
    # I/O and UI helpers
    # -------------------------
    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.fits);;All Files (*)")
        if fn:
            self.input_edit.setText(fn)

    def _browse_output_folder(self):
        dn = QFileDialog.getExistingDirectory(self, "Select output folder", "")
        if dn:
            base = os.path.basename(self.output_edit.text()) or "edges_output"
            self.output_edit.setText(os.path.join(dn, base))

    # -------------------------
    # Load & preview (handles FITS specially when checkbox is set)
    # -------------------------
    def _load_preview(self):
        path = self.input_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Input required", "Select an existing input image first")
            return
        try:
            # If file is FITS and user wants FITS handling, load and optionally check normalization
            if path.lower().endswith((".fits", ".fit")):
                with fits.open(path, memmap=False) as hdul:
                    data = hdul[0].data
                if data is None:
                    raise ValueError("FITS contains no data")
                # If checkbox is set, verify normalization
                if self.check_fits_normalized.isChecked():
                    if not self._check_normalized_0_1(data):
                        # Launch normalizer and abort load so user can normalize first
                        launched = self._warn_and_launch_normalize(path, which="FITS")
                        return
                # store original min/max for remapping later
                self.orig_min = float(np.nanmin(data))
                self.orig_max = float(np.nanmax(data))
                self.loaded_from_fits = True
                preview = scale_float_to_uint8(data)
                self.loaded_image = preview
                pix = ndarray_to_qpix(preview if preview.ndim == 3 else preview)
                if pix:
                    self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                self._log(f"Loaded FITS {os.path.basename(path)} shape={getattr(data,'shape',None)} dtype={getattr(data,'dtype',None)} min={self.orig_min} max={self.orig_max}")
                return

            # Otherwise use OpenCV to read common image formats (including TIFF)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(path)
            self.loaded_from_fits = False
            self.orig_min = None
            self.orig_max = None
            self.loaded_image = img
            pix = ndarray_to_qpix(img if img.ndim == 3 else img)
            if pix:
                self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._log(f"Loaded {os.path.basename(path)} shape={getattr(img,'shape',None)} dtype={getattr(img,'dtype',None)}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"{e}\n\n{tb}")

    # -------------------------
    # Edge computations
    # -------------------------
    def _compute_sobel(self, gray):
        # use ksize=5 as original; produce combined magnitude of sobel x & y
        sx = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        sy = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
        mag = np.hypot(sx, sy)
        # scale to 0..255 uint8 for saving/display
        out = scale_float_to_uint8(mag)
        return out

    def _compute_canny(self, gray, low, high):
        can = cv2.Canny(gray, int(low), int(high))
        return can

    # -------------------------
    # Run processing
    # -------------------------
    def _on_run(self):
        if self.loaded_image is None:
            self._load_preview()
            if self.loaded_image is None:
                return

        outbase = self.output_edit.text().strip()
        if not outbase:
            QMessageBox.warning(self, "Output required", "Specify an output basename or path")
            return

        method = self.method_combo.currentText()
        low = int(self.lower_spin.value())
        high = int(self.upper_spin.value())

        # prepare grayscale blurred image
        try:
            img = self.loaded_image
            # If loaded from FITS we have a uint8 preview in self.loaded_image
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            # ensure uint8
            if gray.dtype != np.uint8:
                gray = scale_float_to_uint8(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        except Exception as e:
            QMessageBox.critical(self, "Processing error", f"Failed to prepare grayscale: {e}")
            return

        try:
            out_files = []
            # Reset last_output_image
            self.last_output_image = None

            if method in ("Sobel", "Both"):
                sobel_img = self._compute_sobel(gray)
                sobel_path = outbase + "_sbl.png" if not os.path.isdir(outbase) else os.path.join(outbase, "edges_sobel.png")
                if os.path.splitext(outbase)[1]:
                    sobel_path = os.path.splitext(outbase)[0] + "_sbl.png"
                cv2.imwrite(sobel_path, sobel_img)
                self._log(f"Wrote Sobel image: {sobel_path}")
                out_files.append(sobel_path)
                # store as last output
                self.last_output_image = sobel_img

            if method in ("Canny", "Both"):
                can_img = self._compute_canny(gray, low, high)
                can_path = outbase + "_cny.png" if not os.path.isdir(outbase) else os.path.join(outbase, "edges_canny.png")
                if os.path.splitext(outbase)[1]:
                    can_path = os.path.splitext(outbase)[0] + "_cny.png"
                cv2.imwrite(can_path, can_img)
                self._log(f"Wrote Canny image: {can_path}")
                out_files.append(can_path)
                # if Both, prefer to keep the last produced as last_output_image; if only Canny, set it
                self.last_output_image = can_img if self.last_output_image is None else self.last_output_image

            # show the first generated image in preview
            if out_files:
                first = cv2.imread(out_files[0], cv2.IMREAD_UNCHANGED)
                pix = ndarray_to_qpix(first if first.ndim == 3 else first)
                if pix:
                    self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            QMessageBox.information(self, "Done", "Edge detection complete")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

    # -------------------------
    # Save helpers
    # -------------------------
    def _save_output_as_fits(self):
        """
        Find the most likely output PNG produced by this tool (sbl or cny or a png matching outbase)
        and write a .fits file with the same pixel values (no rescaling).
        """
        outbase = self.output_edit.text().strip()
        if not outbase:
            QMessageBox.warning(self, "Output required", "Specify an output basename or path first")
            return

        # Build candidate filenames to check (prefer Sobel then Canny then generic)
        candidates = []

        # If user provided a directory, check standard filenames inside it
        if os.path.isdir(outbase):
            candidates.extend([
                os.path.join(outbase, "edges_sobel.png"),
                os.path.join(outbase, "edges_canny.png"),
                os.path.join(outbase, "edges_sobel.jpg"),
                os.path.join(outbase, "edges_canny.jpg"),
            ])
        else:
            # If user provided a basename without extension, try appended tags
            base_root, base_ext = os.path.splitext(outbase)
            if base_ext == "":
                candidates.extend([
                    outbase + "_sbl.png",
                    outbase + "_cny.png",
                    outbase + ".png",
                ])
            else:
                # user provided a filename with extension: try that and also replace extension with png
                candidates.append(outbase)
                candidates.append(base_root + ".png")

            # also try common replacements if user typed a path with extension
            candidates.extend([
                base_root + "_sbl.png",
                base_root + "_cny.png",
            ])

        # Remove duplicates while preserving order
        seen = set()
        uniq_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq_candidates.append(c)

        found = False
        for fn in uniq_candidates:
            if os.path.exists(fn):
                try:
                    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    # Prepare fits path: replace extension with .fits or append .fits
                    p = Path(fn)
                    fits_path = str(p.with_suffix(".fits"))
                    # Write FITS with the same dtype and values (no rescaling)
                    fits.writeto(fits_path, img, overwrite=True)
                    self._log(f"Wrote FITS from {fn}: {fits_path}")
                    QMessageBox.information(self, "Saved", f"Wrote FITS: {fits_path}")
                    found = True
                    # Optionally offer to open in Siril
                    if self.check_open_in_siril.isChecked():
                        self._try_launch_siril(fits_path)
                    break
                except Exception as e:
                    tb = traceback.format_exc()
                    self._log("Error writing FITS:", e)
                    QMessageBox.critical(self, "FITS write error", f"{e}\n\n{tb}")
                    return

        if not found:
            QMessageBox.warning(self, "File not found", "Could not find an output PNG to convert to FITS. Make sure you have run the tool and that the output basename is correct.")

    def _save_processed_as_normalized_fits(self):
        """
        Save the last processed image (self.last_output_image) as a normalized 32-bit FITS file.
        Behavior:
        - If "Normalize output FITS to [0,1]" is checked: output is normalized to [0,1].
        - Else if the input was a FITS and orig_min/orig_max are known: remap 0..255 -> orig_min..orig_max.
        - Otherwise: save normalized to 0..1.
        After saving, optionally offer to open in Siril.
        """
        if getattr(self, "last_output_image", None) is None:
            QMessageBox.warning(self, "No processed image", "No processed image available to save. Run the processing first.")
            return

        outbase = self.output_edit.text().strip()
        if not outbase:
            QMessageBox.warning(self, "Output required", "Specify an output basename or path first")
            return

        # Determine output path
        if os.path.isdir(outbase):
            out_path = os.path.join(outbase, "processed_normalized.fits")
        else:
            root, ext = os.path.splitext(outbase)
            if ext == "":
                out_path = root + "_processed.fits"
            else:
                out_path = root + ".fits"

        try:
            img8 = self.last_output_image
            # Ensure single-channel
            if img8.ndim == 3 and img8.shape[2] >= 3:
                img8 = img8[:, :, 0]
            img8 = np.asarray(img8, dtype=np.uint8)

            if self.check_normalize_output.isChecked():
                # Normalize to [0,1] float32
                img_f = img8.astype(np.float32) / 255.0
                hdr = fits.Header()
                hdr['NORMED'] = (1, "Output normalized to [0,1]")
            else:
                # If loaded from FITS and we have orig_min/orig_max, remap 0..255 -> orig_min..orig_max
                if getattr(self, "loaded_from_fits", False) and self.orig_min is not None and self.orig_max is not None:
                    mn = float(self.orig_min)
                    mx = float(self.orig_max)
                    if mx > mn:
                        img_f = (img8.astype(np.float32) / 255.0) * (mx - mn) + mn
                    else:
                        img_f = img8.astype(np.float32) * 0.0 + mn
                    hdr = fits.Header()
                    hdr['NORMED'] = (0, "Output remapped to original FITS range")
                else:
                    # fallback: normalize to [0,1]
                    img_f = img8.astype(np.float32) / 255.0
                    hdr = fits.Header()
                    hdr['NORMED'] = (1, "Output normalized to [0,1] (no original FITS range available)")

            fits.writeto(out_path, img_f.astype(np.float32), hdr, overwrite=True)
            self._log(f"Wrote normalized FITS: {out_path}")
            QMessageBox.information(self, "Saved", f"Wrote FITS: {out_path}")

            # Optionally offer to open in Siril
            if self.check_open_in_siril.isChecked():
                self._try_launch_siril(out_path)

        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error writing normalized FITS:", e)
            QMessageBox.critical(self, "FITS write error", f"{e}\n\n{tb}")

    def _try_launch_siril(self, filepath):
        """Attempt to launch Siril with the given filepath; log result."""
        try:
            siril_exe = shutil.which("siril")
            if siril_exe:
                subprocess.Popen([siril_exe, os.path.abspath(filepath)])
                self._log(f"Launched Siril with {filepath}")
            else:
                self._log("Siril executable not found in PATH.")
                QMessageBox.information(self, "Siril not found", "Siril executable not found in PATH; cannot launch.")
        except Exception as e:
            self._log(f"Failed to launch Siril: {e}")
            QMessageBox.information(self, "Siril launch failed", f"Failed to launch Siril: {e}")


def main():
    app = QApplication(sys.argv)
    w = EdgeDetectWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()