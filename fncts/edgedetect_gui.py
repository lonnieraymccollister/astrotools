#!/usr/bin/env python3
"""
edgedetect_gui.py
PyQt6 GUI to run Sobel and/or Canny edge detection on a color image and save results.
"""
import sys
import os
import traceback
import numpy as np
import cv2
import tifffile
from astropy.io import fits
from pathlib import Path


from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QMessageBox, QSpinBox
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
                    # If image is color (H,W,3) we write the array as-is; astropy will set BITPIX accordingly.
                    fits.writeto(fits_path, img, overwrite=True)
                    self._log(f"Wrote FITS from {fn}: {fits_path}")
                    QMessageBox.information(self, "Saved", f"Wrote FITS: {fits_path}")
                    found = True
                    break
                except Exception as e:
                    tb = traceback.format_exc()
                    self._log("Error writing FITS:", e)
                    QMessageBox.critical(self, "FITS write error", f"{e}\n\n{tb}")
                    return

        if not found:
            QMessageBox.warning(self, "File not found", "Could not find an output PNG to convert to FITS. Make sure you have run the tool and that the output basename is correct.")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge Detection (Sobel / Canny)")
        self._build_ui()
        self.resize(920, 520)
        self.loaded_image = None  # raw image as read by OpenCV

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

        # NEW: Save output as FITS button (writes an 8-bit FITS from the PNG output, no rescaling)
        self.savefits_btn = QPushButton("Save Output as FITS")
        self.savefits_btn.clicked.connect(self._save_output_as_fits)
        grid.addWidget(self.savefits_btn, 3, 3)

        # Preview label and log
        self.preview_label = QLabel("Preview")
        self.preview_label.setFixedSize(560, 360)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.preview_label, 4, 0, 6, 3)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 4, 3, 6, 3)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)")
        if fn:
            self.input_edit.setText(fn)

    def _browse_output_folder(self):
        dn = QFileDialog.getExistingDirectory(self, "Select output folder", "")
        if dn:
            base = os.path.basename(self.output_edit.text()) or "edges_output"
            self.output_edit.setText(os.path.join(dn, base))

    def _load_preview(self):
        path = self.input_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Input required", "Select an existing input image first")
            return
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(path)
            self.loaded_image = img
            pix = ndarray_to_qpix(img if img.ndim == 3 else img)
            if pix:
                self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._log(f"Loaded {os.path.basename(path)} shape={getattr(img,'shape',None)} dtype={getattr(img,'dtype',None)}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"{e}\n\n{tb}")

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
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        except Exception as e:
            QMessageBox.critical(self, "Processing error", f"Failed to prepare grayscale: {e}")
            return

        try:
            out_files = []
            if method in ("Sobel", "Both"):
                sobel_img = self._compute_sobel(gray)
                # save as basename_sobel.png (use uint8); preserve directory if provided in outbase
                sobel_path = outbase + "_sbl.png" if not os.path.isdir(outbase) else os.path.join(outbase, "edges_sobel.png")
                # if outbase contains an extension, replace it
                if os.path.splitext(outbase)[1]:
                    sobel_path = os.path.splitext(outbase)[0] + "_sbl.png"
                cv2.imwrite(sobel_path, sobel_img)
                self._log(f"Wrote Sobel image: {sobel_path}")
                out_files.append(sobel_path)

            if method in ("Canny", "Both"):
                can_img = self._compute_canny(gray, low, high)
                can_rgb = np.stack([can_img, can_img, can_img], axis=-1)
                can_path = outbase + "_cny.png" if not os.path.isdir(outbase) else os.path.join(outbase, "edges_canny.png")
                if os.path.splitext(outbase)[1]:
                    can_path = os.path.splitext(outbase)[0] + "_cny.png"
                cv2.imwrite(can_path, can_img)
                self._log(f"Wrote Canny image: {can_path}")
                out_files.append(can_path)

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

def main():
    app = QApplication(sys.argv)
    w = EdgeDetectWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()