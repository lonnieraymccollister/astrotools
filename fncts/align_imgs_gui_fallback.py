#!/usr/bin/env python3
"""
align_imgs_gui_fallback.py
PyQt6 GUI to batch-align FITS images in a folder using astropy WCS + reproject.
If reproject.mosaicking.wcs_helpers.find_optimal_celestial_wcs is available it will be used;
otherwise the script falls back to using the first file's WCS and shape as the common grid.
"""
import sys
import os
import glob
import traceback
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# Try to import find_optimal_celestial_wcs from reproject if available
try:
    from reproject.mosaicking.wcs_helpers import find_optimal_celestial_wcs  # type: ignore
    HAS_FIND_OPTIMAL = True
except Exception:
    find_optimal_celestial_wcs = None  # type: ignore
    HAS_FIND_OPTIMAL = False

from reproject import reproject_interp  # reproject is still required for reprojection

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QTextEdit, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# ---------- Worker thread for alignment ----------
class AlignWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, inputs, outputs, ext, overwrite, use_optimal_wcs):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.ext = ext
        self.overwrite = overwrite
        self.use_optimal_wcs = use_optimal_wcs

    def safe_open(self, path, ext):
        with fits.open(path, memmap=False) as hdul:
            if ext is None or ext == "":
                hdu = hdul[0]
            else:
                try:
                    idx = int(ext)
                    hdu = hdul[idx]
                except Exception:
                    hdu = hdul[ext]
            data = hdu.data
            hdr = hdu.header
        return data, hdr

    def run(self):
        try:
            self.log.emit(f"Loading {len(self.inputs)} files...")
            dw = []
            for fn in self.inputs:
                data, hdr = self.safe_open(fn, self.ext)
                if data is None:
                    raise RuntimeError(f"No data found in {fn}")
                dw.append((np.array(data, dtype=np.float64), hdr))
                self.log.emit(f"Loaded: {os.path.basename(fn)} shape={dw[-1][0].shape}")

            # Decide on common WCS + shape
            if self.use_optimal_wcs and HAS_FIND_OPTIMAL:
                self.log.emit("Computing optimal common WCS using reproject.mosaicking.wcs_helpers...")
                try:
                    wcs_out, shape_out = find_optimal_celestial_wcs(dw)
                    self.log.emit("Optimal WCS computed.")
                except Exception as e:
                    self.log.emit(f"find_optimal_celestial_wcs failed: {e}; falling back to first-file WCS")
                    wcs_out, shape_out = self._fallback_first_wcs(dw)
            elif self.use_optimal_wcs and not HAS_FIND_OPTIMAL:
                self.log.emit("find_optimal_celestial_wcs not available; falling back to first-file WCS")
                wcs_out, shape_out = self._fallback_first_wcs(dw)
            else:
                self.log.emit("Using first file's WCS and shape as reference (fallback mode).")
                wcs_out, shape_out = self._fallback_first_wcs(dw)

            self.log.emit(f"Output shape: {shape_out}, ready to reproject.")

            # Reproject each and save
            for (data, hdr), outfn in zip(dw, self.outputs):
                self.log.emit(f"Reprojecting {os.path.basename(outfn)} ...")
                arr_in = data
                # choose 2D plane for reproject if cube
                if arr_in.ndim == 3:
                    if arr_in.shape[0] == 3 or arr_in.shape[0] <= 10:
                        src = arr_in[0]
                    else:
                        src = arr_in[..., 0]
                else:
                    src = arr_in

                arr_reproj, footprint = reproject_interp((src, hdr), wcs_out, shape_out=shape_out)
                out_header = wcs_out.to_header()
                # preserve common provenance keys
                for k in ("TELESCOP", "INSTRUME", "OBJECT"):
                    if k in hdr:
                        out_header[k] = hdr[k]
                if os.path.exists(outfn) and not self.overwrite:
                    self.log.emit(f"Skipping existing (overwrite disabled): {outfn}")
                else:
                    fits.writeto(outfn, arr_reproj.astype(np.float32), header=out_header, overwrite=True)
                    self.log.emit(f"Wrote: {outfn}")
            self.finished.emit(True, "All done")
        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit(f"Error: {e}\n{tb}")
            self.finished.emit(False, str(e))

    def _fallback_first_wcs(self, dw):
        # Use first file's WCS and shape
        first_data, first_hdr = dw[0]
        wcs_out = WCS(first_hdr)
        # derive shape: choose 2D shape from first_data
        if first_data.ndim == 2:
            shape_out = first_data.shape
        elif first_data.ndim == 3:
            # pick (Y, X) shape intelligently
            if first_data.shape[0] in (3, 4) or first_data.shape[0] <= 10:
                # channel-first (C, Y, X)
                shape_out = (first_data.shape[1], first_data.shape[2])
            else:
                # channel-last (Y, X, C)
                shape_out = (first_data.shape[0], first_data.shape[1])
        else:
            raise RuntimeError("Unsupported data ndim in reference file")
        return wcs_out, shape_out

# ---------- GUI ----------
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QTextEdit, QCheckBox
)

class AlignImagesForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Batch Align Images (glob mode)")
        self.resize(700, 420)
        self._build_ui()
        self.worker = None

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Input directory
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Input folder:"))
        self.in_dir_le = QLineEdit()
        h1.addWidget(self.in_dir_le, stretch=1)
        btn_in_dir = QPushButton("Browse Input Dir…")
        btn_in_dir.clicked.connect(self._browse_input_dir)
        h1.addWidget(btn_in_dir)
        layout.addLayout(h1)

        # Output directory
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Output folder:"))
        self.out_dir_le = QLineEdit()
        h2.addWidget(self.out_dir_le, stretch=1)
        btn_out_dir = QPushButton("Browse Output Dir…")
        btn_out_dir.clicked.connect(self._browse_output_dir)
        h2.addWidget(btn_out_dir)
        layout.addLayout(h2)

        # Extension and options
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("FITS extension (name or index, empty=primary):"))
        self.ext_le = QLineEdit()
        self.ext_le.setFixedWidth(120)
        h3.addWidget(self.ext_le)
        self.overwrite_chk = QCheckBox("Overwrite existing outputs")
        h3.addWidget(self.overwrite_chk)
        self.opt_wcs_chk = QCheckBox("Compute optimal common WCS (use if available)")
        self.opt_wcs_chk.setChecked(HAS_FIND_OPTIMAL)
        h3.addWidget(self.opt_wcs_chk)
        layout.addLayout(h3)

        # Buttons
        h4 = QHBoxLayout()
        self.preview_btn = QPushButton("Scan Input Folder")
        self.preview_btn.clicked.connect(self._scan_folder)
        h4.addWidget(self.preview_btn)
        self.align_button = QPushButton("Align All FITS")
        self.align_button.clicked.connect(self._on_align)
        h4.addWidget(self.align_button)
        layout.addLayout(h4)

        # File preview and log
        layout.addWidget(QLabel("Files to process:"))
        self.files_box = QTextEdit()
        self.files_box.setReadOnly(True)
        self.files_box.setFixedHeight(120)
        layout.addWidget(self.files_box)

        layout.addWidget(QLabel("Log:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, stretch=1)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_input_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select input folder with FITS")
        if d:
            self.in_dir_le.setText(d)

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.out_dir_le.setText(d)

    def _scan_folder(self):
        in_dir = self.in_dir_le.text().strip()
        if not in_dir or not os.path.isdir(in_dir):
            QMessageBox.warning(self, "Missing", "Select a valid input folder first")
            return
        pattern = os.path.join(in_dir, "*.fit*")
        files = sorted(glob.glob(pattern))
        if not files:
            QMessageBox.information(self, "No files", f"No .fit* files found in {in_dir}")
            return
        self.files_box.clear()
        for f in files:
            self.files_box.append(os.path.basename(f))
        self._log(f"Found {len(files)} files in {in_dir}")

    def _on_align(self):
        in_dir  = self.in_dir_le.text().strip()
        out_dir = self.out_dir_le.text().strip()
        ext = self.ext_le.text().strip() or None
        overwrite = bool(self.overwrite_chk.isChecked())
        use_optimal = bool(self.opt_wcs_chk.isChecked())

        if not in_dir or not out_dir:
            QMessageBox.warning(self, "Missing", "Please select both folders.")
            return
        if not os.path.isdir(in_dir):
            QMessageBox.critical(self, "Error", "Input folder does not exist")
            return
        os.makedirs(out_dir, exist_ok=True)

        pattern = os.path.join(in_dir, "*.fit*")
        inputs = sorted(glob.glob(pattern))
        if not inputs:
            QMessageBox.critical(self, "No FITS", f"No .fit* files found in {in_dir}")
            return

        outputs = [os.path.join(out_dir, "aligned_" + os.path.basename(fn)) for fn in inputs]
        self.files_box.clear()
        for a,b in zip(inputs, outputs):
            self.files_box.append(f"{os.path.basename(a)}  ->  {os.path.basename(b)}")

        # start worker thread
        self.worker = AlignWorker(inputs, outputs, ext, overwrite, use_optimal)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.align_button.setEnabled(False)
        self.worker.start()
        self._log("Alignment worker started...")

    def _on_finished(self, success, message):
        self.align_button.setEnabled(True)
        if success:
            QMessageBox.information(self, "Done", message)
        else:
            QMessageBox.critical(self, "Failed", message)

def main():
    app = QApplication(sys.argv)
    w = AlignImagesForm()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()