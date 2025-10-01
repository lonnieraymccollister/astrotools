#!/usr/bin/env python3
"""
align_imgs_gui.py
PyQt6 GUI to batch-align FITS images in a folder using astropy WCS + reproject.
Select input folder, output folder, extension, and run. Outputs aligned_*.fits files.
"""
import sys
import os
import glob
import traceback
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import find_optimal_celestial_wcs
from reproject import reproject_interp

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QTextEdit, QSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# ---------- Worker thread for alignment ----------
class AlignWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, inputs, outputs, ext, overwrite, resample_to_common):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.ext = ext
        self.overwrite = overwrite
        self.resample_to_common = resample_to_common

    def safe_open(self, path, ext):
        with fits.open(path, memmap=False) as hdul:
            if ext is None:
                hdu = hdul[0]
            else:
                # allow numeric or string ext name
                try:
                    ext_i = int(ext)
                    hdu = hdul[ext_i]
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
                    raise RuntimeError(f"No data in {fn}")
                dw.append((np.array(data, dtype=np.float64), hdr))
                self.log.emit(f"Loaded: {os.path.basename(fn)} shape={dw[-1][0].shape}")

            # determine optimal WCS and output shape
            if self.resample_to_common:
                self.log.emit("Computing optimal common WCS...")
                wcs_out, shape_out = find_optimal_celestial_wcs(dw)
            else:
                # use first file's header as reference
                self.log.emit("Using reference WCS from first file")
                wcs_out = WCS(dw[0][1])
                # choose shape from first image 2D shape (handle cube -> take first plane)
                data0 = dw[0][0]
                if data0.ndim == 3:
                    h, w = data0.shape[1], data0.shape[2]
                elif data0.ndim == 2:
                    h, w = data0.shape
                else:
                    raise RuntimeError("Unsupported data ndim in reference file")
                shape_out = (h, w)

            self.log.emit(f"Output shape: {shape_out}, WCS ready")

            # reproject and save each
            for (data, hdr), outfn in zip(dw, self.outputs):
                self.log.emit(f"Reprojecting {os.path.basename(outfn)} ...")
                # select 2D data if cube: prefer first plane
                arr_in = data
                if arr_in.ndim == 3:
                    # If arr is (C,Y,X) or (Y,X,C) try to extract a 2D plane for reprojection;
                    # prefer (Y,X) plane if shape matches WCS NAXIS1/NAXIS2 in header
                    if arr_in.shape[0] == 3 or arr_in.shape[0] <= 10:
                        # assume channel-first (C,Y,X)
                        src = arr_in[0]
                    elif arr_in.shape[2] == 3 or arr_in.shape[2] <= 10:
                        # channel-last (Y,X,C)
                        src = arr_in[..., 0]
                    else:
                        src = arr_in[0]
                else:
                    src = arr_in

                arr_reproj, footprint = reproject_interp((src, hdr), wcs_out, shape_out=shape_out)
                # build header from wcs_out
                out_header = wcs_out.to_header()
                # preserve a few provenance keys from original header where present
                for k in ("TELESCOP", "INSTRUME", "OBJECT"):
                    if k in hdr:
                        out_header[k] = hdr[k]
                # save
                if os.path.exists(outfn) and not self.overwrite:
                    self.log.emit(f"Skipping existing file (overwrite disabled): {outfn}")
                else:
                    fits.writeto(outfn, arr_reproj.astype(np.float32), header=out_header, overwrite=True)
                    self.log.emit(f"Wrote: {outfn}")
            self.finished.emit(True, "All done")
        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit(f"Error: {e}\n{tb}")
            self.finished.emit(False, str(e))

# ---------- GUI ----------
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
        self.common_wcs_chk = QCheckBox("Compute optimal common WCS (recommended)")
        self.common_wcs_chk.setChecked(True)
        h3.addWidget(self.common_wcs_chk)
        layout.addLayout(h3)

        # List preview + run button
        h4 = QHBoxLayout()
        self.preview_btn = QPushButton("Scan Input Folder")
        self.preview_btn.clicked.connect(self._scan_folder)
        h4.addWidget(self.preview_btn)
        self.align_button = QPushButton("Align All FITS")
        self.align_button.clicked.connect(self._on_align)
        h4.addWidget(self.align_button)
        layout.addLayout(h4)

        # Log
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
        resample_to_common = bool(self.common_wcs_chk.isChecked())

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
        self.worker = AlignWorker(inputs, outputs, ext, overwrite, resample_to_common)
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