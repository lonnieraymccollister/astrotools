#!/usr/bin/env python3
"""
combine_weighted_gui.py
PyQt6 GUI to combine a directory of reprojected FITS images into one weighted composite.
Behavior: for each input pixel > mask_threshold count it as contribution; final pixel =
 (composite_sum / weight) * sqrt(max_weight) where max_weight is maximum contributions per-pixel.
"""
import sys
import os
import glob
import traceback
import numpy as np
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QMessageBox, QSpinBox, QCheckBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Worker thread for the combine operation
class CombineWorker(QThread):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, pattern, outpath, mask_thresh, overwrite):
        super().__init__()
        self.pattern = pattern
        self.outpath = outpath
        self.mask_thresh = float(mask_thresh)
        self.overwrite = overwrite

    def _emit(self, *args):
        self.log.emit(" ".join(str(a) for a in args))

    def run(self):
        try:
            files = sorted(glob.glob(self.pattern))
            if not files:
                raise FileNotFoundError(f"No files match pattern: {self.pattern}")
            self._emit(f"Found {len(files)} files. Opening first to determine shape...")

            # Open first to determine shape and dtype
            with fits.open(files[0], memmap=False) as hd:
                data0 = hd[0].data
                if data0 is None:
                    raise RuntimeError(f"No data in primary HDU of {files[0]}")
                shape = data0.shape
                self._emit(f"Canvas shape {shape}, dtype {data0.dtype}")

            # Prepare composite and weights arrays (float32)
            composite = np.zeros(shape, dtype=np.float64)
            weights = np.zeros(shape, dtype=np.float32)

            # Iterate files
            for idx, fn in enumerate(files, start=1):
                self._emit(f"[{idx}/{len(files)}] Loading: {os.path.basename(fn)}")
                try:
                    with fits.open(fn, memmap=False) as hd:
                        data = hd[0].data.astype(np.float32)
                except Exception as e:
                    self._emit(f"Failed to read {fn}: {e}; skipping")
                    continue

                if data.shape != shape:
                    self._emit(f"Skipping {fn}: shape mismatch {data.shape} != {shape}")
                    continue

                # mask where pixel contributes (value strictly greater than threshold)
                mask = (data > self.mask_thresh)
                if not mask.any():
                    self._emit(f"No contributions in {os.path.basename(fn)} (thresholded).")
                    continue

                # accumulate
                composite[mask] += data[mask].astype(np.float64)
                weights[mask] += 1.0

            max_weight = float(np.max(weights))
            if max_weight <= 0:
                raise RuntimeError("No valid contributions found across inputs (all weights zero)")

            self._emit(f"Maximum contributions across canvas: {max_weight:.0f}")

            # compute final image with scaling described
            final = np.zeros_like(composite, dtype=np.float32)
            valid = weights > 0
            final[valid] = (composite[valid] / weights[valid]) * np.sqrt(max_weight)

            # Save result
            if os.path.exists(self.outpath) and not self.overwrite:
                raise FileExistsError(f"Output file exists and overwrite is disabled: {self.outpath}")

            # Try to preserve some header info from first file (if present)
            header = None
            try:
                with fits.open(files[0], memmap=False) as hd:
                    header = hd[0].header.copy()
            except Exception:
                header = None

            hdu = fits.PrimaryHDU(final, header=header)
            hdu.writeto(self.outpath, overwrite=self.overwrite)
            self._emit(f"Wrote final composite to: {self.outpath}")
            self.finished.emit(True, f"Wrote {self.outpath}")
        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit(f"Error: {e}\n{tb}")
            self.finished.emit(False, str(e))


# Main GUI
class CombineGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weighted FITS Combiner")
        self.resize(760, 520)
        self._build_ui()
        self.worker = None

    def _build_ui(self):
        v = QVBoxLayout(self)

        # Input folder + pattern
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Input folder:"))
        self.input_folder_le = QLineEdit()
        h1.addWidget(self.input_folder_le, stretch=1)
        btn_in = QPushButton("Browse…")
        btn_in.clicked.connect(self._browse_folder)
        h1.addWidget(btn_in)
        v.addLayout(h1)

        hpat = QHBoxLayout()
        hpat.addWidget(QLabel("File pattern:"))
        self.pattern_le = QLineEdit("*.fit*")
        hpat.addWidget(self.pattern_le)
        hpat.addWidget(QLabel("Mask threshold >"))
        self.mask_spin = QDoubleSpinBox()
        self.mask_spin.setRange(-1e12, 1e12)
        self.mask_spin.setDecimals(6)
        self.mask_spin.setValue(0.0)
        hpat.addWidget(self.mask_spin)
        v.addLayout(hpat)

        # Output file
        hout = QHBoxLayout()
        hout.addWidget(QLabel("Output FITS:"))
        self.output_le = QLineEdit("final_wgted_Al_Img.fits")
        hout.addWidget(self.output_le, stretch=1)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_save)
        hout.addWidget(btn_out)
        v.addLayout(hout)

        # Overwrite checkbox and run button
        hopt = QHBoxLayout()
        self.overwrite_chk = QCheckBox("Overwrite existing output")
        hopt.addWidget(self.overwrite_chk)
        hopt.addStretch(1)
        self.run_btn = QPushButton("Run Combine")
        self.run_btn.clicked.connect(self._on_run)
        hopt.addWidget(self.run_btn)
        v.addLayout(hopt)

        # Files found viewer + controls
        hfiles = QHBoxLayout()
        self.scan_btn = QPushButton("Scan Pattern")
        self.scan_btn.clicked.connect(self._scan_pattern)
        hfiles.addWidget(self.scan_btn)
        self.files_box = QTextEdit()
        self.files_box.setReadOnly(True)
        v.addLayout(hfiles)
        v.addWidget(QLabel("Files found:"))
        v.addWidget(self.files_box, stretch=1)

        # Log
        v.addWidget(QLabel("Log:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(140)
        v.addWidget(self.log)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select input folder")
        if d:
            self.input_folder_le.setText(d)

    def _browse_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save output FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.output_le.setText(fn)

    def _scan_pattern(self):
        folder = self.input_folder_le.text().strip()
        pat = self.pattern_le.text().strip()
        if not folder:
            QMessageBox.warning(self, "Input required", "Select input folder first")
            return
        pattern = os.path.join(folder, pat)
        files = sorted(glob.glob(pattern))
        self.files_box.clear()
        if not files:
            self.files_box.append("No files found.")
            self._log(f"No files matched: {pattern}")
            return
        for f in files:
            self.files_box.append(os.path.basename(f))
        self._log(f"Found {len(files)} files matching {pattern}")

    def _on_run(self):
        folder = self.input_folder_le.text().strip()
        pat = self.pattern_le.text().strip()
        outp = self.output_le.text().strip()
        mask_thresh = float(self.mask_spin.value())
        overwrite = bool(self.overwrite_chk.isChecked())

        if not folder:
            QMessageBox.warning(self, "Input required", "Select input folder first")
            return
        if not pat:
            QMessageBox.warning(self, "Pattern required", "Enter a file glob pattern (for example *.fits)")
            return
        if not outp:
            QMessageBox.warning(self, "Output required", "Specify an output FITS filename")
            return

        pattern = os.path.join(folder, pat)
        files = sorted(glob.glob(pattern))
        if not files:
            QMessageBox.critical(self, "No files", f"No files match: {pattern}")
            return

        # disable UI while running
        self.run_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self._log("Starting combine operation...")
        self.files_box.clear()
        for f in files:
            self.files_box.append(os.path.basename(f))

        # start worker
        self.worker = CombineWorker(pattern, outp, mask_thresh, overwrite)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_finished(self, success, message):
        self.run_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "Done", message)
        else:
            QMessageBox.critical(self, "Failed", message)
        self._log("Worker finished: " + message)


def main():
    app = QApplication(sys.argv)
    w = CombineGui()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()