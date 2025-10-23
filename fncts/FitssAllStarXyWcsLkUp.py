#!/usr/bin/env python3
"""
PyQt6 GUI for filling RA/Dec in photometry CSVs using FITS WCS.
Save as fill_wcs_gui.py and run:

  python fill_wcs_gui.py

Requirements: PyQt6, astropy, numpy
"""
import sys
import csv
import os
from pathlib import Path
import traceback

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QTextEdit, QFileDialog, QCheckBox, QProgressBar, QMessageBox
)

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# ----------------------------------------------------------------------
# Worker thread to keep UI responsive
# ----------------------------------------------------------------------
class FillWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    log = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(str, int)  # outpath, updated_count

    def __init__(self, csv_path, out_path, single_fits_path, prefer_single_fits, debug, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.out_path = out_path
        self.single_fits_path = single_fits_path
        self.prefer_single_fits = prefer_single_fits
        self.debug = debug
        self._stop = False

    def stop(self):
        self._stop = True

    def emit_log(self, s):
        if self.debug:
            self.log.emit(s)

    def compute_radec_from_fits(self, fits_path, x_pixel, y_pixel):
        """Scan HDUs, try WCS constructions, try origin 0 and 1. Return (ra,dec,info) or (None,None,reason)."""
        try:
            with fits.open(fits_path, memmap=False) as hdul:
                for idx, hdu in enumerate(hdul):
                    hdr = getattr(hdu, "header", None)
                    if hdr is None:
                        continue
                    wcs_candidates = []
                    try:
                        wcs_candidates.append(WCS(hdr, naxis=2))
                    except Exception:
                        pass
                    try:
                        w_full = WCS(hdr)
                        wcs_candidates.append(w_full)
                        if getattr(w_full, "celestial", None) is not None:
                            wcs_candidates.append(w_full.celestial)
                    except Exception:
                        pass
                    try:
                        wcs_candidates.append(WCS(hdr, relax=True))
                    except Exception:
                        pass

                    for w in wcs_candidates:
                        if w is None:
                            continue
                        try:
                            if not getattr(w, "has_celestial", False):
                                continue
                        except Exception:
                            continue
                        for origin in (0, 1):
                            try:
                                ra_arr, dec_arr = w.all_pix2world([x_pixel], [y_pixel], origin)
                                ra = float(ra_arr[0]); dec = float(dec_arr[0])
                                if np.isfinite(ra) and np.isfinite(dec):
                                    info = f"HDU={idx} origin={origin}"
                                    return ra, dec, info
                            except Exception:
                                continue
                return None, None, "no usable WCS found"
        except Exception as e:
            return None, None, f"failed to open FITS: {e}"

    def run(self):
        try:
            if not self.csv_path or not os.path.exists(self.csv_path):
                self.log.emit("Input CSV not found.")
                self.finished_signal.emit("", 0)
                return

            rows = []
            with open(self.csv_path, newline='') as fp:
                reader = csv.DictReader(fp)
                fieldnames = list(reader.fieldnames or [])
                for r in reader:
                    rows.append(r)

            # Ensure output path
            if not self.out_path:
                p = Path(self.csv_path)
                self.out_path = str(p.with_name(p.stem + "_with_radec.csv"))

            # Ensure ra_deg/dec_deg fields present
            if 'ra_deg' not in fieldnames:
                fieldnames.append('ra_deg')
            if 'dec_deg' not in fieldnames:
                fieldnames.append('dec_deg')

            total = len(rows)
            updated = 0

            csv_dir = Path(self.csv_path).parent

            # for progress updates
            def set_progress(i):
                pct = int((i / max(total, 1)) * 100)
                self.progress.emit(pct)

            # iterate rows
            for i, r in enumerate(rows):
                if self._stop:
                    self.emit_log("Operation cancelled by user.")
                    break

                self.emit_log(f"ROW {i}: start")

                # ensure fields exist
                r.setdefault('ra_deg', '')
                r.setdefault('dec_deg', '')

                # skip if already present non-empty
                if r.get('ra_deg') not in (None, "", "nan") and r.get('dec_deg') not in (None, "", "nan"):
                    self.emit_log("  skipping: already has RA/Dec")
                    set_progress(i + 1)
                    continue

                # parse pixel coords
                try:
                    x = float(r.get('x_pixel', ''))
                    y = float(r.get('y_pixel', ''))
                except Exception:
                    self.emit_log(f"  invalid pixel coords x_pixel='{r.get('x_pixel')}' y_pixel='{r.get('y_pixel')}' - skipping")
                    set_progress(i + 1)
                    continue

                # determine fits_path
                fits_path = None
                used_col = None
                if self.prefer_single_fits and self.single_fits_path:
                    fits_path = self.single_fits_path
                    used_col = "<single fits>"
                    self.emit_log(f"  using single FITS: {fits_path}")
                else:
                    candidate_file_cols = ['original_file', 'path', 'filename', 'file', 'image', 'input_file', 'orig', 'fits']
                    for c in candidate_file_cols:
                        if c in r and r.get(c) not in (None, '', 'nan'):
                            fits_path = str(r.get(c)).strip()
                            used_col = c
                            break
                    if not fits_path:
                        # last-resort: any string that endswith .fits/.fit
                        for k, v in r.items():
                            if isinstance(v, str) and v.lower().endswith(('.fits', '.fit')):
                                fits_path = v.strip(); used_col = k; break

                if not fits_path:
                    self.emit_log("  missing FITS path for row - skipping")
                    set_progress(i + 1)
                    continue

                # resolve relative paths
                if not os.path.exists(fits_path):
                    candidate = csv_dir / fits_path
                    if candidate.exists():
                        fits_path = str(candidate)
                        self.emit_log(f"  resolved relative path -> {fits_path}")
                    else:
                        self.emit_log(f"  FITS not found at '{fits_path}' - skipping")
                        set_progress(i + 1)
                        continue

                # try compute RA/Dec
                ra, dec, info = self.compute_radec_from_fits(fits_path, x, y)
                if ra is None:
                    self.emit_log(f"  compute_radec_from_fits failed: {info}")
                    set_progress(i + 1)
                    continue

                r['ra_deg'] = f"{ra:.8f}"
                r['dec_deg'] = f"{dec:.8f}"
                updated += 1
                self.emit_log(f"  wrote RA={r['ra_deg']} DEC={r['dec_deg']} ({info})")
                set_progress(i + 1)

            # write output CSV
            with open(self.out_path, "w", newline='') as outfp:
                writer = csv.DictWriter(outfp, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)

            self.emit_log(f"Wrote {self.out_path} (filled RA/Dec for {updated} rows)")
            self.finished_signal.emit(self.out_path, updated)
        except Exception as e:
            self.log.emit("FATAL: " + str(e))
            self.log.emit(traceback.format_exc())
            self.finished_signal.emit("", 0)

# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RA/Dec Filler (WCS) - PyQt6")
        self.resize(800, 560)
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # file selectors
        h1 = QHBoxLayout()
        self.csv_edit = QLineEdit()
        btn_csv = QPushButton("Open CSV")
        btn_csv.clicked.connect(self.open_csv)
        h1.addWidget(QLabel("Input CSV:"))
        h1.addWidget(self.csv_edit)
        h1.addWidget(btn_csv)
        layout.addLayout(h1)

        h2 = QHBoxLayout()
        self.fits_edit = QLineEdit()
        btn_fits = QPushButton("Single FITS (optional)")
        btn_fits.clicked.connect(self.open_fits)
        h2.addWidget(QLabel("Single FITS (optional):"))
        h2.addWidget(self.fits_edit)
        h2.addWidget(btn_fits)
        layout.addLayout(h2)

        # options
        h3 = QHBoxLayout()
        self.checkbox_single = QCheckBox("Use single FITS for all rows")
        self.checkbox_debug = QCheckBox("Enable debug logging")
        self.out_edit = QLineEdit()
        h3.addWidget(self.checkbox_single)
        h3.addWidget(self.checkbox_debug)
        h3.addStretch()
        layout.addLayout(h3)

        # output path
        h4 = QHBoxLayout()
        btn_out = QPushButton("Select Output CSV")
        btn_out.clicked.connect(self.select_out)
        h4.addWidget(QLabel("Output CSV:"))
        h4.addWidget(self.out_edit)
        h4.addWidget(btn_out)
        layout.addLayout(h4)

        # controls
        h5 = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self.run_fill)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_cancel.setEnabled(False)
        h5.addWidget(self.btn_run)
        h5.addWidget(self.btn_cancel)
        h5.addStretch()
        layout.addLayout(h5)

        # progress and log
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

    def open_csv(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv);;All files (*.*)")
        if fn:
            self.csv_edit.setText(fn)
            # propose default output path
            p = Path(fn)
            self.out_edit.setText(str(p.with_name(p.stem + "_with_radec.csv")))

    def open_fits(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS", "", "FITS files (*.fits *.fit);;All files (*.*)")
        if fn:
            self.fits_edit.setText(fn)

    def select_out(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Output CSV", self.out_edit.text() or "", "CSV files (*.csv);;All files (*.*)")
        if fn:
            self.out_edit.setText(fn)

    def append_log(self, s):
        self.log_view.append(s)
        # also print minimal to stdout for convenience
        print(s)

    def run_fill(self):
        csv_path = self.csv_edit.text().strip()
        if not csv_path or not os.path.exists(csv_path):
            QMessageBox.warning(self, "Missing CSV", "Please select an existing input CSV.")
            return

        out_path = self.out_edit.text().strip() or None
        single_fits = self.fits_edit.text().strip() or None
        prefer_single = bool(self.checkbox_single.isChecked())
        debug = bool(self.checkbox_debug.isChecked())

        self.log_view.clear()
        self.append_log("Starting RA/Dec fill...")
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        # create and start worker
        self._worker = FillWorker(csv_path, out_path, single_fits, prefer_single, debug)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.log.connect(self.append_log)
        self._worker.finished_signal.connect(self.on_finished)
        self._worker.start()

    def cancel(self):
        if self._worker:
            self._worker.stop()
            self.append_log("Cancel requested...")

    def on_finished(self, outpath, updated):
        if outpath:
            self.append_log(f"Finished. Wrote {outpath} (filled {updated} rows)")
            QMessageBox.information(self, "Done", f"Wrote {outpath}\nFilled {updated} rows")
        else:
            self.append_log("Finished with no output (see log).")
            QMessageBox.warning(self, "Finished", "No output was produced; check the log.")
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._worker = None
        self.progress.setValue(100)

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()