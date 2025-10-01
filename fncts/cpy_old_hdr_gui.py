#!/usr/bin/env python3
"""
cpy_old_hdr_gui.py
Copy an "old" FITS header into a new image FITS file, optionally patching WCS
for a binning factor (adjust CRPIX and CDELT). PyQt6 GUI.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QSpinBox, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt

def print_and_patch_wcs(header, bin_factor):
    """
    Return a header copied from `header` with WCS patched for integer bin_factor.
    Updates CRPIX and CDELT consistently and ensures CD/CDelt/CUNIT/RADESYS/EQUINOX keys.
    """
    w = WCS(header)

    # original values (arrays)
    orig_crpix = w.wcs.crpix.copy()
    orig_cdelt = w.wcs.cdelt.copy()

    # compute new CRPIX, CDELT for integer bin factor
    new_crpix = (orig_crpix - 0.5) / bin_factor + 0.5
    new_cdelt = orig_cdelt * bin_factor

    # update WCS object
    w.wcs.crpix = new_crpix
    w.wcs.cdelt = new_cdelt

    # build CD header (this will include CD1_1 etc if appropriate)
    hdr_cd = w.to_header(relax=True)

    # explicit CDELT/CUNIT/RADESYS/EQUINOX block
    hdr_cdelt = fits.Header()
    hdr_cdelt["CDELT1"] = (new_cdelt[0], "deg/pix")
    hdr_cdelt["CDELT2"] = (new_cdelt[1], "deg/pix")
    hdr_cdelt["CUNIT1"] = ("deg", "units of CRVAL and CDELT")
    hdr_cdelt["CUNIT2"] = ("deg", "units of CRVAL and CDELT")
    # attempt to preserve RADESYS and EQUINOX if present in WCS
    radesys = getattr(w.wcs, "radesys", None) or header.get("RADESYS")
    equinox = getattr(w.wcs, "equinox", None) or header.get("EQUINOX")
    if radesys is not None:
        hdr_cdelt["RADESYS"] = (radesys, "frame")
    if equinox is not None:
        hdr_cdelt["EQUINOX"] = (equinox, "equinox")

    # merge into a copy of the original header (preserve other keywords)
    out = header.copy()
    for k in hdr_cd:
        out[k] = hdr_cd[k]
    for k in hdr_cdelt:
        out[k] = hdr_cdelt[k]

    # Add human-readable provenance
    out["HISTORY"] = "Header copied from older file and patched for bin_factor={}".format(bin_factor)
    return out, orig_crpix, orig_cdelt, new_crpix, new_cdelt

class CpyOldHdrWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Copy Old Header (with optional WCS patch)")
        self._build_ui()
        self.resize(820, 360)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        grid.addWidget(QLabel("Old-header FITS:"), 0, 0)
        self.old_edit = QLineEdit()
        grid.addWidget(self.old_edit, 0, 1, 1, 3)
        btn_old = QPushButton("Browse")
        btn_old.clicked.connect(lambda: self._pick_file(self.old_edit))
        grid.addWidget(btn_old, 0, 4)

        grid.addWidget(QLabel("New-image FITS (data):"), 1, 0)
        self.new_edit = QLineEdit()
        grid.addWidget(self.new_edit, 1, 1, 1, 3)
        btn_new = QPushButton("Browse")
        btn_new.clicked.connect(lambda: self._pick_file(self.new_edit))
        grid.addWidget(btn_new, 1, 4)

        grid.addWidget(QLabel("Output FITS:"), 2, 0)
        self.out_edit = QLineEdit("out_with_old_header.fits")
        grid.addWidget(self.out_edit, 2, 1, 1, 3)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._pick_save)
        grid.addWidget(btn_out, 2, 4)

        grid.addWidget(QLabel("Binning factor (int, 0=no patch):"), 3, 0)
        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(0, 1000)
        self.bin_spin.setValue(0)
        grid.addWidget(self.bin_spin, 3, 1)

        self.run_btn = QPushButton("Run Copy/Write")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 3, 2)

        self.show_hdr_btn = QPushButton("Show WCS Preview")
        self.show_hdr_btn.clicked.connect(self._show_wcs_preview)
        grid.addWidget(self.show_hdr_btn, 3, 3)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 4, 0, 6, 5)

    def _pick_file(self, line_edit):
        fn, _ = QFileDialog.getOpenFileName(self, "Select FITS file", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            line_edit.setText(fn)

    def _pick_save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save FITS as", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.out_edit.setText(fn)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _show_wcs_preview(self):
        old_path = self.old_edit.text().strip()
        if not old_path:
            QMessageBox.information(self, "No file", "Select an Old-header FITS file first")
            return
        try:
            with fits.open(old_path) as h:
                hdr = h[0].header
            bin_val = int(self.bin_spin.value())
            if bin_val > 0:
                patched_hdr, orig_crpix, orig_cdelt, new_crpix, new_cdelt = print_and_patch_wcs(hdr, bin_val)[0:5]
            else:
                patched_hdr = hdr
                w = WCS(hdr)
                orig_crpix = w.wcs.crpix.copy()
                orig_cdelt = w.wcs.cdelt.copy()
                new_crpix = orig_crpix
                new_cdelt = orig_cdelt
            self._log("Old-header file:", Path(old_path).name)
            self._log(f"Original CRPIX: {orig_crpix}")
            self._log(f"Original CDELT: {orig_cdelt}")
            if bin_val > 0:
                self._log(f"Patched CRPIX (bin={bin_val}): {new_crpix}")
                self._log(f"Patched CDELT (bin={bin_val}): {new_cdelt}")
            else:
                self._log("No patch requested (bin=0).")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Preview error", f"{e}\n\n{tb}")

    def _on_run(self):
        old_path = self.old_edit.text().strip()
        new_path = self.new_edit.text().strip()
        out_path = self.out_edit.text().strip()
        bin_val = int(self.bin_spin.value())

        if not old_path:
            QMessageBox.warning(self, "Missing input", "Select Old-header FITS file")
            return
        if not new_path:
            QMessageBox.warning(self, "Missing input", "Select New-image FITS file (contains data)")
            return
        if not out_path:
            QMessageBox.warning(self, "Missing output", "Specify an output FITS filename")
            return

        try:
            # load old header
            with fits.open(old_path) as h_old:
                old_header = h_old[0].header.copy()
            # optionally patch
            if bin_val > 0:
                new_header, orig_crpix, orig_cdelt, new_crpix, new_cdelt = print_and_patch_wcs(old_header, bin_val)
                self._log(f"Patched header for bin={bin_val}")
                self._log(f"Original CRPIX: {orig_crpix}  -> Patched CRPIX: {new_crpix}")
                self._log(f"Original CDELT: {orig_cdelt}  -> Patched CDELT: {new_cdelt}")
            else:
                new_header = old_header
                self._log("Using old header unchanged (bin=0)")

            # load new image data
            with fits.open(new_path) as h_new:
                data = h_new[0].data
            if data is None:
                raise ValueError("No data found in new-image FITS primary HDU")

            # write output once
            fits.writeto(out_path, data, header=new_header, overwrite=True)
            self._log(f"Wrote output: {out_path}")
            QMessageBox.information(self, "Done", f"Wrote output FITS: {out_path}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = CpyOldHdrWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()