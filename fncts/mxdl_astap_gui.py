#!/usr/bin/env python3
"""
mxdl_astap_gui.py
PyQt6 GUI to read MaxDL/ASTAP-like FITS headers and write a computed high-precision WCS (CD matrix) into the header.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import mpmath as mp
from astropy.io import fits
from astropy.coordinates import Angle

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QTextEdit, QMessageBox, QHBoxLayout, QVBoxLayout, QCheckBox
)
from PyQt6.QtCore import Qt

class MxdlAstapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MaxDL/ASTAP WCS Writer")
        self._build_ui()
        self.resize(780, 420)
        self.input_path = None
        self.header = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Row 0: FITS picker
        grid.addWidget(QLabel("Input FITS file:"), 0, 0)
        self.fits_edit = QLineEdit()
        grid.addWidget(self.fits_edit, 0, 1, 1, 3)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self._browse_fits)
        grid.addWidget(btn_in, 0, 4)

        # Row 1: optional manual fields
        grid.addWidget(QLabel("OBJCTRA (HH MM SS):"), 1, 0)
        self.objctra_edit = QLineEdit()
        grid.addWidget(self.objctra_edit, 1, 1)

        grid.addWidget(QLabel("OBJCTDEC (DD MM SS):"), 1, 2)
        self.objctdec_edit = QLineEdit()
        grid.addWidget(self.objctdec_edit, 1, 3)

        grid.addWidget(QLabel("Pixel scale (deg/pix) CDELT1:"), 2, 0)
        self.cdelt1_edit = QLineEdit()
        grid.addWidget(self.cdelt1_edit, 2, 1)

        grid.addWidget(QLabel("Pixel scale (deg/pix) CDELT2:"), 2, 2)
        self.cdelt2_edit = QLineEdit()
        grid.addWidget(self.cdelt2_edit, 2, 3)

        # Row 3: solved parameters (user-provided or example)
        grid.addWidget(QLabel("Solved CD1_1 (example):"), 3, 0)
        self.solved_cd11_edit = QLineEdit("-3.542246e-4")
        grid.addWidget(self.solved_cd11_edit, 3, 1)

        grid.addWidget(QLabel("Rotation (deg) [e.g. 179.7]:"), 3, 2)
        self.rotation_edit = QLineEdit("179.7")
        grid.addWidget(self.rotation_edit, 3, 3)

        # Row 4: options and buttons
        self.overwrite_chk = QCheckBox("Overwrite original FITS (if unchecked will write _wcs.fit)")
        self.overwrite_chk.setChecked(False)
        grid.addWidget(self.overwrite_chk, 4, 0, 1, 3)

        btn_compute = QPushButton("Compute WCS (preview)")
        btn_compute.clicked.connect(self._compute_wcs_preview)
        grid.addWidget(btn_compute, 4, 3)

        # Row 5: save
        grid.addWidget(QLabel("Output filename (optional):"), 5, 0)
        self.output_edit = QLineEdit()
        grid.addWidget(self.output_edit, 5, 1, 1, 3)
        btn_save = QPushButton("Write WCS to FITS")
        btn_save.clicked.connect(self._write_wcs_to_fits)
        grid.addWidget(btn_save, 5, 4)

        # Lower area: computed values and log
        vbox = QVBoxLayout()
        grid.addLayout(vbox, 6, 0, 1, 5)

        self.values_box = QTextEdit()
        self.values_box.setReadOnly(True)
        self.values_box.setFixedHeight(160)
        vbox.addWidget(QLabel("Computed WCS values (preview)"))
        vbox.addWidget(self.values_box)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(120)
        vbox.addWidget(QLabel("Log"))
        vbox.addWidget(self.log)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_fits(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.fits_edit.setText(fn)
            self.input_path = fn
            try:
                with fits.open(fn) as hd:
                    hdr = hd[0].header
                    self.header = hdr.copy()
                # Auto-fill fields from header if present
                if hdr.get("OBJCTRA"):
                    self.objctra_edit.setText(str(hdr.get("OBJCTRA")))
                if hdr.get("OBJCTDEC"):
                    self.objctdec_edit.setText(str(hdr.get("OBJCTDEC")))
                if hdr.get("CDELT1"):
                    self.cdelt1_edit.setText(str(hdr.get("CDELT1")))
                if hdr.get("CDELT2"):
                    self.cdelt2_edit.setText(str(hdr.get("CDELT2")))
                self._log(f"Loaded header from {Path(fn).name}")
            except Exception as e:
                self._log("Failed to read header:", e)

    def _compute_wcs_preview(self):
        try:
            # set mpmath precision
            mp.mp.dps = 50

            # gather inputs: prefer user fields, else header, else error
            hdr = self.header if self.header is not None else {}
            objctra = self.objctra_edit.text().strip() or hdr.get("OBJCTRA")
            objctdec = self.objctdec_edit.text().strip() or hdr.get("OBJCTDEC")
            cdelt1_s = self.cdelt1_edit.text().strip() or hdr.get("CDELT1")
            cdelt2_s = self.cdelt2_edit.text().strip() or hdr.get("CDELT2")

            if objctra is None or objctdec is None:
                raise ValueError("OBJCTRA and OBJCTDEC must be provided either in header or the form fields.")
            if cdelt1_s is None or cdelt2_s is None:
                raise ValueError("CDELT1 and CDELT2 (pixel scales) must be provided either in header or the form fields.")

            # convert RA/DEC to degrees
            ra_deg = Angle(objctra, unit="hourangle").degree
            dec_deg = Angle(objctdec, unit="deg").degree

            # reference pixel: center of image if available
            if self.header is not None and self.header.get("NAXIS1") and self.header.get("NAXIS2"):
                naxis1 = float(self.header.get("NAXIS1"))
                naxis2 = float(self.header.get("NAXIS2"))
                crpix1 = (naxis1 + 1.0) / 2.0
                crpix2 = (naxis2 + 1.0) / 2.0
            else:
                crpix1 = crpix2 = 0.0

            # high precision mpmath floats for scales
            cdelt1_mp = mp.mpf(str(cdelt1_s))
            cdelt2_mp = mp.mpf(str(cdelt2_s))

            solved_cd11 = mp.mpf(self.solved_cd11_edit.text().strip() or "-3.542246e-4")
            scale_factor = abs(solved_cd11) / cdelt1_mp
            scale = scale_factor * cdelt1_mp

            rotation_deg = mp.mpf(self.rotation_edit.text().strip() or "179.7")
            theta = rotation_deg * (mp.pi / mp.mpf("180"))

            cd1_1 = -scale * mp.cos(theta)
            cd1_2 =  scale * mp.sin(theta)
            cd2_1 =  scale * mp.sin(theta)
            cd2_2 =  scale * mp.cos(theta)

            # Prepare preview text
            lines = [
                f"CRPIX1 = {crpix1}",
                f"CRPIX2 = {crpix2}",
                f"CRVAL1 (RA deg) = {ra_deg}",
                f"CRVAL2 (DEC deg) = {dec_deg}",
                f"CROTA (deg) = {float(rotation_deg)}",
                f"CD1_1 = {mp.nstr(cd1_1, 12)}",
                f"CD1_2 = {mp.nstr(cd1_2, 12)}",
                f"CD2_1 = {mp.nstr(cd2_1, 12)}",
                f"CD2_2 = {mp.nstr(cd2_2, 12)}",
            ]
            self.values_box.setPlainText("\n".join(lines))
            self._log("WCS computed (preview). You may write these values into the FITS header.")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Compute error", f"{e}\n\n{tb}")

    def _write_wcs_to_fits(self):
        try:
            if not (self.input_path or self.fits_edit.text().strip()):
                QMessageBox.warning(self, "Input required", "Select an input FITS first")
                return
            fits_path = self.input_path or self.fits_edit.text().strip()
            outpath = self.output_edit.text().strip()
            overwrite_orig = bool(self.overwrite_chk.isChecked())

            # compute preview again to ensure values available
            self._compute_wcs_preview()
            txt = self.values_box.toPlainText().strip()
            if not txt:
                raise RuntimeError("No computed WCS values available to write.")

            # parse values from preview text (simple robust parsing)
            parsed = {}
            for line in txt.splitlines():
                if "=" in line:
                    key, val = [s.strip() for s in line.split("=", 1)]
                    parsed[key.split()[0]] = val

            # determine target filename
            if outpath:
                target = outpath
            else:
                if overwrite_orig:
                    target = fits_path
                else:
                    p = Path(fits_path)
                    target = str(p.with_name(p.stem + "_wcs" + p.suffix))

            # open and write header
            with fits.open(fits_path, mode="update") as hdul:
                hdr = hdul[0].header
                # write keys (convert to numeric where appropriate)
                if "CRPIX1" in parsed:
                    hdr["CRPIX1"] = float(parsed["CRPIX1"])
                if "CRPIX2" in parsed:
                    hdr["CRPIX2"] = float(parsed["CRPIX2"])
                if "CRVAL1" in parsed:
                    hdr["CRVAL1"] = float(parsed["CRVAL1"])
                if "CRVAL2" in parsed:
                    hdr["CRVAL2"] = float(parsed["CRVAL2"])
                if "CROTA" in parsed:
                    hdr["CROTA1"] = float(parsed["CROTA"])
                    hdr["CROTA2"] = float(parsed["CROTA"])
                # CD entries
                if "CD1_1" in parsed:
                    hdr["CD1_1"] = float(parsed["CD1_1"])
                if "CD1_2" in parsed:
                    hdr["CD1_2"] = float(parsed["CD1_2"])
                if "CD2_1" in parsed:
                    hdr["CD2_1"] = float(parsed["CD2_1"])
                if "CD2_2" in parsed:
                    hdr["CD2_2"] = float(parsed["CD2_2"])
                hdr["PLTSOLVD"] = True
                hdr["CTYPE1"] = hdr.get("CTYPE1", "RA---TAN")
                hdr["CTYPE2"] = hdr.get("CTYPE2", "DEC--TAN")
                hdr["CUNIT1"] = hdr.get("CUNIT1", "deg")
                hdr["EQUINOX"] = hdr.get("EQUINOX", 2000.0)
                # flush changes to disk if updating original file
                if overwrite_orig:
                    hdul.flush()
                    written = fits_path
                else:
                    # write to new file
                    hdul.writeto(target, overwrite=True)
                    written = target

            self._log(f"Wrote WCS keywords to: {written}")
            QMessageBox.information(self, "Saved", f"Wrote WCS keywords to:\n{written}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Write error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = MxdlAstapWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()