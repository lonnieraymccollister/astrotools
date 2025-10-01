#!/usr/bin/env python3
"""
radectwoptang_gui.py
PyQt6 GUI: compute angular separation and arc length between two sky coordinates.
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QComboBox, QGridLayout, QMessageBox, QTextEdit
)
from PyQt6.QtCore import Qt
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

class RaDecWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RaDec Two-Point Angle")
        self._build_ui()
        self.resize(640, 240)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Format selector
        grid.addWidget(QLabel("Input format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "RA: hms / Dec: dms (e.g. 10h21m00s, +20d30m00s)",
            "RA: degrees / Dec: degrees (e.g. 155.25, 20.5)"
        ])
        grid.addWidget(self.format_combo, 0, 1, 1, 3)

        # First coordinate
        grid.addWidget(QLabel("RA 1:"), 1, 0)
        self.ra1_edit = QLineEdit("10h21m00s")
        grid.addWidget(self.ra1_edit, 1, 1)
        grid.addWidget(QLabel("Dec 1:"), 1, 2)
        self.dec1_edit = QLineEdit("+20d30m00s")
        grid.addWidget(self.dec1_edit, 1, 3)

        # Second coordinate
        grid.addWidget(QLabel("RA 2:"), 2, 0)
        self.ra2_edit = QLineEdit("11h15m30s")
        grid.addWidget(self.ra2_edit, 2, 1)
        grid.addWidget(QLabel("Dec 2:"), 2, 2)
        self.dec2_edit = QLineEdit("+22d05m15s")
        grid.addWidget(self.dec2_edit, 2, 3)

        # Compute button
        self.compute_btn = QPushButton("Compute Separation")
        self.compute_btn.clicked.connect(self._compute)
        grid.addWidget(self.compute_btn, 3, 0, 1, 2)

        # Clear / Copy buttons
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear)
        grid.addWidget(self.clear_btn, 3, 2)
        self.copy_btn = QPushButton("Copy Result")
        self.copy_btn.clicked.connect(self._copy_result)
        grid.addWidget(self.copy_btn, 3, 3)

        # Result display
        grid.addWidget(QLabel("Result:"), 4, 0)
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        grid.addWidget(self.result_box, 5, 0, 1, 4)

    def _parse_coord(self, ra_text, dec_text):
        fmt = self.format_combo.currentIndex()
        try:
            if fmt == 0:
                # hourangle for RA, degrees for Dec (strings like 10h21m00s, +20d30m00s)
                c = SkyCoord(ra_text, dec_text, unit=(u.hourangle, u.deg))
            else:
                # numeric degrees for both
                ra_val = float(ra_text)
                dec_val = float(dec_text)
                c = SkyCoord(ra_val * u.deg, dec_val * u.deg)
            return c
        except Exception as e:
            raise ValueError(f"Invalid coordinate input: {e}")

    def _to_dms_tuple(self, angle_deg):
        # angle_deg is a Quantity in degrees or float degrees
        deg = float(angle_deg)
        d = int(np.floor(deg))
        m = int(np.floor((deg - d) * 60.0))
        s = (deg - d - m/60.0) * 3600.0
        return d, m, s

    def _compute(self):
        ra1 = self.ra1_edit.text().strip()
        dec1 = self.dec1_edit.text().strip()
        ra2 = self.ra2_edit.text().strip()
        dec2 = self.dec2_edit.text().strip()
        if not (ra1 and dec1 and ra2 and dec2):
            QMessageBox.warning(self, "Input error", "Please fill all four coordinate fields.")
            return
        try:
            c1 = self._parse_coord(ra1, dec1)
            c2 = self._parse_coord(ra2, dec2)
        except ValueError as e:
            QMessageBox.critical(self, "Parse error", str(e))
            return

        # separation as Angle
        sep = c1.separation(c2)
        sep_deg = sep.to(u.deg).value
        sep_rad = sep.to(u.rad).value
        sep_arcsec = sep.to(u.arcsec).value

        d, m, s = self._to_dms_tuple(sep_deg)

        # Build human-readable result
        lines = []
        lines.append(f"Separation: {d}° {m}' {s:.3f}\"")
        lines.append(f"Separation (degrees): {sep_deg:.9f}°")
        lines.append(f"Separation (arcseconds): {sep_arcsec:.6f}\"")
        lines.append(f"Arc length on unit sphere (radians): {sep_rad:.9f}")
        lines.append(f"Arc length on sphere radius R: multiply by R (e.g., R in km)")

        # Also show midpoint optionally (useful)
        try:
            mid = SkyCoord.ra_dec_midpoint(c1, c2)
            lines.append(f"Midpoint (ICRS) RA: {mid.ra.to_string(unit=u.hour, sep=':', pad=True)}  Dec: {mid.dec.to_string(unit=u.deg, sep=':', pad=True)}")
        except Exception:
            # older astropy may not have ra_dec_midpoint; compute simple average in Cartesian
            try:
                vec = (c1.cartesian + c2.cartesian) / 2.0
                mid = SkyCoord(vec, frame=c1.frame)
                lines.append(f"Midpoint (approx) RA: {mid.ra.to_string(unit=u.hour, sep=':', pad=True)}  Dec: {mid.dec.to_string(unit=u.deg, sep=':', pad=True)}")
            except Exception:
                pass

        self.result_box.setPlainText("\n".join(lines))

    def _clear(self):
        self.ra1_edit.clear()
        self.dec1_edit.clear()
        self.ra2_edit.clear()
        self.dec2_edit.clear()
        self.result_box.clear()

    def _copy_result(self):
        txt = self.result_box.toPlainText()
        if txt:
            QApplication.clipboard().setText(txt)
            QMessageBox.information(self, "Copied", "Result copied to clipboard.")
        else:
            QMessageBox.information(self, "Nothing", "No result to copy.")

def main():
    app = QApplication(sys.argv)
    win = RaDecWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()