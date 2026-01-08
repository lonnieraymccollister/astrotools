#!/usr/bin/env python3
"""
ProperMotionGUI with FITS load/save support and WCS-aware proper motion propagation.

Features added:
- Time baseline (years) spin box to draw motion for an arbitrary interval.
- Option to treat RA proper motion as pm_ra_cosdec (checkbox, default checked).
- Uses astropy.coordinates.SkyCoord.apply_space_motion for accurate propagation when WCS is available.
- Falls back to legacy pixel-vector behavior when WCS is missing or propagation fails.
- Includes WCS axis orientation check (check_wcs_axes)
- Includes Line Width (px) input box (default 1)
"""
import sys
from pathlib import Path
import traceback
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QMessageBox, QDoubleSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt


# ------------------------------------------------------------
#  WCS AXIS CHECK HELPER
# ------------------------------------------------------------
def check_wcs_axes(header, x=None, y=None):
    """
    Check WCS axis directions near pixel (x,y). If x,y are None, use image center.
    Returns:
      {
        'ra_per_px': float,
        'dec_per_px': float,
        'ra_decreases_right': bool,
        'dec_increases_up': bool
      }
    """
    if header is None:
        raise ValueError("No header provided for WCS check")

    wcs = WCS(header, naxis=2)

    # Determine test pixel
    if x is None or y is None:
        try:
            nx = int(header.get("NAXIS1", 0))
            ny = int(header.get("NAXIS2", 0))
            if nx > 0 and ny > 0:
                x = nx / 2
                y = ny / 2
            else:
                x = y = 0.0
        except Exception:
            x = y = 0.0

    # Convert pixel → world
    w0 = wcs.wcs_pix2world([[x, y]], 0)[0]
    w_x1 = wcs.wcs_pix2world([[x + 1, y]], 0)[0]
    w_y1 = wcs.wcs_pix2world([[x, y + 1]], 0)[0]

    ra0, dec0 = w0
    ra_x1, dec_x1 = w_x1
    ra_y1, dec_y1 = w_y1

    # RA wrap-safe difference
    def ra_diff(a, b):
        return (b - a + 180) % 360 - 180

    dra_dx = ra_diff(ra0, ra_x1)
    ddec_dy = dec_y1 - dec0

    ra_per_px = dra_dx
    dec_per_px = ddec_dy

    return {
        "ra_per_px": ra_per_px,
        "dec_per_px": dec_per_px,
        "ra_decreases_right": (ra_per_px < 0),
        "dec_increases_up": (dec_per_px < 0),
        "dra_dx": dra_dx,
        "ddec_dy": ddec_dy
    }


# ------------------------------------------------------------
#  MAIN GUI CLASS
# ------------------------------------------------------------
class ProperMotionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Proper Motion Line Drawer")
        self._build_ui()
        self.input_path = None
        self.output_path = None
        self._last_fits_header = None
        self._wcs_orientation = None

    def _build_ui(self):
        grid = QGridLayout()
        self.setLayout(grid)

        # Proper motion inputs
        grid.addWidget(QLabel("Proper Motion RA (mas/yr)X(-1 using Png,jpg):"), 0, 0)
        self.pm_ra_edit = QLineEdit("6.74")
        grid.addWidget(self.pm_ra_edit, 0, 1)

        grid.addWidget(QLabel("Proper Motion Dec (mas/yr):"), 1, 0)
        self.pm_dec_edit = QLineEdit("-1.48")
        grid.addWidget(self.pm_dec_edit, 1, 1)

        # pm_ra_cosdec checkbox
        self.pm_ra_cosdec_check = QCheckBox("pm_ra is pm_ra_cosdec (catalog convention)")
        self.pm_ra_cosdec_check.setChecked(True)
        grid.addWidget(self.pm_ra_cosdec_check, 2, 0, 1, 2)

        # Time baseline
        grid.addWidget(QLabel("Time baseline (years):"), 3, 0)
        self.years_spin = QDoubleSpinBox()
        self.years_spin.setRange(0, 1e6)
        self.years_spin.setDecimals(3)
        self.years_spin.setValue(1.0)
        grid.addWidget(self.years_spin, 3, 1)

        # Start coordinates
        grid.addWidget(QLabel("Start X (px)Values in Siril:"), 4, 0)
        self.x_edit = QLineEdit("100")
        grid.addWidget(self.x_edit, 4, 1)

        grid.addWidget(QLabel("Start Y (px)Values in Siril:Using fits clk Mirror then use"), 5, 0)
        self.y_edit = QLineEdit("100")
        grid.addWidget(self.y_edit, 5, 1)

        # Length
        grid.addWidget(QLabel("Line Length (px):"), 6, 0)
        self.len_edit = QLineEdit("200")
        grid.addWidget(self.len_edit, 6, 1)

        # ------------------------------------------------------------
        # NEW: Line Width
        # ------------------------------------------------------------
        grid.addWidget(QLabel("Line Width (px):"), 7, 0)
        self.line_width_edit = QLineEdit("1")
        grid.addWidget(self.line_width_edit, 7, 1)

        # Browse input
        self.btn_in = QPushButton("Browse Input Image or FITS")
        self.btn_in.clicked.connect(self._browse_input)
        grid.addWidget(self.btn_in, 8, 0, 1, 2)

        # Browse output
        self.btn_out = QPushButton("Browse Output Image or FITS")
        self.btn_out.clicked.connect(self._browse_output)
        grid.addWidget(self.btn_out, 9, 0, 1, 2)

        # Run button
        self.run_btn = QPushButton("Draw Line and Save")
        self.run_btn.clicked.connect(self._run)
        grid.addWidget(self.run_btn, 10, 0, 1, 2)

    # ------------------------------------------------------------
    #  INPUT BROWSER (RUNS WCS CHECK)
    # ------------------------------------------------------------
    def _browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image or FITS",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff);;FITS Files (*.fits *.fit);;All Files (*)"
        )
        if not fn:
            return

        self.input_path = fn
        self.btn_in.setText(f"Input: {Path(fn).name}")

        # If FITS, run WCS axis check
        if fn.lower().endswith((".fits", ".fit")):
            try:
                hdr = fits.getheader(fn)
                self._last_fits_header = hdr.copy()

                orient = check_wcs_axes(hdr)
                self._wcs_orientation = orient

                msg = (
                    f"RA per +x pixel: {orient['ra_per_px']:.6e} deg/px\n"
                    f"Dec per +y pixel: {orient['dec_per_px']:.6e} deg/px\n\n"
                    f"RA decreases left→right? {'YES' if orient['ra_decreases_right'] else 'NO'}\n"
                    f"Dec increases bottom→top? {'YES' if orient['dec_increases_up'] else 'NO'}"
                )
                QMessageBox.information(self, "WCS Axis Check", msg)

            except Exception as e:
                self._wcs_orientation = None
                QMessageBox.warning(self, "WCS Check Failed", str(e))

    # ------------------------------------------------------------
    #  OUTPUT BROWSER
    # ------------------------------------------------------------
    def _browse_output(self):
        fn, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output Image or FITS",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;FITS (*.fits);;All Files (*)"
        )
        if fn:
            self.output_path = fn
            self.btn_out.setText(f"Output: {Path(fn).name}")

    # ------------------------------------------------------------
    #  LOAD IMAGE (FITS OR RASTER)
    # ------------------------------------------------------------
    def _load_image_as_pil(self, path):
        p = Path(path)
        if p.suffix.lower() in (".fits", ".fit"):
            with fits.open(path) as hdul:
                data = hdul[0].data
                hdr = hdul[0].header

            self._last_fits_header = hdr.copy()

            arr = np.asarray(data)
            if arr.ndim == 3:
                if arr.shape[0] == 3:
                    rgb = np.transpose(arr[:3], (1, 2, 0))
                elif arr.shape[2] == 3:
                    rgb = arr[..., :3]
                else:
                    raise ValueError("Unsupported FITS RGB shape")
            elif arr.ndim == 2:
                rgb = np.stack([arr]*3, axis=-1)
            else:
                raise ValueError("Unsupported FITS ndim")

            # Normalize to 0–255
            out = np.zeros_like(rgb, dtype=np.uint8)
            for c in range(3):
                ch = rgb[..., c].astype(float)
                finite = np.isfinite(ch)
                if not finite.any():
                    continue
                ch[~finite] = np.nanmedian(ch[finite])
                mn, mx = np.nanmin(ch), np.nanmax(ch)
                if mx > mn:
                    out[..., c] = ((ch - mn) / (mx - mn) * 255).astype(np.uint8)

            return Image.fromarray(out, "RGB")

        # Raster
        self._last_fits_header = None
        return Image.open(path).convert("RGB")

    # ------------------------------------------------------------
    #  SAVE PIL → FITS
    # ------------------------------------------------------------
    def _save_pil_to_fits(self, pil_img, outpath):
        arr = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
        fits_data = np.transpose(arr, (2, 0, 1))

        hdr = self._last_fits_header.copy() if self._last_fits_header else fits.Header()
        hdr["NAXIS"] = 3
        hdr["NAXIS1"] = fits_data.shape[2]
        hdr["NAXIS2"] = fits_data.shape[1]
        hdr["NAXIS3"] = fits_data.shape[0]

        fits.PrimaryHDU(fits_data, header=hdr).writeto(outpath, overwrite=True)

    # ------------------------------------------------------------
    #  WCS PROPAGATION
    # ------------------------------------------------------------
    def _pm_vector_pixels_from_wcs(self, header, x0, y0, pm_ra_mas, pm_dec_mas, years, pm_ra_is_cosdec):
        wcs = WCS(header, naxis=2)

        ra0, dec0 = wcs.wcs_pix2world([[x0, y0]], 0)[0]

        # Determine epoch
        if "DATE-OBS" in header:
            obstime = Time(header["DATE-OBS"])
        else:
            obstime = Time(2000.0, format="jyear")

        new_time = obstime + years * u.year

        if pm_ra_is_cosdec:
            sc = SkyCoord(
                ra=ra0*u.deg, dec=dec0*u.deg,
                pm_ra_cosdec=pm_ra_mas*u.mas/u.yr,
                pm_dec=pm_dec_mas*u.mas/u.yr,
                frame="icrs", obstime=obstime
            )
        else:
            sc = SkyCoord(
                ra=ra0*u.deg, dec=dec0*u.deg,
                pm_ra=pm_ra_mas*u.mas/u.yr,
                pm_dec=pm_dec_mas*u.mas/u.yr,
                frame="icrs", obstime=obstime
            )

        sc_new = sc.apply_space_motion(new_time)

        ra1, dec1 = sc_new.ra.deg, sc_new.dec.deg
        x1, y1 = wcs.wcs_world2pix([[ra1, dec1]], 0)[0]
        return float(x1), float(y1)

    # ------------------------------------------------------------
    #  MAIN RUN
    # ------------------------------------------------------------
    def _run(self):
        try:
            if not self.input_path:
                raise ValueError("No input image selected")
            if not self.output_path:
                raise ValueError("No output filename selected")

            pm_ra = float(self.pm_ra_edit.text())
            pm_dec = float(self.pm_dec_edit.text())
            x0 = float(self.x_edit.text())
            y0 = float(self.y_edit.text())
            length = float(self.len_edit.text())
            years = float(self.years_spin.value())
            pm_ra_is_cosdec = self.pm_ra_cosdec_check.isChecked()

            # NEW: line width
            line_width = int(self.line_width_edit.text())
            if line_width < 1:
                line_width = 1

            # Compute vector
            used_wcs = False
            if self._last_fits_header:
                try:
                    x1w, y1w = self._pm_vector_pixels_from_wcs(
                        self._last_fits_header, x0, y0,
                        pm_ra, pm_dec, years, pm_ra_is_cosdec
                    )
                    dx, dy = x1w - x0, y1w - y0
                    mag = np.hypot(dx, dy)
                    if mag > 0:
                        ux, uy = dx/mag, dy/mag
                        x1 = x0 + ux*length
                        y1 = y0 + uy*length
                        used_wcs = True
                except Exception:
                    used_wcs = False

            if not used_wcs:
                dx, dy = pm_ra, -pm_dec
                mag = np.hypot(dx, dy)
                if mag == 0:
                    raise ValueError("Proper motion vector is zero")
                ux, uy = dx/mag, dy/mag
                x1 = x0 + ux*length
                y1 = y0 + uy*length

            # Load image
            pil = self._load_image_as_pil(self.input_path)

            # Draw line
            draw = ImageDraw.Draw(pil)
            draw.line((x0, y0, x1, y1), fill=(255,0,0), width=line_width)

            # Save
            out = Path(self.output_path)
            if out.suffix.lower() in (".fits", ".fit"):
                self._save_pil_to_fits(pil, str(out))
            else:
                if out.suffix.lower() in (".jpg", ".jpeg"):
                    pil.save(str(out), quality=95)
                else:
                    pil.save(str(out))

            QMessageBox.information(
                self, "Done",
                f"Saved to:\n{self.output_path}\n"
                + ("(WCS used)" if used_wcs else "(pixel fallback)")
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", traceback.format_exc())


def main():
    app = QApplication(sys.argv)
    w = ProperMotionGUI()
    w.resize(520, 420)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()