#!/usr/bin/env python3
"""
fits_lrgb_combine_gui.py
PyQt6 GUI to combine L, R, G, B FITS into a 3-plane (C,Y,X) cube.
Options: L*RGB (multiply), L-weighted blend, or assemble channels directly.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QComboBox, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt

# ---------- Helpers ----------
def read_primary_fits(path):
    with fits.open(path) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header.copy()
    if data is None:
        raise ValueError(f"No data in primary HDU of {path}")
    return np.array(data, dtype=np.float64), hdr

def safe_normalize(arr):
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a)
    mn = np.nanmin(a[finite])
    mx = np.nanmax(a[finite])
    if mx <= mn:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)

def make_rgb_cube(lum, r, g, b, mode="L*RGB", l_weight=1.0):
    """
    Returns cube shaped (3, Y, X), dtype float32 in 0..1 range (or close).
    Modes:
      - "L*RGB": multiply normalized R,G,B by normalized L (per-pixel).
      - "L-weighted": blend: out = normalize( (1-l_weight)*RGB + l_weight*(L replicated) )
      - "Channels only": use R,G,B normalized, ignore L (use L only for luminance if requested)
    """
    # ensure shapes compatible: prefer broadcasting rules; require same Y,X
    # Attempt to convert shapes: if lum is (Y,X) and channels are (Y,X) good.
    # If any channel has 3D (C,Y,X) use first plane.
    def ensure2d(x):
        x = np.asarray(x)
        if x.ndim == 3:
            # try channel-first (C,Y,X)
            if x.shape[0] in (3,4):
                return x[0]
            # else channel-last Y,X,C
            return x[..., 0]
        if x.ndim == 2:
            return x
        raise ValueError("Unsupported input array shape for combining")
    L = ensure2d(lum)
    R = ensure2d(r)
    G = ensure2d(g)
    B = ensure2d(b)
    if L.shape != R.shape or L.shape != G.shape or L.shape != B.shape:
        raise ValueError("All images must share the same Y,X shape")
    # normalize
    Ln = safe_normalize(L)
    Rn = safe_normalize(R)
    Gn = safe_normalize(G)
    Bn = safe_normalize(B)
    if mode == "L*RGB":
        r_out = Rn * Ln
        g_out = Gn * Ln
        b_out = Bn * Ln
    elif mode == "L-weighted":
        # blend RGB with replicated L (equally to three channels)
        rgb = np.stack((Rn, Gn, Bn), axis=-1)
        Lrep = np.stack([Ln, Ln, Ln], axis=-1)
        blended = (1.0 - l_weight) * rgb + l_weight * Lrep
        # channel-wise normalize blended to 0..1
        r_out, g_out, b_out = blended[...,0], blended[...,1], blended[...,2]
    elif mode == "Channels only":
        r_out, g_out, b_out = Rn, Gn, Bn
    else:
        raise ValueError("Unknown combine mode")
    # clip
    r_out = np.clip(r_out, 0.0, 1.0)
    g_out = np.clip(g_out, 0.0, 1.0)
    b_out = np.clip(b_out, 0.0, 1.0)
    # assemble channel-first cube
    cube = np.stack((r_out, g_out, b_out), axis=0).astype(np.float32)
    return cube

# ---------- GUI ----------
class FitsCombinerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS LRGB Combine")
        self._build_ui()
        self.resize(700, 360)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        def row(label_text, key):
            h = QHBoxLayout()
            lbl = QLabel(label_text)
            le = QLineEdit()
            btn = QPushButton("Browse")
            btn.clicked.connect(lambda _, e=le: self._browse_open(e))
            h.addWidget(lbl)
            h.addWidget(le, stretch=1)
            h.addWidget(btn)
            layout.addLayout(h)
            return le

        self.lum_le = row("Luminance FITS:", "lum")
        self.blue_le = row("Blue FITS:", "blue")
        self.green_le = row("Green FITS:", "green")
        self.red_le = row("Red FITS:", "red")

        hout = QHBoxLayout()
        out_lbl = QLabel("Output FITS:")
        self.out_le = QLineEdit()
        out_btn = QPushButton("Save As")
        out_btn.clicked.connect(lambda: self._browse_save(self.out_le))
        hout.addWidget(out_lbl)
        hout.addWidget(self.out_le, stretch=1)
        hout.addWidget(out_btn)
        layout.addLayout(hout)

        # Combine options
        opts_h = QHBoxLayout()
        opts_h.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["L-weighted", "L*RGB", "Channels only"])
        opts_h.addWidget(self.mode_combo)
        opts_h.addWidget(QLabel("L weight (for L-weighted):"))
        self.lweight_le = QLineEdit("1.0")
        self.lweight_le.setFixedWidth(60)
        opts_h.addWidget(self.lweight_le)
        layout.addLayout(opts_h)

        # Buttons
        btn_h = QHBoxLayout()
        combine_btn = QPushButton("Combine and Save")
        combine_btn.clicked.connect(self._on_combine)
        btn_h.addWidget(combine_btn)
        preview_btn = QPushButton("Preview Stats (no file)")
        preview_btn.clicked.connect(self._on_preview_stats)
        btn_h.addWidget(preview_btn)
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(lambda: self.log.clear())
        btn_h.addWidget(clear_btn)
        layout.addLayout(btn_h)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log, stretch=1)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_open(self, lineedit):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            lineedit.setText(fn)

    def _browse_save(self, lineedit):
        fn, _ = QFileDialog.getSaveFileName(self, "Save FITS as", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            lineedit.setText(fn)

    def _validate_paths(self):
        paths = {
            "Luminance": self.lum_le.text().strip(),
            "Blue": self.blue_le.text().strip(),
            "Green": self.green_le.text().strip(),
            "Red": self.red_le.text().strip(),
            "Output": self.out_le.text().strip()
        }
        for k,v in paths.items():
            if not v:
                raise ValueError(f"{k} path is empty")
        return paths

    def _on_preview_stats(self):
        try:
            paths = self._validate_paths()
            lum, _ = read_primary_fits(paths["Luminance"])
            r, _ = read_primary_fits(paths["Red"])
            g, _ = read_primary_fits(paths["Green"])
            b, _ = read_primary_fits(paths["Blue"])
            self._log("Shapes: L={}, R={}, G={}, B={}".format(lum.shape, r.shape, g.shape, b.shape))
            # basic stats
            def stats(a):
                a = np.asarray(a).ravel()
                a = a[np.isfinite(a)]
                if a.size == 0:
                    return ("nan","nan","nan")
                return (float(a.min()), float(np.median(a)), float(a.max()))
            self._log("L min/med/max: {:.5g} / {:.5g} / {:.5g}".format(*stats(lum)))
            self._log("R min/med/max: {:.5g} / {:.5g} / {:.5g}".format(*stats(r)))
            self._log("G min/med/max: {:.5g} / {:.5g} / {:.5g}".format(*stats(g)))
            self._log("B min/med/max: {:.5g} / {:.5g} / {:.5g}".format(*stats(b)))
            self._log("Preview normalization will be applied before combining.")
        except Exception as e:
            QMessageBox.critical(self, "Preview error", str(e))

    def _on_combine(self):
        try:
            paths = self._validate_paths()
            outpath = paths["Output"]
            mode = self.mode_combo.currentText()
            try:
                l_weight = float(self.lweight_le.text().strip())
            except Exception:
                l_weight = 1.0

            # read inputs
            lum, header = read_primary_fits(paths["Luminance"])
            blue, _ = read_primary_fits(paths["Blue"])
            green, _ = read_primary_fits(paths["Green"])
            red, _ = read_primary_fits(paths["Red"])

            self._log("Read inputs. shapes: L={}, R={}, G={}, B={}".format(lum.shape, red.shape, green.shape, blue.shape))

            cube = make_rgb_cube(lum, red, green, blue, mode=mode, l_weight=l_weight)

            # update header provenance
            if header is None:
                header = fits.Header()
            header["NAXIS"] = 3
            header["NAXIS1"] = cube.shape[2]
            header["NAXIS2"] = cube.shape[1]
            header["NAXIS3"] = cube.shape[0]
            header["COMBINE"] = (mode, "Combine mode")
            header["LWEIGHT"] = (l_weight, "L weight used (if applicable)")

            fits.writeto(outpath, cube, header=header, overwrite=True)
            self._log(f"Wrote combined FITS to: {outpath}  shape={cube.shape} dtype={cube.dtype}")
            QMessageBox.information(self, "Done", f"Combined saved to:\n{outpath}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error: " + str(e))
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = FitsCombinerGUI()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()