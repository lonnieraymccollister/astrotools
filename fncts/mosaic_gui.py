#!/usr/bin/env python3
"""
mosaic_gui.py
PyQt6 GUI for combining two aligned FITS images with brightness equalization
and feathered edge blending.
"""

import sys
import os
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QTextEdit, QMessageBox, QComboBox, QSpinBox
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import stats as sp_stats


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_fits(path):
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float64), hdul[0].header

def write_fits(path, data, header=None):
    fits.writeto(path, data.astype(np.float64), header=header, overwrite=True)

def compute_stats(arr):
    a = np.asarray(arr).ravel()
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"min": np.nan, "max": np.nan, "median": np.nan, "mean": np.nan, "std": np.nan}
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "std": float(a.std())
    }

def sigma_clipped_stats(arr, sigma=3.0):
    a = np.asarray(arr).ravel()
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan, np.nan, np.nan
    clipped, _, _ = sp_stats.sigmaclip(a, low=sigma, high=sigma)
    if clipped.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(clipped)), float(clipped.mean()), float(clipped.std())


# ------------------------------------------------------------
# Histogram Canvas
# ------------------------------------------------------------

class HistCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6,3))
        super().__init__(fig)
        self.ax1 = fig.add_subplot(1,2,1)
        self.ax2 = fig.add_subplot(1,2,2)
        fig.tight_layout()

    def plot(self, imgA=None, imgB=None, bins=256):
        self.ax1.clear()
        self.ax2.clear()

        if imgA is not None:
            self.ax1.hist(imgA.ravel(), bins=bins, color='blue', histtype='step')
            self.ax1.set_title("Image A")
        else:
            self.ax1.set_title("Image A: none")

        if imgB is not None:
            self.ax2.hist(imgB.ravel(), bins=bins, color='red', histtype='step')
            self.ax2.set_title("Image B / Mosaic")
        else:
            self.ax2.set_title("Image B: none")

        self.draw()


# ------------------------------------------------------------
# Main Window
# ------------------------------------------------------------

class MosaicWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FITS Mosaic / Edge Blend Tool")
        self.imgA = None
        self.imgB = None
        self.hdrA = None
        self.hdrB = None
        self.mosaic = None
        self._build_ui()
        self.resize(1100, 650)

    def _build_ui(self):
        from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout

        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)

        # ----------------------------------------------------
        # INPUT SECTION
        # ----------------------------------------------------
        box_input = QGroupBox("Input Images")
        grid_in = QGridLayout()
        box_input.setLayout(grid_in)

        lblA = QLabel("Image A:")
        lblA.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid_in.addWidget(lblA, 0, 0)

        self.inA = QLineEdit()
        grid_in.addWidget(self.inA, 0, 1)
        btnA = QPushButton("Browse")
        btnA.clicked.connect(lambda: self._browse_file(self.inA, "A"))
        grid_in.addWidget(btnA, 0, 2)

        lblB = QLabel("Image B:")
        lblB.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid_in.addWidget(lblB, 1, 0)

        self.inB = QLineEdit()
        grid_in.addWidget(self.inB, 1, 1)
        btnB = QPushButton("Browse")
        btnB.clicked.connect(lambda: self._browse_file(self.inB, "B"))
        grid_in.addWidget(btnB, 1, 2)

        # Output
        lblOut = QLabel("Output FITS:")
        lblOut.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid_in.addWidget(lblOut, 2, 0)

        self.out = QLineEdit("mosaic_output.fits")
        grid_in.addWidget(self.out, 2, 1)
        btnOut = QPushButton("Browse")
        btnOut.clicked.connect(self._browse_output)
        grid_in.addWidget(btnOut, 2, 2)

        main.addWidget(box_input)

        # ----------------------------------------------------
        # SETTINGS
        # ----------------------------------------------------
        box_set = QGroupBox("Mosaic Settings")
        grid_set = QGridLayout()
        box_set.setLayout(grid_set)

        lblLayout = QLabel("Layout:")
        lblLayout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid_set.addWidget(lblLayout, 0, 0)

        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Left-Right", "Top-Bottom"])
        grid_set.addWidget(self.layout_combo, 0, 1)

        lblOv = QLabel("Overlap width:")
        lblOv.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid_set.addWidget(lblOv, 1, 0)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 10000)
        self.overlap_spin.setValue(0)
        grid_set.addWidget(self.overlap_spin, 1, 1)

        main.addWidget(box_set)

        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------
        box_act = QGroupBox("Actions")
        h_act = QHBoxLayout()
        box_act.setLayout(h_act)

        btn_preview = QPushButton("Preview Histograms")
        btn_preview.clicked.connect(self._preview)
        h_act.addWidget(btn_preview)

        btn_blend = QPushButton("Blend & Preview Mosaic")
        btn_blend.clicked.connect(self._blend_preview)
        h_act.addWidget(btn_blend)

        btn_save = QPushButton("Save Mosaic")
        btn_save.clicked.connect(self._save_mosaic)
        h_act.addWidget(btn_save)

        main.addWidget(box_act)

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------
        box_info = QGroupBox("Information")
        v_info = QVBoxLayout()
        box_info.setLayout(v_info)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        v_info.addWidget(self.info)

        main.addWidget(box_info)

        # ----------------------------------------------------
        # HISTOGRAM
        # ----------------------------------------------------
        box_hist = QGroupBox("Histogram Preview")
        v_hist = QVBoxLayout()
        box_hist.setLayout(v_hist)

        self.canvas = HistCanvas()
        v_hist.addWidget(self.canvas)

        main.addWidget(box_hist)


    # ------------------------------------------------------------
    # File browsing
    # ------------------------------------------------------------

    def _browse_file(self, lineedit, which):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS", "", "FITS Files (*.fits *.fit)")
        if fn:
            lineedit.setText(fn)
            try:
                data, hdr = load_fits(fn)
                if which == "A":
                    self.imgA, self.hdrA = data, hdr
                else:
                    self.imgB, self.hdrB = data, hdr
                st = compute_stats(data)
                self.info.append(f"Loaded {os.path.basename(fn)} stats: min={st['min']:.6g} max={st['max']:.6g}")
            except Exception as e:
                QMessageBox.critical(self, "Load error", f"Failed to load FITS: {e}")

    def _browse_output(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save mosaic as", "", "FITS Files (*.fits *.fit)")
        if fn:
            self.out.setText(fn)

    # ------------------------------------------------------------
    # Preview histograms
    # ------------------------------------------------------------

    def _preview(self):
        self.canvas.plot(self.imgA, self.imgB)

    # ------------------------------------------------------------
    # Blend preview
    # ------------------------------------------------------------

    def _blend_preview(self):
        if self.imgA is None or self.imgB is None:
            QMessageBox.warning(self, "Missing images", "Load both FITS images first")
            return

        try:
            if self.imgA.shape != self.imgB.shape:
                QMessageBox.warning(self, "Shape mismatch", "Images must be same shape for now")
                return

            # Brightness equalization
            medA, meanA, stdA = sigma_clipped_stats(self.imgA)
            medB, meanB, stdB = sigma_clipped_stats(self.imgB)

            scale = 1.0 if stdB == 0 else stdA / stdB
            offset = medA - scale * medB

            imgB_eq = scale * self.imgB + offset

            self.info.append(
                f"Equalization: scale={scale:.6g}, offset={offset:.6g}"
            )

            # Simple blend for now
            self.mosaic = 0.5 * self.imgA + 0.5 * imgB_eq

            self.canvas.plot(self.imgA, self.mosaic)
            st = compute_stats(self.mosaic)
            self.info.append(f"Mosaic preview: min={st['min']:.6g} max={st['max']:.6g}")

        except Exception as e:
            QMessageBox.critical(self, "Blend error", str(e))

    # ------------------------------------------------------------
    # Save mosaic
    # ------------------------------------------------------------

    def _save_mosaic(self):
        if self.mosaic is None:
            QMessageBox.warning(self, "No mosaic", "Run Blend first")
            return
        outpath = self.out.text().strip()
        if not outpath:
            QMessageBox.warning(self, "No output", "Specify output filename")
            return
        try:
            write_fits(outpath, self.mosaic, header=self.hdrA)
            self.info.append(f"Saved mosaic to {outpath}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    w = MosaicWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
