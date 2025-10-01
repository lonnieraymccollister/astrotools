#!/usr/bin/env python3
"""
plot3d_gui.py
PyQt6 GUI to load a grayscale image (8/16-bit) and plot a 3D surface of pixel values.
Includes dropdowns for colormap and plot style, downsample factor, and save option.
"""
import sys
import os
import traceback
from pathlib import Path

import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QComboBox, QTextEdit, QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ---------- Matplotlib canvas ----------
class SurfaceCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=(6,5), dpi=110):
        fig = Figure(figsize=figsize, dpi=dpi)
        super().__init__(fig)
        self.fig = fig
        self.ax = fig.add_subplot(1,1,1, projection='3d')

    def plot_surface(self, arr, downsample=1, cmap='viridis', style='surface'):
        self.fig.clf()
        try:
            ax = self.fig.add_subplot(1,1,1, projection='3d')
            # prepare grid
            h, w = arr.shape
            if downsample > 1:
                arr_ds = arr[::downsample, ::downsample]
            else:
                arr_ds = arr
            yy, xx = np.mgrid[0:arr_ds.shape[0], 0:arr_ds.shape[1]]
            # choose plotting style
            if style == 'surface':
                surf = ax.plot_surface(xx, yy, arr_ds, cmap=cmap, rstride=1, cstride=1, linewidth=0, antialiased=True)
                self.fig.colorbar(surf, ax=ax, shrink=0.6)
            elif style == 'wire':
                ax.plot_wireframe(xx, yy, arr_ds, rstride=1, cstride=1, color='k')
            elif style == 'contour':
                ax.contourf(xx, yy, arr_ds, zdir='z', offset=np.min(arr_ds) - (np.ptp(arr_ds)*0.1), cmap=cmap)
            else:
                surf = ax.plot_surface(xx, yy, arr_ds, cmap=cmap, rstride=1, cstride=1, linewidth=0)
                self.fig.colorbar(surf, ax=ax, shrink=0.6)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Intensity")
            ax.view_init(elev=45, azim=-60)
            self.ax = ax
        except Exception:
            self.fig.clf()
            ax = self.fig.add_subplot(1,1,1)
            ax.text(0.5,0.5,"Error plotting surface", ha='center')
        self.draw()

    def save(self, outpath, dpi=200):
        self.fig.savefig(outpath, dpi=dpi, bbox_inches='tight')

# ---------- GUI ----------
class Plot3DWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image -> 3D Surface")
        self._build_ui()
        self.resize(1000, 700)
        self.img = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Row 0: file picker
        grid.addWidget(QLabel("Input image:"), 0, 0)
        self.file_edit = QLineEdit()
        grid.addWidget(self.file_edit, 0, 1, 1, 3)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._browse_file)
        grid.addWidget(btn_browse, 0, 4)

        # Row 1: downsample factor
        grid.addWidget(QLabel("Downsample factor (int >=1):"), 1, 0)
        self.down_spin = QSpinBox(); self.down_spin.setRange(1, 64); self.down_spin.setValue(1)
        grid.addWidget(self.down_spin, 1, 1)

        # Row 1: colormap
        grid.addWidget(QLabel("Colormap:"), 1, 2)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'jet', 'gray'])
        grid.addWidget(self.cmap_combo, 1, 3)

        # Row 1: style
        grid.addWidget(QLabel("Style:"), 1, 4)
        self.style_combo = QComboBox()
        self.style_combo.addItems(['surface', 'wire', 'contour'])
        grid.addWidget(self.style_combo, 1, 5)

        # Row 2: buttons
        self.load_btn = QPushButton("Load image")
        self.load_btn.clicked.connect(self._load_image)
        grid.addWidget(self.load_btn, 2, 0)

        self.plot_btn = QPushButton("Plot 3D Surface")
        self.plot_btn.clicked.connect(self._on_plot)
        grid.addWidget(self.plot_btn, 2, 1)

        self.save_btn = QPushButton("Save Figure")
        self.save_btn.clicked.connect(self._on_save)
        grid.addWidget(self.save_btn, 2, 2)

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(lambda: self.log.clear())
        grid.addWidget(clear_btn, 2, 3)

        # Canvas and log
        self.canvas = SurfaceCanvas(figsize=(6,5), dpi=110)
        grid.addWidget(self.canvas, 3, 0, 6, 6)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(140)
        grid.addWidget(self.log, 9, 0, 1, 6)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_file(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)")
        if fn:
            self.file_edit.setText(fn)

    def _load_image(self):
        path = self.file_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No file", "Select an image file first")
            return
        try:
            # Read as grayscale, preserving bit depth if possible
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(path)
            # If multi-channel, convert to grayscale
            if img.ndim == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            # convert to float64 for plotting; preserve dynamic range
            imgf = img_gray.astype(np.float64)
            self.img = imgf
            self._log(f"Loaded {Path(path).name} shape={imgf.shape} dtype={img_gray.dtype} min={np.nanmin(imgf):.3g} max={np.nanmax(imgf):.3g}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"{e}\n\n{tb}")

    def _on_plot(self):
        if self.img is None:
            self._load_image()
            if self.img is None:
                return
        try:
            down = int(self.down_spin.value())
            cmap = self.cmap_combo.currentText()
            style = self.style_combo.currentText()
            # optionally normalize to 0..1 for nicer color ranges
            arr = np.array(self.img, dtype=np.float64)
            if np.isfinite(arr).any():
                mn, mx = np.nanmin(arr), np.nanmax(arr)
                if mx > mn:
                    arrn = (arr - mn) / (mx - mn)
                    # scale back to a useful Z-range while preserving relative shape
                    arr_plot = arrn
                else:
                    arr_plot = np.zeros_like(arr)
            else:
                raise ValueError("Image contains no finite values")
            self.canvas.plot_surface(arr_plot, downsample=down, cmap=cmap, style=style)
            self._log(f"Plotted surface with downsample={down}, cmap={cmap}, style={style}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Plot error", f"{e}\n\n{tb}")

    def _on_save(self):
        if self.canvas is None or self.canvas.fig is None:
            QMessageBox.information(self, "Nothing", "No plotted figure to save")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save figure", "", "PNG Image (*.png);;JPEG Image (*.jpg);;PDF (*.pdf);;All Files (*)")
        if not fn:
            return
        try:
            self.canvas.save(fn, dpi=300)
            self._log(f"Saved figure to: {fn}")
            QMessageBox.information(self, "Saved", f"Saved: {fn}")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Save error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = Plot3DWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()