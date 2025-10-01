#!/usr/bin/env python3
"""
imghiststretch_gui.py
PyQt6 GUI for histogram specification (Rayleigh, Gaussian, Uniform, Exponential, Lognormal)
on a single 2D FITS image. Saves result and shows original vs specified histograms.
"""

import sys
import os
import numpy as np
from astropy.io import fits
from scipy.special import erfinv
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QVBoxLayout, QMessageBox,
    QDoubleSpinBox, QSpinBox, QTextEdit
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# -----------------------------
# Histogram specification functions
# -----------------------------
def rayleigh_specification(image, sigma):
    vals, inv_idx, counts = np.unique(image.ravel(), return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
    epsilon = 1e-12
    safe = np.clip(1 - cdf, epsilon, None)
    new_vals = sigma * np.sqrt(-2.0 * np.log(safe))
    return new_vals[inv_idx].reshape(image.shape)

def gaussian_specification(image, mu, sigma):
    vals, inv_idx, counts = np.unique(image.ravel(), return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
    eps = 1e-12
    safe = np.clip(cdf, eps, 1 - eps)
    new_vals = mu + sigma * np.sqrt(2.0) * erfinv(2.0 * safe - 1.0)
    return new_vals[inv_idx].reshape(image.shape)

def uniform_specification(image, lower, upper):
    vals, inv_idx, counts = np.unique(image.ravel(), return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
    new_vals = lower + cdf * (upper - lower)
    return new_vals[inv_idx].reshape(image.shape)

def exponential_specification(image, lamb):
    vals, inv_idx, counts = np.unique(image.ravel(), return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
    eps = 1e-12
    safe = np.clip(1 - cdf, eps, None)
    new_vals = - (1.0 / lamb) * np.log(safe)
    return new_vals[inv_idx].reshape(image.shape)

def lognormal_specification(image, mu, sigma_ln):
    vals, inv_idx, counts = np.unique(image.ravel(), return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
    eps = 1e-12
    safe = np.clip(cdf, eps, 1 - eps)
    new_vals = np.exp(mu + sigma_ln * np.sqrt(2.0) * erfinv(2.0 * safe - 1.0))
    return new_vals[inv_idx].reshape(image.shape)

# -----------------------------
# GUI
# -----------------------------
class HistCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6,3))
        super().__init__(fig)
        self.axes_orig = fig.add_subplot(1,2,1)
        self.axes_spec = fig.add_subplot(1,2,2)
        fig.tight_layout()

    def plot_histograms(self, orig, spec, bins=256):
        self.axes_orig.clear()
        self.axes_spec.clear()
        self.axes_orig.hist(orig.ravel(), bins=bins, color='blue', histtype='step')
        self.axes_orig.set_title("Original Histogram")
        self.axes_spec.hist(spec.ravel(), bins=bins, color='red', histtype='step')
        self.axes_spec.set_title("Specified Histogram")
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Histogram Specification")
        self._build_ui()
        self.image = None
        self.header = None
        self.specified_image = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Row 0: input FITS
        grid.addWidget(QLabel("Input FITS:"), 0, 0)
        self.input_edit = QLineEdit()
        grid.addWidget(self.input_edit, 0, 1, 1, 3)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self._browse_input)
        grid.addWidget(btn_in, 0, 4)

        # Row 1: distribution type
        grid.addWidget(QLabel("Specification Type:"), 1, 0)
        self.dist_combo = QComboBox()
        self.dist_combo.addItems(["Rayleigh", "Gaussian", "Uniform", "Exponential", "Lognormal"])
        self.dist_combo.currentIndexChanged.connect(self._update_params_ui)
        grid.addWidget(self.dist_combo, 1, 1)

        # Parameter widgets container
        self.param_widgets = {}
        # Gaussian: mu, sigma
        self.param_widgets['gauss_mu'] = QDoubleSpinBoxWithDefault(0.0)
        self.param_widgets['gauss_sigma'] = QDoubleSpinBoxWithDefault(1.0)
        grid.addWidget(QLabel("mu"), 2, 0); grid.addWidget(self.param_widgets['gauss_mu'], 2, 1)
        grid.addWidget(QLabel("sigma"), 2, 2); grid.addWidget(self.param_widgets['gauss_sigma'], 2, 3)

        # Rayleigh: sigma
        self.param_widgets['ray_sigma'] = QDoubleSpinBoxWithDefault(1.0)
        grid.addWidget(QLabel("sigma (Rayleigh)"), 3, 0); grid.addWidget(self.param_widgets['ray_sigma'], 3, 1)

        # Uniform: lower, upper
        self.param_widgets['uni_lower'] = QDoubleSpinBoxWithDefault(0.0)
        self.param_widgets['uni_upper'] = QDoubleSpinBoxWithDefault(1.0)
        grid.addWidget(QLabel("lower"), 4, 0); grid.addWidget(self.param_widgets['uni_lower'], 4, 1)
        grid.addWidget(QLabel("upper"), 4, 2); grid.addWidget(self.param_widgets['uni_upper'], 4, 3)

        # Exponential: lambda
        self.param_widgets['exp_lambda'] = QDoubleSpinBoxWithDefault(0.1)
        grid.addWidget(QLabel("lambda (exp)"), 5, 0); grid.addWidget(self.param_widgets['exp_lambda'], 5, 1)

        # Lognormal: mu, sigma_ln
        self.param_widgets['log_mu'] = QDoubleSpinBoxWithDefault(0.0)
        self.param_widgets['log_sigma'] = QDoubleSpinBoxWithDefault(0.5)
        grid.addWidget(QLabel("mu (log)"), 6, 0); grid.addWidget(self.param_widgets['log_mu'], 6, 1)
        grid.addWidget(QLabel("sigma (log)"), 6, 2); grid.addWidget(self.param_widgets['log_sigma'], 6, 3)

        # Output file
        grid.addWidget(QLabel("Output FITS:"), 7, 0)
        self.output_edit = QLineEdit("specified_output.fits")
        grid.addWidget(self.output_edit, 7, 1, 1, 3)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_output)
        grid.addWidget(btn_out, 7, 4)

        # Run + info
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 8, 0)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setFixedHeight(60)
        grid.addWidget(self.info, 8, 1, 1, 4)

        # Histogram canvas
        self.canvas = HistCanvas()
        grid.addWidget(self.canvas, 9, 0, 6, 5)

        self._update_params_ui()

    def _browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS", "", "FITS (*.fits *.fit);;All Files (*)")
        if fn:
            self.input_edit.setText(fn)
            try:
                data, hdr = fits.getdata(fn, header=True)
                if data is None:
                    raise ValueError("No data found in primary HDU")
                if data.ndim != 2:
                    raise ValueError("Expected a 2D image in primary HDU")
                self.image = data.astype(np.float64)
                self.header = hdr
                self.info.append(f"Loaded {fn} shape={self.image.shape}")
                # set sensible defaults based on image statistics
                self.param_widgets['gauss_mu'].setValue(np.mean(self.image))
                self.param_widgets['gauss_sigma'].setValue(np.std(self.image))
                self.param_widgets['ray_sigma'].setValue(np.std(self.image))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load FITS: {e}")

    def _browse_output(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save FITS", "", "FITS (*.fits *.fit);;All Files (*)")
        if fn:
            self.output_edit.setText(fn)

    def _update_params_ui(self):
        typ = self.dist_combo.currentText()
        # hide all then show relevant
        for w in self.param_widgets.values():
            w.setVisible(False)
        # layout rows we used; just toggle visibility of the widgets
        if typ == "Gaussian":
            self.param_widgets['gauss_mu'].setVisible(True)
            self.param_widgets['gauss_sigma'].setVisible(True)
        elif typ == "Rayleigh":
            self.param_widgets['ray_sigma'].setVisible(True)
        elif typ == "Uniform":
            self.param_widgets['uni_lower'].setVisible(True)
            self.param_widgets['uni_upper'].setVisible(True)
        elif typ == "Exponential":
            self.param_widgets['exp_lambda'].setVisible(True)
        elif typ == "Lognormal":
            self.param_widgets['log_mu'].setVisible(True)
            self.param_widgets['log_sigma'].setVisible(True)

    def _on_run(self):
        if self.image is None:
            QMessageBox.warning(self, "No image", "Load an input FITS first")
            return
        typ = self.dist_combo.currentText()
        try:
            if typ == "Rayleigh":
                sigma = float(self.param_widgets['ray_sigma'].value())
                specified = rayleigh_specification(self.image, sigma)
                outname = self.output_edit.text().strip()
            elif typ == "Gaussian":
                mu = float(self.param_widgets['gauss_mu'].value())
                sigma = float(self.param_widgets['gauss_sigma'].value())
                specified = gaussian_specification(self.image, mu, sigma)
                outname = self.output_edit.text().strip()
            elif typ == "Uniform":
                lower = float(self.param_widgets['uni_lower'].value())
                upper = float(self.param_widgets['uni_upper'].value())
                if upper <= lower:
                    raise ValueError("upper must be > lower")
                specified = uniform_specification(self.image, lower, upper)
                outname = self.output_edit.text().strip()
            elif typ == "Exponential":
                lamb = float(self.param_widgets['exp_lambda'].value())
                if lamb <= 0:
                    raise ValueError("lambda must be > 0")
                specified = exponential_specification(self.image, lamb)
                outname = self.output_edit.text().strip()
            elif typ == "Lognormal":
                mu = float(self.param_widgets['log_mu'].value())
                sigma_ln = float(self.param_widgets['log_sigma'].value())
                specified = lognormal_specification(self.image, mu, sigma_ln)
                outname = self.output_edit.text().strip()
            else:
                raise RuntimeError("Unknown specification type")
        except Exception as e:
            QMessageBox.critical(self, "Parameter error", str(e))
            return

        # Save specified image
        try:
            if not outname:
                outname = "specified_output.fits"
            fits.writeto(outname, specified.astype(np.float32), header=self.header if self.header is not None else None, overwrite=True)
            self.info.append(f"Saved specified image to {outname}")
            self.specified_image = specified
            # update histograms
            self.canvas.plot_histograms(self.image, specified)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save: {e}")

# small convenience wrapper QDoubleSpinBox with larger range
class QDoubleSpinBoxWithDefault(QDoubleSpinBox):
    def __init__(self, default=0.0):
        super().__init__()
        self.setRange(-1e12, 1e12)
        self.setDecimals(6)
        self.setValue(float(default))

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1000,700)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()