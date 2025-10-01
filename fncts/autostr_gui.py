#!/usr/bin/env python3
"""
autostr_gui.py
PyQt6 GUI to perform SMH (shadows/midtones/highlights) automatic stretch on a FITS image.
Saves a stretched FITS and records parameters in header.
"""
import sys
import traceback
import numpy as np
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QDoubleSpinBox, QTextEdit, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------- stretch logic ----------
def smh_stretch(data, lower_percent=0.5, upper_percent=99.5):
    """
    Perform SMH-style histogram stretch on numeric array.
    Returns (stretched in [0,1], low, med, high, gamma).
    """
    arr = np.asarray(data)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("No finite pixels in input data")

    # percentiles (use numpy.percentile; method argument removed for compatibility)
    low = np.percentile(arr[finite], lower_percent)
    high = np.percentile(arr[finite], upper_percent)
    med = np.percentile(arr[finite], 50.0)
    med = np.clip(med, low, high)

    if high == low:
        gamma = 1.0
        stretched = np.clip((arr - low), 0, 1) * 0.0
        return stretched, low, med, high, gamma

    m = (med - low) / (high - low)
    if m <= 0:
        m = 0.001
    gamma = np.log(0.5) / np.log(m)

    normalized = (arr - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)
    stretched = normalized ** gamma

    return stretched, low, med, high, gamma

# ---------- matplotlib canvas ----------
class HistCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=2.6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.ax = fig.add_subplot(1,1,1)
        fig.tight_layout()

    def plot_hist(self, data, bins=256, title="Histogram"):
        self.ax.clear()
        if data is None:
            self.ax.set_title("No data")
            self.draw()
            return
        a = np.asarray(data).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            self.ax.set_title("No finite data")
            self.draw()
            return
        self.ax.hist(a, bins=bins, color='C0', histtype='stepfilled', alpha=0.6)
        self.ax.set_title(title)
        self.draw()

# ---------- GUI ----------
class AutoStrWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto SMH Stretch")
        self._build_ui()
        self.resize(920, 520)
        self.data = None
        self.header = None
        self.input_path = None

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        grid.addWidget(QLabel("Input FITS:"), 0, 0)
        self.input_edit = QLineEdit()
        grid.addWidget(self.input_edit, 0, 1, 1, 3)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self._browse_input)
        grid.addWidget(btn_in, 0, 4)

        grid.addWidget(QLabel("Output filename (optional):"), 1, 0)
        self.output_edit = QLineEdit("")
        self.output_edit.setPlaceholderText("leave empty to auto add _smh.fit")
        grid.addWidget(self.output_edit, 1, 1, 1, 3)
        btn_out = QPushButton("Browse Save Location")
        btn_out.clicked.connect(self._browse_save)
        grid.addWidget(btn_out, 1, 4)

        grid.addWidget(QLabel("Lower percentile (%):"), 2, 0)
        self.lower_spin = QDoubleSpinBox()
        self.lower_spin.setDecimals(6)
        self.lower_spin.setRange(0.0, 50.0)
        self.lower_spin.setSingleStep(0.1)
        self.lower_spin.setValue(0.5)
        grid.addWidget(self.lower_spin, 2, 1)

        grid.addWidget(QLabel("Upper percentile (%):"), 2, 2)
        self.upper_spin = QDoubleSpinBox()
        self.upper_spin.setDecimals(6)
        self.upper_spin.setRange(50.0, 100.0)
        self.upper_spin.setSingleStep(0.1)
        self.upper_spin.setValue(99.999)
        grid.addWidget(self.upper_spin, 2, 3)

        self.keep_header_chk = QCheckBox("Preserve original header (add STRETCH keys)")
        self.keep_header_chk.setChecked(True)
        grid.addWidget(self.keep_header_chk, 3, 0, 1, 3)

        self.preview_btn = QPushButton("Load & Preview Histogram")
        self.preview_btn.clicked.connect(self._load_and_preview)
        grid.addWidget(self.preview_btn, 3, 3)

        self.run_btn = QPushButton("Run Stretch & Save")
        self.run_btn.clicked.connect(self._run_stretch)
        grid.addWidget(self.run_btn, 4, 0)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(lambda: self.log.clear())
        grid.addWidget(self.clear_btn, 4, 1)

        # Results text
        grid.addWidget(QLabel("Computed parameters:"), 5, 0)
        self.params_box = QTextEdit()
        self.params_box.setReadOnly(True)
        self.params_box.setFixedHeight(120)
        grid.addWidget(self.params_box, 6, 0, 1, 5)

        # Histogram canvas
        self.canvas = HistCanvas(width=8, height=3)
        grid.addWidget(self.canvas, 7, 0, 4, 5)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(80)
        grid.addWidget(self.log, 11, 0, 1, 5)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    def _browse_input(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open FITS", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            self.input_edit.setText(fn)

    def _browse_save(self):
        dn = QFileDialog.getExistingDirectory(self, "Select folder to save output", "")
        if dn:
            base = self.output_edit.text().strip() or ""
            if base and (base.endswith(".fit") or base.endswith(".fits")):
                self.output_edit.setText(base)
            else:
                self.output_edit.setText(dn + "/" + (base if base else "output_smh"))

    def _load_and_preview(self):
        path = self.input_edit.text().strip()
        if not path:
            QMessageBox.information(self, "Input required", "Select an input FITS first")
            return
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data
                header = hdul[0].header
            if data is None:
                raise ValueError("No data in primary HDU")
            self.input_path = path
            self.data = np.asarray(data).astype(np.float64)
            self.header = header
            self._log(f"Loaded {path} shape={self.data.shape} dtype={self.data.dtype}")
            self.canvas.plot_hist(self.data, bins=256, title="Input histogram")
            self.params_box.setPlainText("") 
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"{e}\n\n{tb}")

    def _run_stretch(self):
        if self.data is None:
            self._load_and_preview()
            if self.data is None:
                return
        try:
            lower = float(self.lower_spin.value())
            upper = float(self.upper_spin.value())
            if not (0.0 <= lower < upper <= 100.0):
                raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

            stretched, low, med, high, gamma = smh_stretch(self.data, lower, upper)
            # Prepare header
            out_header = self.header.copy() if (self.header is not None and self.keep_header_chk.isChecked()) else fits.Header()
            out_header['STRETCH'] = ('SMH', 'Shadows/Midtones/Highlights histogram stretch')
            out_header['LOWPCT'] = (lower, 'Lower percentile for shadow threshold')
            out_header['HIGPCT'] = (upper, 'Upper percentile for highlight threshold')
            out_header['SHADOW'] = (float(low), 'Shadow threshold value')
            out_header['MIDTONE'] = (float(med), 'Midtone (median) value')
            out_header['HIGHLT'] = (float(high), 'Highlight threshold value')
            out_header['GAMMA'] = (float(gamma), 'Gamma value used')

            outpath = self.output_edit.text().strip()
            if not outpath:
                # build from input path
                if self.input_path:
                    base = self.input_path
                    if base.lower().endswith(".fits") or base.lower().endswith(".fit"):
                        outpath = base.rsplit(".",1)[0] + "_smh.fit"
                    else:
                        outpath = base + "_smh.fit"
                else:
                    outpath = "output_smh.fit"

            # Save stretched data as float32 for compactness while preserving dynamic range [0,1]
            fits.writeto(outpath, stretched.astype(np.float32), header=out_header, overwrite=True)
            self._log(f"Wrote stretched FITS: {outpath}")
            # update UI with parameters and histogram of stretched result
            params = [
                f"lower_percent: {lower}",
                f"upper_percent: {upper}",
                f"shadow (low): {low}",
                f"midtone (med): {med}",
                f"highlight (high): {high}",
                f"gamma: {gamma}"
            ]
            self.params_box.setPlainText("\n".join(params))
            self.canvas.plot_hist(stretched, bins=256, title="Stretched histogram")
            QMessageBox.information(self, "Done", f"Wrote stretched FITS: {outpath}")
        except Exception as e:
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")

def main():
    app = QApplication(sys.argv)
    w = AutoStrWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()