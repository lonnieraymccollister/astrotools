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

# ---------- utility ----------
def _safe_float_for_header(x, fallback=0.0):
    """
    Convert x to a Python float and ensure it is finite.
    If not finite or conversion fails, return fallback.
    """
    try:
        xf = float(x)
    except Exception:
        return float(fallback)
    if not np.isfinite(xf):
        return float(fallback)
    return xf

# ---------- stretch logic ----------
def smh_stretch(data, lower_percent=0.5, upper_percent=99.5, mid_override=None):
    """
    Perform SMH-style histogram stretch on numeric array.
    If mid_override is provided (numeric), use that as the midtone value (clipped to [low, high]).
    Returns (stretched in [0,1], low, med, high, gamma).
    """
    arr = np.asarray(data)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        raise ValueError("No finite pixels in input data")

    # percentiles
    low = np.percentile(arr[finite_mask], lower_percent)
    high = np.percentile(arr[finite_mask], upper_percent)

    # handle degenerate case
    if high == low:
        gamma = 1.0
        # produce zeros array in [0,1]
        stretched = np.zeros_like(arr, dtype=np.float64)
        med = low
        return stretched, low, med, high, gamma

    # choose median or override
    if mid_override is None:
        med = np.percentile(arr[finite_mask], 50.0)
    else:
        med = float(mid_override)

    # clip midtone to [low, high]
    med = np.clip(med, low, high)

    # compute gamma robustly
    m = (med - low) / (high - low)
    if m <= 0:
        m = 0.001
    # guard against log domain errors
    try:
        gamma = np.log(0.5) / np.log(m)
    except Exception:
        gamma = 1.0

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

        # New: custom midtone checkbox and entry
        self.custom_mid_chk = QCheckBox("Use custom midtone value")
        self.custom_mid_chk.setChecked(False)
        grid.addWidget(self.custom_mid_chk, 3, 0, 1, 2)

        self.mid_spin = QDoubleSpinBox()
        self.mid_spin.setDecimals(6)
        # Allow a wide range; user should enter a value in the data units
        self.mid_spin.setRange(-1e12, 1e12)
        self.mid_spin.setSingleStep(0.1)
        self.mid_spin.setValue(0.0)
        self.mid_spin.setEnabled(False)
        grid.addWidget(QLabel("Midtone value:"), 3, 2)
        grid.addWidget(self.mid_spin, 3, 3)

        # Use toggled which passes a boolean and is less error prone
        self.custom_mid_chk.toggled.connect(self.mid_spin.setEnabled)

        self.keep_header_chk = QCheckBox("Preserve original header (add STRETCH keys)")
        self.keep_header_chk.setChecked(True)
        grid.addWidget(self.keep_header_chk, 4, 0, 1, 3)

        self.preview_btn = QPushButton("Load & Preview Histogram")
        self.preview_btn.clicked.connect(self._load_and_preview)
        grid.addWidget(self.preview_btn, 4, 3)

        self.run_btn = QPushButton("Run Stretch & Save")
        self.run_btn.clicked.connect(self._run_stretch)
        grid.addWidget(self.run_btn, 5, 0)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(lambda: self.log.clear())
        grid.addWidget(self.clear_btn, 5, 1)

        # Results text
        grid.addWidget(QLabel("Computed parameters:"), 6, 0)
        self.params_box = QTextEdit()
        self.params_box.setReadOnly(True)
        self.params_box.setFixedHeight(120)
        grid.addWidget(self.params_box, 7, 0, 1, 5)

        # Histogram canvas
        self.canvas = HistCanvas(width=8, height=3)
        grid.addWidget(self.canvas, 8, 0, 4, 5)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(80)
        grid.addWidget(self.log, 12, 0, 1, 5)

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

            # compute median of finite pixels and populate mid_spin so user sees a sensible default
            finite_vals = self.data[np.isfinite(self.data)]
            if finite_vals.size > 0:
                med = float(np.median(finite_vals))
                # set mid_spin value but do not enable it unless checkbox is checked
                try:
                    self.mid_spin.setValue(med)
                except Exception:
                    # ignore if value out of spinbox range
                    pass

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

            mid_override = None
            custom_mid_used = False
            if self.custom_mid_chk.isChecked():
                mid_override = float(self.mid_spin.value())
                if not np.isfinite(mid_override):
                    raise ValueError("Midtone must be a finite number")
                custom_mid_used = True

            stretched, low, med, high, gamma = smh_stretch(self.data, lower, upper, mid_override=mid_override)

            # sanitize header values
            safe_low = _safe_float_for_header(low)
            safe_med = _safe_float_for_header(med)
            safe_high = _safe_float_for_header(high)
            safe_gamma = _safe_float_for_header(gamma)
            safe_lower_pct = _safe_float_for_header(lower)
            safe_upper_pct = _safe_float_for_header(upper)

            out_header = self.header.copy() if (self.header is not None and self.keep_header_chk.isChecked()) else fits.Header()
            out_header['STRETCH'] = ('SMH', 'Shadows/Midtones/Highlights histogram stretch')
            out_header['LOWPCT']  = (safe_lower_pct, 'Lower percentile for shadow threshold')
            out_header['HIGPCT']  = (safe_upper_pct, 'Upper percentile for highlight threshold')
            out_header['SHADOW']  = (safe_low, 'Shadow threshold value')
            out_header['MIDTONE'] = (safe_med, 'Midtone (median or override) value')
            out_header['HIGHLT']  = (safe_high, 'Highlight threshold value')
            out_header['GAMMA']   = (safe_gamma, 'Gamma value used')
            # store boolean as integer 0/1
            out_header['MIDOVR'] = (1 if custom_mid_used else 0, 'Custom midtone used?')
            if custom_mid_used:
                safe_midval = _safe_float_for_header(mid_override)
                out_header['MIDVAL'] = (safe_midval, 'User-specified midtone value')

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
                f"gamma: {gamma}",
                f"custom_mid_used: {custom_mid_used}"
            ]
            if custom_mid_used:
                params.append(f"custom_mid_value: {mid_override}")
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