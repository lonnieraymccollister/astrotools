#!/usr/bin/env python3
"""
autostr_gui.py
PyQt6 GUI to perform SMH (shadows/midtones/highlights) automatic stretch on a FITS image.
Saves a stretched FITS and records parameters in header.

Enhancements:
 - Add "Clip extremes" option similar to Siril's histogram transform clip button.
 - UI: checkbox to enable clipping and two spinboxes to set clip low/high percentiles.
 - Logging of clipping behavior and values.
 - Added two global checkboxes:
     * "Check input normalized to [0,1]" (checked by default). If enabled and the input FITS
       primary data min/max are not approximately 0 and 1, the GUI will launch normalize_gui.py
       with the input filename and abort the current operation so the user can normalize the file.
     * "Offer to open result in Siril after save" (checked by default). If enabled, the GUI will
       attempt to launch the `siril` executable with the saved output path after a successful save.
"""
import sys
import traceback
import subprocess
import shutil
from pathlib import Path

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

# ---------- normalization and Siril helpers (new) ----------
def _check_normalized_0_1(arr, tol=1e-8):
    """
    Return True if arr min is approximately 0 and max approximately 1 within tol.
    Works for 2D and 3D arrays.
    """
    if arr is None:
        return False
    a = np.array(arr, dtype=np.float64)
    if a.size == 0:
        return False
    amin = float(np.nanmin(a))
    amax = float(np.nanmax(a))
    if np.isnan(amin) or np.isnan(amax):
        return False
    return (abs(amin - 0.0) <= tol) and (abs(amax - 1.0) <= tol)

def _launch_normalize_gui(filepath):
    """
    Launch normalize_gui.py with the given filepath using the same Python interpreter.
    Non-blocking. Returns True if launch succeeded, False otherwise.
    """
    try:
        script = Path("normalize_gui.py")
        if not script.exists():
            script = Path(__file__).resolve().parent / "normalize_gui.py"
        if not script.exists():
            return False
        subprocess.Popen([sys.executable, str(script), str(filepath)])
        return True
    except Exception:
        return False

def _maybe_launch_siril(output_path):
    """
    Try to launch Siril with the output file. Returns (launched: bool, message: str).
    """
    try:
        siril_exe = shutil.which("siril")
        if siril_exe:
            subprocess.Popen([siril_exe, str(Path(output_path).resolve())])
            return True, "Siril launched."
        else:
            return False, "Siril executable not found in PATH."
    except Exception as e:
        return False, f"Failed to launch Siril: {e}"

# ---------- stretch logic ----------
def smh_stretch(data, lower_percent=0.5, upper_percent=99.5, mid_override=None,
                clip_enabled=False, clip_low_pct=0.0, clip_high_pct=0.0):
    """
    Perform SMH-style histogram stretch on numeric array.
    If mid_override is provided (numeric), use that as the midtone value (clipped to [low, high]).
    If clip_enabled is True, pixels below the clip_low_pct percentile and above the
    (100 - clip_high_pct) percentile are excluded from percentile and median calculations.
    Returns (stretched in [0,1], low, med, high, gamma, used_mask_info)
    where used_mask_info is a dict with clipping details for logging.
    """
    arr = np.asarray(data)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        raise ValueError("No finite pixels in input data")

    # Build mask of pixels to use for percentile calculations
    use_mask = finite_mask.copy()

    clip_info = {"clip_enabled": bool(clip_enabled), "clip_low_pct": float(clip_low_pct),
                 "clip_high_pct": float(clip_high_pct), "n_total": int(np.count_nonzero(finite_mask)),
                 "n_used": None}

    if clip_enabled:
        # compute clip thresholds as percentiles on finite pixels
        low_clip_val = np.percentile(arr[finite_mask], clip_low_pct) if clip_low_pct > 0 else -np.inf
        high_clip_val = np.percentile(arr[finite_mask], 100.0 - clip_high_pct) if clip_high_pct > 0 else np.inf
        # update mask to exclude clipped extremes
        use_mask &= (arr >= low_clip_val) & (arr <= high_clip_val)
        clip_info.update({"low_clip_val": float(low_clip_val), "high_clip_val": float(high_clip_val)})
        if not use_mask.any():
            # if clipping removed all pixels, fall back to finite_mask
            use_mask = finite_mask.copy()
            clip_info["warning"] = "Clipping removed all finite pixels; using all finite pixels instead."
    clip_info["n_used"] = int(np.count_nonzero(use_mask))

    # percentiles computed on the masked set
    low = np.percentile(arr[use_mask], lower_percent)
    high = np.percentile(arr[use_mask], upper_percent)

    # handle degenerate case
    if high == low:
        gamma = 1.0
        stretched = np.zeros_like(arr, dtype=np.float64)
        med = low
        return stretched, low, med, high, gamma, clip_info

    # choose median or override (median computed on masked set)
    if mid_override is None:
        med = np.percentile(arr[use_mask], 50.0)
    else:
        med = float(mid_override)

    # clip midtone to [low, high]
    med = np.clip(med, low, high)

    # compute gamma robustly
    m = (med - low) / (high - low)
    if m <= 0:
        m = 0.001
    try:
        gamma = np.log(0.5) / np.log(m)
    except Exception:
        gamma = 1.0

    normalized = (arr - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)
    stretched = normalized ** gamma

    return stretched, low, med, high, gamma, clip_info

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

        # Top row: input and global checkboxes
        grid.addWidget(QLabel("Input FITS:"), 0, 0)
        self.input_edit = QLineEdit()
        grid.addWidget(self.input_edit, 0, 1, 1, 3)
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self._browse_input)
        grid.addWidget(btn_in, 0, 4)

        # Global checkboxes: normalization check and Siril launch
        self.check_normalized = QCheckBox("Check input normalized to [0,1]")
        self.check_normalized.setChecked(True)
        grid.addWidget(self.check_normalized, 1, 1, 1, 2)

        self.check_open_siril = QCheckBox("Offer to open result in Siril after save")
        self.check_open_siril.setChecked(True)
        grid.addWidget(self.check_open_siril, 1, 3, 1, 2)

        grid.addWidget(QLabel("Output filename (optional):"), 2, 0)
        self.output_edit = QLineEdit("")
        self.output_edit.setPlaceholderText("leave empty to auto add _smh.fit")
        grid.addWidget(self.output_edit, 2, 1, 1, 3)
        btn_out = QPushButton("Browse Save Location")
        btn_out.clicked.connect(self._browse_save)
        grid.addWidget(btn_out, 2, 4)

        grid.addWidget(QLabel("Lower percentile (%):"), 3, 0)
        self.lower_spin = QDoubleSpinBox()
        self.lower_spin.setDecimals(6)
        self.lower_spin.setRange(0.0, 50.0)
        self.lower_spin.setSingleStep(0.1)
        self.lower_spin.setValue(0.5)
        grid.addWidget(self.lower_spin, 3, 1)

        grid.addWidget(QLabel("Upper percentile (%):"), 3, 2)
        self.upper_spin = QDoubleSpinBox()
        self.upper_spin.setDecimals(6)
        self.upper_spin.setRange(50.0, 100.0)
        self.upper_spin.setSingleStep(0.1)
        self.upper_spin.setValue(99.999)
        grid.addWidget(self.upper_spin, 3, 3)

        # New: custom midtone checkbox and entry
        self.custom_mid_chk = QCheckBox("Use custom midtone value")
        self.custom_mid_chk.setChecked(False)
        grid.addWidget(self.custom_mid_chk, 4, 0, 1, 2)

        self.mid_spin = QDoubleSpinBox()
        self.mid_spin.setDecimals(6)
        # Allow a wide range; user should enter a value in the data units
        self.mid_spin.setRange(-1e12, 1e12)
        self.mid_spin.setSingleStep(0.1)
        self.mid_spin.setValue(0.0)
        self.mid_spin.setEnabled(False)
        grid.addWidget(QLabel("Midtone value:"), 4, 2)
        grid.addWidget(self.mid_spin, 4, 3)

        # Use toggled which passes a boolean and is less error prone
        self.custom_mid_chk.toggled.connect(self.mid_spin.setEnabled)

        self.keep_header_chk = QCheckBox("Preserve original header (add STRETCH keys)")
        self.keep_header_chk.setChecked(True)
        grid.addWidget(self.keep_header_chk, 5, 0, 1, 3)

        # New: clipping controls (Siril-like clip button behavior)
        self.clip_chk = QCheckBox("Enable clipping of histogram extremes (affects parameter estimation)")
        self.clip_chk.setChecked(False)
        grid.addWidget(self.clip_chk, 6, 0, 1, 4)

        grid.addWidget(QLabel("Clip low percentile (%):"), 7, 0)
        self.clip_low_spin = QDoubleSpinBox()
        self.clip_low_spin.setDecimals(6)
        self.clip_low_spin.setRange(0.0, 49.0)
        self.clip_low_spin.setSingleStep(0.1)
        self.clip_low_spin.setValue(0.0)
        grid.addWidget(self.clip_low_spin, 7, 1)

        grid.addWidget(QLabel("Clip high percentile (%):"), 7, 2)
        self.clip_high_spin = QDoubleSpinBox()
        self.clip_high_spin.setDecimals(6)
        self.clip_high_spin.setRange(0.0, 49.0)
        self.clip_high_spin.setSingleStep(0.1)
        self.clip_high_spin.setValue(0.0)
        grid.addWidget(self.clip_high_spin, 7, 3)

        self.preview_btn = QPushButton("Load & Preview Histogram")
        self.preview_btn.clicked.connect(self._load_and_preview)
        grid.addWidget(self.preview_btn, 8, 3)

        self.run_btn = QPushButton("Run Stretch & Save")
        self.run_btn.clicked.connect(self._run_stretch)
        grid.addWidget(self.run_btn, 9, 0)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(lambda: self.log.clear())
        grid.addWidget(self.clear_btn, 9, 1)

        # Results text
        grid.addWidget(QLabel("Computed parameters:"), 10, 0)
        self.params_box = QTextEdit()
        self.params_box.setReadOnly(True)
        self.params_box.setFixedHeight(120)
        grid.addWidget(self.params_box, 11, 0, 1, 5)

        # Histogram canvas
        self.canvas = HistCanvas(width=8, height=3)
        grid.addWidget(self.canvas, 12, 0, 4, 5)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(80)
        grid.addWidget(self.log, 16, 0, 1, 5)

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
                try:
                    self.mid_spin.setValue(med)
                except Exception:
                    pass

            # If clipping is enabled, show clipped histogram overlay in the log
            if self.clip_chk.isChecked():
                clip_low = float(self.clip_low_spin.value())
                clip_high = float(self.clip_high_spin.value())
                # compute clip thresholds for logging
                finite = finite_vals
                if finite.size > 0:
                    low_clip_val = np.percentile(finite, clip_low) if clip_low > 0 else -np.inf
                    high_clip_val = np.percentile(finite, 100.0 - clip_high) if clip_high > 0 else np.inf
                    self._log(f"Clipping enabled: excluding values < {low_clip_val} and > {high_clip_val} for parameter estimation")
                else:
                    self._log("Clipping enabled but no finite pixels found")

            self.canvas.plot_hist(self.data, bins=256, title="Input histogram")
            self.params_box.setPlainText("")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Load error", f"{e}\n\n{tb}")

    def _run_stretch(self):
        # If no data loaded, attempt to load
        if self.data is None:
            self._load_and_preview()
            if self.data is None:
                return

        # If normalization check is enabled, verify input primary data is 0..1
        if self.check_normalized.isChecked():
            try:
                with fits.open(self.input_edit.text().strip()) as hdul:
                    raw = hdul[0].data
                if raw is None:
                    QMessageBox.critical(self, "Normalization check", "Input FITS contains no data.")
                    return
                arr_check = np.array(raw, dtype=np.float64)
                if not _check_normalized_0_1(arr_check):
                    launched = _launch_normalize_gui(self.input_edit.text().strip())
                    if launched:
                        QMessageBox.information(self, "Normalization required", "Input not normalized to [0,1]. Launched normalize_gui.py for the input file. Please normalize and re-run.")
                    else:
                        QMessageBox.information(self, "Normalization required", "Input not normalized to [0,1] and normalize_gui.py was not found.")
                    return
            except Exception as e:
                QMessageBox.critical(self, "Normalization check error", str(e))
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

            # clipping parameters
            clip_enabled = bool(self.clip_chk.isChecked())
            clip_low_pct = float(self.clip_low_spin.value())
            clip_high_pct = float(self.clip_high_spin.value())
            if clip_low_pct < 0 or clip_high_pct < 0 or (clip_low_pct + clip_high_pct) >= 100.0:
                raise ValueError("Invalid clipping percentiles; ensure non-negative and sum < 100")

            stretched, low, med, high, gamma, clip_info = smh_stretch(
                self.data,
                lower_percent=lower,
                upper_percent=upper,
                mid_override=mid_override,
                clip_enabled=clip_enabled,
                clip_low_pct=clip_low_pct,
                clip_high_pct=clip_high_pct
            )

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
            out_header['MIDOVR'] = (1 if custom_mid_used else 0, 'Custom midtone used?')
            if custom_mid_used:
                safe_midval = _safe_float_for_header(mid_override)
                out_header['MIDVAL'] = (safe_midval, 'User-specified midtone value')

            # record clipping parameters in header
            out_header['CLIPEN'] = (1 if clip_enabled else 0, 'Clipping enabled for parameter estimation')
            out_header['CLIPLO'] = (float(clip_low_pct), 'Clip low percentile used')
            out_header['CLIPHI'] = (float(clip_high_pct), 'Clip high percentile used')
            # if clip_info contains numeric thresholds, store them too
            if clip_info.get("clip_enabled", False):
                if "low_clip_val" in clip_info:
                    out_header['CLIPVLO'] = (_safe_float_for_header(clip_info["low_clip_val"]), 'Clip low value')
                if "high_clip_val" in clip_info:
                    out_header['CLIPVHI'] = (_safe_float_for_header(clip_info["high_clip_val"]), 'Clip high value')
                out_header['CLIPNIN'] = (clip_info.get("n_used", 0), 'Number of pixels used after clipping')

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
            if clip_enabled:
                params.append(f"clipping enabled: low_pct={clip_low_pct}, high_pct={clip_high_pct}")
                if "low_clip_val" in clip_info and "high_clip_val" in clip_info:
                    params.append(f"clip values: low={clip_info['low_clip_val']}, high={clip_info['high_clip_val']}")
                params.append(f"pixels used for estimation: {clip_info.get('n_used')} / {clip_info.get('n_total')}")
                if "warning" in clip_info:
                    params.append(f"clip warning: {clip_info['warning']}")

            self.params_box.setPlainText("\n".join(params))
            self.canvas.plot_hist(stretched, bins=256, title="Stretched histogram")

            # Optionally launch Siril
            if self.check_open_siril.isChecked():
                launched, siril_msg = _maybe_launch_siril(outpath)
                if launched:
                    QMessageBox.information(self, "Done", f"Wrote stretched FITS: {outpath}\n\nSiril launched.")
                else:
                    QMessageBox.information(self, "Done", f"Wrote stretched FITS: {outpath}\n\n{siril_msg}")
            else:
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