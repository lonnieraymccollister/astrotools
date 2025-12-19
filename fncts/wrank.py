#!/usr/bin/env python3
import sys
from pathlib import Path
import subprocess
import shutil
import numpy as np
from astropy.io import fits
from scipy import ndimage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QSlider, QCheckBox
)
from PyQt6.QtCore import Qt

# ---------------- utilities: footprint and single-channel rank filter ----------------
def circular_footprint(radius):
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    return (xx**2 + yy**2) <= (radius**2)

def _rank_filter_channel(chan, radius=3.0, percentile=50.0):
    chan = chan.astype(np.float64)
    nan_mask = np.isnan(chan)
    if nan_mask.any():
        finite_min = np.nanmin(chan)
        fill_val = finite_min if np.isfinite(finite_min) else 0.0
        work = chan.copy()
        work[nan_mask] = fill_val
    else:
        work = chan

    fp = circular_footprint(radius)
    N = int(fp.sum())
    if N <= 0:
        raise ValueError("Footprint size is zero; check radius value.")
    order = int(round((percentile / 100.0) * (N - 1)))
    order = max(0, min(N - 1, order))
    filt = ndimage.rank_filter(work, rank=order, footprint=fp, mode='mirror')
    if nan_mask.any():
        filt[nan_mask] = np.nan
    return filt

def _blend_channel(orig, filt, weight):
    return weight * filt + (1.0 - weight) * orig

# ---------------- color-capable rank filter (supports grayscale, RGB(A), multi-frame) ----------------
def rank_filter_color_array(arr, radius=3.0, percentile=50.0, weight=0.6, filter_alpha=False):
    """
    Accepts arr shapes:
      - (ny, nx) -> grayscale
      - (ny, nx, nch) with nch in (1,3,4)
      - (nframes, ny, nx, nch) or (nframes, nch, ny, nx)
      - (nch, ny, nx) where nch in (1,3,4)
    Returns filtered array with same layout and dtype=float64.
    """
    if arr is None:
        raise ValueError("Input array is None")

    # 4D frames-first or frames-last
    if arr.ndim == 4:
        # try to detect frames-last (n, ny, nx, nch)
        if arr.shape[-1] in (1, 3, 4):
            out = np.empty_like(arr, dtype=np.float64)
            for i in range(arr.shape[0]):
                out[i] = rank_filter_color_array(arr[i], radius=radius, percentile=percentile, weight=weight, filter_alpha=filter_alpha)
            return out
        # frames-first (nframes, nch, ny, nx)
        if arr.shape[1] in (1, 3, 4):
            nframes = arr.shape[0]
            out = np.empty_like(arr, dtype=np.float64)
            for i in range(nframes):
                frame = np.transpose(arr[i], (1, 2, 0))  # -> (ny, nx, nch)
                out_frame = rank_filter_color_array(frame, radius=radius, percentile=percentile, weight=weight, filter_alpha=filter_alpha)
                out[i] = np.transpose(out_frame, (2, 0, 1))
            return out
        raise ValueError(f"Unsupported 4D input shape: {arr.shape}")

    # normalize to (ny, nx, nch)
    squeeze_out = False
    if arr.ndim == 2:
        arr3 = arr[..., None]
        squeeze_out = True
    elif arr.ndim == 3:
        if arr.shape[2] in (1, 3, 4):
            arr3 = arr
        elif arr.shape[0] in (1, 3, 4):
            arr3 = np.transpose(arr, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported 3D input shape for color filter: {arr.shape}")
    else:
        raise ValueError(f"Unsupported array ndim: {arr.ndim}")

    ny, nx, nch = arr3.shape
    out3 = np.empty((ny, nx, nch), dtype=np.float64)

    alpha_idx = 3 if nch == 4 else None
    for c in range(nch):
        if alpha_idx is not None and c == alpha_idx and not filter_alpha:
            out3[..., c] = arr3[..., c].astype(np.float64)
            continue
        chan = arr3[..., c]
        if np.all(np.isnan(chan)):
            out3[..., c] = chan.astype(np.float64)
            continue
        filt = _rank_filter_channel(chan, radius=radius, percentile=percentile)
        out3[..., c] = _blend_channel(chan.astype(np.float64), filt, weight)

    if squeeze_out:
        return out3[..., 0]
    # if original was (nch, ny, nx) return that shape
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        return np.transpose(out3, (2, 0, 1))
    return out3

# ---------------- FITS IO wrapper that detects color layout ----------------
def rank_filter_fits_color(inpath, outpath, radius=3.0, percentile=50.0, weight=0.6, filter_alpha=False):
    with fits.open(inpath, memmap=False) as hdul:
        # pick first HDU with usable data
        src_hdu = None
        for hdu in hdul:
            if hdu.data is not None:
                src_hdu = hdu
                break
        if src_hdu is None:
            raise ValueError("No data array found in FITS")

        data = src_hdu.data.copy()
        hdr = src_hdu.header.copy()

    filtered = rank_filter_color_array(data, radius=radius, percentile=percentile, weight=weight, filter_alpha=filter_alpha)

    hdr.add_history(f"Color/gray rank filter radius={radius}, pct={percentile}, weight={weight}, filter_alpha={filter_alpha}")
    # write filtered array as primary HDU (preserve header from source_hdu)
    fits.writeto(outpath, filtered.astype(np.float32), header=hdr, overwrite=True)
    return filtered

# ---------------- Normalization and Siril helpers ----------------
def _check_normalized_0_1(arr, tol=1e-8):
    """
    Return True if arr min is approximately 0 and max approximately 1 within tol.
    Works for 2D and 3D arrays.
    """
    if arr is None or arr.size == 0:
        return False
    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))
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

# ---------------- PyQt6 GUI ----------------
class RankFilterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rank Filter (color-capable)")
        self._inpath = ""
        self._outpath = ""
        self._build_ui()

    def _build_ui(self):
        w = QWidget()
        v = QVBoxLayout()

        # input
        row = QHBoxLayout()
        self.in_label = QLineEdit()
        self.in_label.setPlaceholderText("Input FITS file")
        btn_in = QPushButton("Open Input")
        btn_in.clicked.connect(self._choose_input)
        row.addWidget(self.in_label)
        row.addWidget(btn_in)
        v.addLayout(row)

        # output
        row2 = QHBoxLayout()
        self.out_label = QLineEdit()
        self.out_label.setPlaceholderText("Output FITS file")
        btn_out = QPushButton("Save As")
        btn_out.clicked.connect(self._choose_output)
        row2.addWidget(self.out_label)
        row2.addWidget(btn_out)
        v.addLayout(row2)

        # normalization and Siril checkboxes (global)
        chk_row = QHBoxLayout()
        self.checkInputNormalized = QCheckBox("Check input normalized to [0,1]")
        self.checkInputNormalized.setChecked(True)
        chk_row.addWidget(self.checkInputNormalized)
        self.checkOpenInSiril = QCheckBox("Offer to open result in Siril after save")
        self.checkOpenInSiril.setChecked(True)
        chk_row.addWidget(self.checkOpenInSiril)
        chk_row.addStretch()
        v.addLayout(chk_row)

        # radius
        rrow = QHBoxLayout()
        rrow.addWidget(QLabel("Radius (px):"))
        self.radius_edit = QLineEdit("3.0")
        rrow.addWidget(self.radius_edit)
        v.addLayout(rrow)

        # percentile
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Percentile (0-100):"))
        self.percentile_edit = QLineEdit("50")
        prow.addWidget(self.percentile_edit)
        v.addLayout(prow)

        # weight slider + edit
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("Blend weight (filtered):"))
        self.weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.weight_slider.setRange(0, 100)
        self.weight_slider.setValue(60)
        self.weight_slider.valueChanged.connect(self._sync_weight_edit)
        self.weight_edit = QLineEdit("0.60")
        self.weight_edit.editingFinished.connect(self._sync_weight_slider)
        wrow.addWidget(self.weight_slider)
        wrow.addWidget(self.weight_edit)
        v.addLayout(wrow)

        # filter alpha checkbox (text-only for simplicity)
        arow = QHBoxLayout()
        self.filter_alpha_edit = QLineEdit("False")  # user may type True/False
        arow.addWidget(QLabel("Filter alpha channel (True/False):"))
        arow.addWidget(self.filter_alpha_edit)
        v.addLayout(arow)

        # run button (instance attr so we can disable)
        self.btn_run = QPushButton("Run Rank Filter")
        self.btn_run.clicked.connect(self._run_filter)
        v.addWidget(self.btn_run)

        w.setLayout(v)
        self.setCentralWidget(w)

        # initialize weight edit text
        self._sync_weight_edit()

    def _choose_input(self):
        start = str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "Select input FITS", start, "FITS Files (*.fits);;All Files (*)")
        if path:
            self._inpath = path
            self.in_label.setText(path)
            # quick detection of color-like data
            try:
                with fits.open(path, memmap=False) as hdul:
                    data = hdul[0].data
                    if data is not None:
                        if data.ndim == 3 and data.shape[-1] in (3, 4):
                            QMessageBox.information(self, "Detected color", "Input appears to be a color image (ny, nx, 3/4).")
                        elif data.ndim == 3 and data.shape[0] in (3, 4):
                            QMessageBox.information(self, "Detected color", "Input appears to be a color image (3/4, ny, nx).")
            except Exception:
                pass

    def _choose_output(self):
        start = str(Path.cwd())
        path, _ = QFileDialog.getSaveFileName(self, "Save filtered FITS as", start, "FITS Files (*.fits)")
        if path:
            self._outpath = path
            self.out_label.setText(path)

    def _sync_weight_edit(self):
        val = self.weight_slider.value() / 100.0
        self.weight_edit.setText(f"{val:.2f}")

    def _sync_weight_slider(self):
        try:
            v = float(self.weight_edit.text())
            v = max(0.0, min(1.0, v))
            self.weight_slider.setValue(int(round(v * 100)))
        except Exception:
            self.weight_edit.setText(f"{self.weight_slider.value() / 100.0:.2f}")

    def _run_filter(self):
        if not self._inpath:
            QMessageBox.warning(self, "Missing input", "Please select an input FITS file.")
            return
        if not self._outpath:
            QMessageBox.warning(self, "Missing output", "Please select an output path.")
            return

        try:
            radius = float(self.radius_edit.text())
            percentile = float(self.percentile_edit.text())
            weight = float(self.weight_edit.text())
            if radius <= 0:
                raise ValueError("Radius must be > 0")
            if not (0.0 <= percentile <= 100.0):
                raise ValueError("Percentile must be 0..100")
            if not (0.0 <= weight <= 1.0):
                raise ValueError("Weight must be 0..1")
            filter_alpha = str(self.filter_alpha_edit.text()).strip().lower() in ("1","true","t","yes","y")
        except Exception as e:
            QMessageBox.critical(self, "Parameter error", str(e))
            return

        # Normalization check (global)
        if self.checkInputNormalized.isChecked():
            try:
                with fits.open(self._inpath, memmap=False) as hdul:
                    data = None
                    for hdu in hdul:
                        if hdu.data is not None:
                            data = hdu.data
                            break
                    if data is None:
                        QMessageBox.critical(self, "Normalization check", "Input FITS contains no data.")
                        return
                    if not _check_normalized_0_1(np.array(data)):
                        launched = _launch_normalize_gui(self._inpath)
                        if launched:
                            QMessageBox.information(self, "Normalization required", "Input not normalized to [0,1]. Launched normalize_gui.py for the input file. Please normalize and re-run.")
                        else:
                            QMessageBox.information(self, "Normalization required", "Input not normalized to [0,1] and normalize_gui.py was not found.")
                        return
            except Exception as e:
                QMessageBox.critical(self, "Normalization check error", str(e))
                return

        self.btn_run.setEnabled(False)
        try:
            rank_filter_fits_color(self._inpath, self._outpath, radius=radius, percentile=percentile, weight=weight, filter_alpha=filter_alpha)
            # After successful write, optionally launch Siril
            siril_msg = ""
            if self.checkOpenInSiril.isChecked():
                launched, siril_msg = _maybe_launch_siril(self._outpath)
            info_msg = f"Filtered FITS written to:\n{self._outpath}"
            if siril_msg:
                info_msg += f"\n\n{siril_msg}"
            QMessageBox.information(self, "Done", info_msg)
        except Exception as e:
            QMessageBox.critical(self, "Processing error", str(e))
        finally:
            self.btn_run.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RankFilterWindow()
    win.show()
    sys.exit(app.exec())