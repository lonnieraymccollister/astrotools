#!/usr/bin/env python3
"""
image_filters.py

PyQt6 GUI to run a set of FITS image filters (Unsharp, RL deconv, FFT high-pass, morphology, Gaussian, HpMore, LocAdapt).

Enhancement:
 - Robust handling of 3-plane RGB FITS preferred:
     * Accepts and normalizes input layouts (primary HDU data in either (3, Y, X) or (Y, X, 3) or (Y, X)).
     * Returns/uses arrays in canonical (Y, X, C) form for processing.
     * Preserves original header and original channel layout on write (will restore (3, Y, X) if input was channel-first).
 - Safer numeric casts and clearer error messages.
 - Added "moderate sharpening" checkbox to High Pass (HpMore) filter page.
 - Added global checkboxes:
     * Check input normalized to [0,1] (checked by default) — launches normalize_gui.py if not normalized.
     * Offer to open result in Siril after save (checked by default).
"""
import sys
import os
import subprocess
import shutil
from pathlib import Path

import numpy as np
import cv2
from astropy.io import fits
from astropy.nddata import Cutout2D

from scipy.signal import convolve2d
from scipy.ndimage import convolve as nd_convolve

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QComboBox,
    QStackedWidget, QCheckBox
)
from PyQt6.QtCore import Qt

# ----------------------------
# Helpers for FITS color handling
# ----------------------------
def load_fits_rgb(path):
    """
    Load primary HDU and return tuple (arr_yxc, header, layout_info)
    - arr_yxc is a numpy array in shape (Y, X) or (Y, X, 3)
    - header is the primary header
    - layout_info is dict with 'orig_shape' and 'channel_first' boolean indicating if original was (3, Y, X)
    """
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header

    if data is None:
        raise ValueError("FITS primary HDU has no data")

    arr = np.array(data)  # keep original shape
    layout_info = {"orig_shape": arr.shape, "channel_first": False}

    # Normalize to (Y, X, C) if needed
    if arr.ndim == 3:
        # common case: (3, Y, X) -> transpose to (Y, X, 3)
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
            layout_info["channel_first"] = True
        # already (Y, X, 3)
        elif arr.shape[2] == 3:
            layout_info["channel_first"] = False
        else:
            # other 3D layouts are unexpected; try to handle if first dim == 3
            if arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
                layout_info["channel_first"] = True
            else:
                raise ValueError(f"Unsupported 3D FITS shape: {arr.shape}")
    elif arr.ndim == 2:
        # single plane -> expand to (Y, X, 3) by duplication for convenience
        arr = np.stack([arr, arr, arr], axis=-1)
        layout_info["channel_first"] = False
    else:
        raise ValueError(f"Unsupported FITS dimensionality: {arr.shape}")

    # Ensure float64 for processing
    arr = arr.astype(np.float64, copy=False)
    return arr, hdr, layout_info

def write_fits_preserve_layout(path, arr_yxc, header, layout_info):
    """
    Write arr_yxc back to FITS preserving original layout.
    If original layout was channel_first, transpose to (3, Y, X) before writing.
    Otherwise write as (Y, X, 3) or (Y, X) depending on arr_yxc shape.
    """
    out = np.array(arr_yxc, copy=False)
    if layout_info.get("channel_first", False):
        # expect arr_yxc shape (Y, X, C) -> transpose to (C, Y, X)
        if out.ndim == 3 and out.shape[2] == 3:
            out_write = np.transpose(out, (2, 0, 1))
        elif out.ndim == 2:
            out_write = out[np.newaxis, :, :]
        else:
            out_write = out
    else:
        out_write = out

    # If header exists, copy and try to update BITPIX depending on dtype
    hdr = header.copy() if header is not None else fits.Header()
    # astropy will set BITPIX based on dtype; but set explicitly for floats
    try:
        if np.issubdtype(out_write.dtype, np.floating):
            hdr['BITPIX'] = -64 if out_write.dtype == np.float64 else -32
    except Exception:
        pass

    fits.writeto(path, out_write, hdr, overwrite=True)

# ----------------------------
# Small utility helpers for preview
# ----------------------------
def qpix_from_rgb(arr):
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 2:
        arr8 = np.clip(a, 0, 255).astype(np.uint8)
        h, w = arr8.shape
        qimg = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    if a.ndim == 3:
        # expect RGB for preview; if C,Y,X convert first
        if a.shape[0] == 3:
            rgb = np.transpose(a, (1,2,0))
        else:
            rgb = a
        # scale floats / large ints -> uint8 for preview
        if rgb.dtype != np.uint8:
            b = rgb.astype(np.float64)
            mn, mx = np.nanmin(b), np.nanmax(b)
            if mx > mn:
                b8 = ((b - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                b8 = np.zeros_like(b, dtype=np.uint8)
        else:
            b8 = rgb
        if b8.shape[2] >= 3:
            rgb8 = b8[:, :, :3]
            h, w, ch = rgb8.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg)
    return None

def downsample_for_preview(arr, maxdim=(320,320)):
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[0] == 3:
        img = np.transpose(arr, (1,2,0))
    else:
        img = arr
    h, w = img.shape[:2]
    mw, mh = maxdim
    scale = min(1.0, mw / w, mh / h)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        try:
            import cv2
            if img.ndim == 2:
                disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            disp = img
        return disp
    return img

def bin_array_blockwise(img, bin_size):
    # img is 2D numpy array
    h, w = img.shape
    nh = h // bin_size
    nw = w // bin_size
    # trim edges
    img_trim = img[:nh*bin_size, :nw*bin_size]
    if bin_size == 1:
        return img_trim
    reshaped = img_trim.reshape(nh, bin_size, nw, bin_size)
    return reshaped.mean(axis=(1,3))

# ----------------------------
# Normalization and Siril helpers (new)
# ----------------------------
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
            # try to find normalize_gui.py in same directory as this script
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
            subprocess.Popen([siril_exe, os.path.abspath(output_path)])
            return True, "Siril launched."
        else:
            return False, "Siril executable not found in PATH."
    except Exception as e:
        return False, f"Failed to launch Siril: {e}"

# ----------------------------
# Main Window Definition
# ----------------------------
class ImageFiltersWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Filters")
        self.initUI()

    def initUI(self):
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        mainLayout = QVBoxLayout(centralWidget)

        # Top: Dropdown for filter selection
        topLayout = QHBoxLayout()
        topLayout.addWidget(QLabel("Select Filter:"))
        self.filterCombo = QComboBox()
        self.filterCombo.addItems([
            "Unsharp Mask", "LrDeconv", "FFT", "Erosion",
            "Dilation", "Gaussian", "HpMore", "LocAdapt"
        ])
        self.filterCombo.currentIndexChanged.connect(self.changePage)
        topLayout.addWidget(self.filterCombo)
        topLayout.addStretch()

        # New global checkboxes
        self.checkInputNormalized = QCheckBox("Check input normalized to [0,1]")
        self.checkInputNormalized.setChecked(True)
        topLayout.addWidget(self.checkInputNormalized)

        self.checkOpenInSiril = QCheckBox("Offer to open result in Siril after save")
        self.checkOpenInSiril.setChecked(True)
        topLayout.addWidget(self.checkOpenInSiril)

        mainLayout.addLayout(topLayout)

        # Stacked widget for parameters for each filter
        self.stack = QStackedWidget()
        mainLayout.addWidget(self.stack)

        self.stack.addWidget(self.createUnsharpPage())   # Page 0
        self.stack.addWidget(self.createLrDeconvPage())  # Page 1
        self.stack.addWidget(self.createFFTPage())       # Page 2
        self.stack.addWidget(self.createErosionPage())   # Page 3
        self.stack.addWidget(self.createDilationPage())  # Page 4
        self.stack.addWidget(self.createGaussianPage())  # Page 5
        self.stack.addWidget(self.createHpMorePage())    # Page 6
        self.stack.addWidget(self.createLocAdaptPage())  # Page 7

        # Run button and status label
        btnLayout = QHBoxLayout()
        self.runButton = QPushButton("Run")
        self.runButton.clicked.connect(self.runFilter)
        btnLayout.addWidget(self.runButton)
        btnLayout.addStretch()
        mainLayout.addLayout(btnLayout)
        self.statusLabel = QLabel("")
        mainLayout.addWidget(self.statusLabel)

    def changePage(self, index):
        self.stack.setCurrentIndex(index)
        self.statusLabel.setText("")

    # Helper: browse for FITS file
    def browseFile(self, lineEdit, save=False):
        if save:
            fname, _ = QFileDialog.getSaveFileName(self, "Select File", "", "FITS Files (*.fits)")
        else:
            fname, _ = QFileDialog.getOpenFileName(self, "Select File", "", "FITS Files (*.fits)")
        if fname:
            lineEdit.setText(fname)

    # Page 0: Unsharp Mask
    def createUnsharpPage(self):
        page = QWidget()
        layout = QGridLayout(page)
        layout.addWidget(QLabel("Input FITS File:"), 0, 0)
        self.usInput = QLineEdit()
        layout.addWidget(self.usInput, 0, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.browseFile(self.usInput))
        layout.addWidget(btn, 0, 2)

        layout.addWidget(QLabel("Output FITS File:"), 1, 0)
        self.usOutput = QLineEdit()
        layout.addWidget(self.usOutput, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.clicked.connect(lambda: self.browseFile(self.usOutput, save=True))
        layout.addWidget(btn2, 1, 2)
        return page

    # Page 1: LrDeconv (Richardson-Lucy)
    def createLrDeconvPage(self):
        page = QWidget()
        layout = QGridLayout(page)
        layout.addWidget(QLabel("Input FITS File:"), 0, 0)
        self.lrInput = QLineEdit()
        layout.addWidget(self.lrInput, 0, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.browseFile(self.lrInput))
        layout.addWidget(btn, 0, 2)
        layout.addWidget(QLabel("Output FITS File:"), 1, 0)
        self.lrOutput = QLineEdit()
        layout.addWidget(self.lrOutput, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.clicked.connect(lambda: self.browseFile(self.lrOutput, save=True))
        layout.addWidget(btn2, 1, 2)
        layout.addWidget(QLabel("PSF Mode:"), 2, 0)
        self.lrPsfMode = QComboBox()
        self.lrPsfMode.addItems(["Analytical", "Extract"])
        layout.addWidget(self.lrPsfMode, 2, 1)
        layout.addWidget(QLabel("PSF X (if extract):"), 3, 0)
        self.lrPsfX = QLineEdit()
        layout.addWidget(self.lrPsfX, 3, 1)
        layout.addWidget(QLabel("PSF Y:"), 4, 0)
        self.lrPsfY = QLineEdit()
        layout.addWidget(self.lrPsfY, 4, 1)
        layout.addWidget(QLabel("PSF Size:"), 5, 0)
        self.lrPsfSize = QLineEdit()
        layout.addWidget(self.lrPsfSize, 5, 1)
        layout.addWidget(QLabel("Iterations:"), 6, 0)
        self.lrIter = QLineEdit("30")
        layout.addWidget(self.lrIter, 6, 1)
        return page

    # Page 2: FFT Filter
    def createFFTPage(self):
        page = QWidget()
        layout = QGridLayout(page)
        layout.addWidget(QLabel("Input FITS File:"), 0, 0)
        self.fftInput = QLineEdit()
        layout.addWidget(self.fftInput, 0, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.browseFile(self.fftInput))
        layout.addWidget(btn, 0, 2)
        layout.addWidget(QLabel("Output FITS File:"), 1, 0)
        self.fftOutput = QLineEdit()
        layout.addWidget(self.fftOutput, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.clicked.connect(lambda: self.browseFile(self.fftOutput, save=True))
        layout.addWidget(btn2, 1, 2)
        layout.addWidget(QLabel("Cutoff:"), 2, 0)
        self.fftCutoff = QLineEdit("25")
        layout.addWidget(self.fftCutoff, 2, 1)
        layout.addWidget(QLabel("Weight:"), 3, 0)
        self.fftWeight = QLineEdit("50")
        layout.addWidget(self.fftWeight, 3, 1)
        layout.addWidget(QLabel("Denom:"), 4, 0)
        self.fftDenom = QLineEdit("100")
        layout.addWidget(self.fftDenom, 4, 1)
        layout.addWidget(QLabel("Radius:"), 5, 0)
        self.fftRadius = QLineEdit("1")
        layout.addWidget(self.fftRadius, 5, 1)
        layout.addWidget(QLabel("Second Cutoff:"), 6, 0)
        self.fftSecondCutoff = QLineEdit("10")
        layout.addWidget(self.fftSecondCutoff, 6, 1)
        return page

    # Page 3: Erosion
    def createErosionPage(self):
        page = QWidget()
        layout = QGridLayout(page)
        layout.addWidget(QLabel("Input FITS File:"), 0, 0)
        self.erInput = QLineEdit()
        layout.addWidget(self.erInput, 0, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.browseFile(self.erInput))
        layout.addWidget(btn, 0, 2)
        layout.addWidget(QLabel("Output FITS File:"), 1, 0)
        self.erOutput = QLineEdit("erosion_out.fits")
        layout.addWidget(self.erOutput, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.clicked.connect(lambda: self.browseFile(self.erOutput, save=True))
        layout.addWidget(btn2, 1, 2)
        layout.addWidget(QLabel("Iterations:"), 2, 0)
        self.erIter = QLineEdit("3")
        layout.addWidget(self.erIter, 2, 1)
        layout.addWidget(QLabel("Kernel Size:"), 3, 0)
        self.erKernel = QLineEdit("3")
        layout.addWidget(self.erKernel, 3, 1)
        return page

    # Page 4: Dilation
    def createDilationPage(self):
        page = QWidget()
        layout = QGridLayout(page)
        layout.addWidget(QLabel("Input FITS File:"), 0, 0)
        self.diInput = QLineEdit()
        layout.addWidget(self.diInput, 0, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.browseFile(self.diInput))
        layout.addWidget(btn, 0, 2)
        layout.addWidget(QLabel("Output FITS File:"), 1, 0)
        self.diOutput = QLineEdit("dilation_out.fits")
        layout.addWidget(self.diOutput, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.clicked.connect(lambda: self.browseFile(self.diOutput, save=True))
        layout.addWidget(btn2, 1, 2)
        layout.addWidget(QLabel("Iterations:"), 2, 0)
        self.diIter = QLineEdit("3")
        layout.addWidget(self.diIter, 2, 1)
        layout.addWidget(QLabel("Kernel Size:"), 3, 0)
        self.diKernel = QLineEdit("3")
        layout.addWidget(self.diKernel, 3, 1)
        return page

    # Page 5: Gaussian Blur
    def createGaussianPage(self):
        page = QWidget()
        layout = QGridLayout(page)
        layout.addWidget(QLabel("Input FITS File(Split Color First):"), 0, 0)
        self.gaInput = QLineEdit()
        layout.addWidget(self.gaInput, 0, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.browseFile(self.gaInput))
        layout.addWidget(btn, 0, 2)
        layout.addWidget(QLabel("Output FITS File:"), 1, 0)
        self.gaOutput = QLineEdit("_Gaus_Out.fits")
        layout.addWidget(self.gaOutput, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.clicked.connect(lambda: self.browseFile(self.gaOutput, save=True))
        layout.addWidget(btn2, 1, 2)
        layout.addWidget(QLabel("Sigma = (FWHM ∕ 2.355)="), 2, 0)
        self.gaSigma = QLineEdit("2.0")
        layout.addWidget(self.gaSigma, 2, 1)
        return page

    # Page 6: HpMore (High-pass More) with moderate checkbox
    def createHpMorePage(self):
        page = QWidget()
        layout = QGridLayout(page)
        layout.addWidget(QLabel("Input FITS File:"), 0, 0)
        self.hpInput = QLineEdit()
        layout.addWidget(self.hpInput, 0, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.browseFile(self.hpInput))
        layout.addWidget(btn, 0, 2)

        layout.addWidget(QLabel("Output FITS File:"), 1, 0)
        self.hpOutput = QLineEdit()
        layout.addWidget(self.hpOutput, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.clicked.connect(lambda: self.browseFile(self.hpOutput, save=True))
        layout.addWidget(btn2, 1, 2)

        # New checkbox for moderate sharpening
        self.hpModerateChk = QCheckBox("Use moderate sharpening")
        layout.addWidget(self.hpModerateChk, 2, 0, 1, 3)

        return page

    # Page 7: LocAdapt
    def createLocAdaptPage(self):
        page = QWidget()
        layout = QGridLayout(page)
        layout.addWidget(QLabel("Input FITS File:"), 0, 0)
        self.laInput = QLineEdit()
        layout.addWidget(self.laInput, 0, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.browseFile(self.laInput))
        layout.addWidget(btn, 0, 2)
        layout.addWidget(QLabel("Output FITS File:"), 1, 0)
        self.laOutput = QLineEdit("locadapt_out.fits")
        layout.addWidget(self.laOutput, 1, 1)
        btn2 = QPushButton("Browse")
        btn2.clicked.connect(lambda: self.browseFile(self.laOutput, save=True))
        layout.addWidget(btn2, 1, 2)
        layout.addWidget(QLabel("Neighborhood Size:"), 2, 0)
        self.laNeigh = QLineEdit("15")
        layout.addWidget(self.laNeigh, 2, 1)
        layout.addWidget(QLabel("Contrast (target std):"), 3, 0)
        self.laContrast = QLineEdit("50")
        layout.addWidget(self.laContrast, 3, 1)
        layout.addWidget(QLabel("Feather Distance:"), 4, 0)
        self.laFeather = QLineEdit("5")
        layout.addWidget(self.laFeather, 4, 1)
        return page

    # Run Filter button handler:
    def runFilter(self):
        idx = self.filterCombo.currentIndex()
        try:
            if idx == 0:
                self.runUnsharpMask()
            elif idx == 1:
                self.runLrDeconv()
            elif idx == 2:
                self.runFFT()
            elif idx == 3:
                self.runErosion()
            elif idx == 4:
                self.runDilation()
            elif idx == 5:
                self.runGaussian()
            elif idx == 6:
                self.runHpMore()
            elif idx == 7:
                self.runLocAdapt()
        except Exception as e:
            self.statusLabel.setText("Error: " + str(e))

    # -------------------------
    # Processing routines
    # -------------------------
    def runUnsharpMask(self):
        inp = self.usInput.text().strip()
        outp = self.usOutput.text().strip()
        if not inp or not outp:
            self.statusLabel.setText("Provide input and output files.")
            return

        arr, header, info = load_fits_rgb(inp)

        # Normalization check (global)
        if self.checkInputNormalized.isChecked():
            if not _check_normalized_0_1(arr):
                launched = _launch_normalize_gui(inp)
                if launched:
                    self.statusLabel.setText("Input not normalized. Launched normalize_gui.py for input.")
                else:
                    self.statusLabel.setText("Input not normalized and normalize_gui.py not found.")
                return

        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
            blurred = cv2.GaussianBlur(arr.astype(np.float32), (0, 0), 2.0)
            result = cv2.addWeighted(arr.astype(np.float32), 2.0, blurred, -1.0, 0.0)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            res_channels = []
            for c in range(3):
                channel = arr[:, :, c].astype(np.float32)
                blurred = cv2.GaussianBlur(channel, (0, 0), 2.0)
                res_channels.append(cv2.addWeighted(channel, 2.0, blurred, -1.0, 0.0))
            result = np.stack(res_channels, axis=2)
        else:
            self.statusLabel.setText("Unsupported dimensions for Unsharp Mask.")
            return
        write_fits_preserve_layout(outp, result.astype(np.float64), header, info)
        msg = "Unsharp Mask saved to " + outp
        # Siril launch if requested
        if self.checkOpenInSiril.isChecked():
            launched, siril_msg = _maybe_launch_siril(outp)
            msg += "  " + siril_msg
        self.statusLabel.setText(msg)

    def runLrDeconv(self):
        inp = self.lrInput.text().strip()
        outp = self.lrOutput.text().strip()
        if not inp or not outp:
            self.statusLabel.setText("Provide input and output files.")
            return
        try:
            iters = int(self.lrIter.text().strip())
        except Exception:
            iters = 30
        psf_mode = self.lrPsfMode.currentText()

        arr, header, info = load_fits_rgb(inp)

        # Normalization check
        if self.checkInputNormalized.isChecked():
            if not _check_normalized_0_1(arr):
                launched = _launch_normalize_gui(inp)
                if launched:
                    self.statusLabel.setText("Input not normalized. Launched normalize_gui.py for input.")
                else:
                    self.statusLabel.setText("Input not normalized and normalize_gui.py not found.")
                return

        from astropy.convolution import convolve_fft

        def richardson_lucy(im, psf, iterations):
            im = im.astype(np.float64)
            im_est = np.maximum(im.copy(), 1e-12)
            psf_mirror = psf[::-1, ::-1]
            for i in range(iterations):
                conv_est = convolve_fft(im_est, psf, normalize_kernel=True)
                conv_est[conv_est == 0] = 1e-7
                relative_blur = im / conv_est
                correction = convolve_fft(relative_blur, psf_mirror, normalize_kernel=True)
                im_est *= correction
            return im_est

        # Build PSF
        if psf_mode == "Analytical":
            sigma = 2.0
            ks = 25
            ax = np.linspace(-(ks - 1) / 2., (ks - 1) / 2., ks)
            xx, yy = np.meshgrid(ax, ax)
            psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
            psf /= psf.sum()
        else:
            x = float(self.lrPsfX.text().strip() or 0.0)
            y = float(self.lrPsfY.text().strip() or 0.0)
            size = int(self.lrPsfSize.text().strip() or 25)
            # Work on first channel to extract psf region
            base_image = arr[:, :, 0] if arr.ndim == 3 else arr
            cutout = Cutout2D(base_image, (x, y), size)
            psf = cutout.data.copy()
            psf -= np.median(psf)
            psf[psf < 0] = 0
            if psf.sum() != 0:
                psf /= psf.sum()

        if arr.ndim == 2:
            deconv = richardson_lucy(arr, psf, iters)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            deconv_channels = [richardson_lucy(arr[:, :, c], psf, iters) for c in range(3)]
            deconv = np.stack(deconv_channels, axis=2)
        else:
            self.statusLabel.setText("Unsupported dimensions for LrDeconv.")
            return

        write_fits_preserve_layout(outp, deconv.astype(np.float64), header, info)
        msg = "LrDeconv saved to " + outp
        if self.checkOpenInSiril.isChecked():
            launched, siril_msg = _maybe_launch_siril(outp)
            msg += "  " + siril_msg
        self.statusLabel.setText(msg)

    def runFFT(self):
        inp = self.fftInput.text().strip()
        outp = self.fftOutput.text().strip()
        if not inp or not outp:
            self.statusLabel.setText("Provide input and output files.")
            return
        try:
            cutoff = float(self.fftCutoff.text().strip())
            weight = float(self.fftWeight.text().strip())
            denom = float(self.fftDenom.text().strip())
            radius = int(self.fftRadius.text().strip())
            cutoff2 = float(self.fftSecondCutoff.text().strip())
        except Exception:
            self.statusLabel.setText("Invalid FFT parameters.")
            return

        data, header, info = load_fits_rgb(inp)

        # Normalization check
        if self.checkInputNormalized.isChecked():
            if not _check_normalized_0_1(data):
                launched = _launch_normalize_gui(inp)
                if launched:
                    self.statusLabel.setText("Input not normalized. Launched normalize_gui.py for input.")
                else:
                    self.statusLabel.setText("Input not normalized and normalize_gui.py not found.")
                return

        if data.ndim != 3 or data.shape[2] != 3:
            self.statusLabel.setText("FFT requires a 3-channel image.")
            return

        # Work channel-first for FFT processing convenience: convert to (C,Y,X)
        b, g, r = data[:, :, 0], data[:, :, 1], data[:, :, 2]

        def high_pass_filter(im, cutoff_frac, weight_frac):
            fft = np.fft.fft2(im)
            fft_shift = np.fft.fftshift(fft)
            rows, cols = im.shape
            crow, ccol = rows // 2, cols // 2
            rad = int(cutoff_frac * min(rows, cols))
            mask = np.ones((rows, cols), np.float32)
            mask[max(0, crow - rad):min(rows, crow + rad), max(0, ccol - rad):min(cols, ccol + rad)] = 0
            fft_shift_filtered = fft_shift * mask
            fft_inverse = np.fft.ifftshift(fft_shift_filtered)
            im_filt = np.abs(np.fft.ifft2(fft_inverse))
            im_weighted = cv2.addWeighted(im.astype(np.float32), 1 - weight_frac, im_filt.astype(np.float32), weight_frac, 0)
            return im_weighted

        def feather_image(im, radius_px, distance):
            radius_px = max(1, int(radius_px) | 1)
            im_blur = cv2.GaussianBlur(im, (radius_px, radius_px), 0) if radius_px > 1 else im
            mask = np.full(im.shape, 255, dtype=np.uint8)
            k = max(1, int(distance) * 2 + 1)
            mask_blur = cv2.GaussianBlur(mask, (k, k), 0)
            return (im_blur * (mask_blur / 255.0)).astype(im.dtype)

        def process_channel(ch):
            ch_norm = np.interp(ch, (np.nanmin(ch), np.nanmax(ch)), (0, 1)).astype(np.float32)
            filtered = high_pass_filter(ch_norm, cutoff / denom, weight / denom)
            return feather_image(filtered, radius, int(cutoff2))

        b_proc = process_channel(b)
        g_proc = process_channel(g)
        r_proc = process_channel(r)
        processed = np.stack([b_proc, g_proc, r_proc], axis=2)
        write_fits_preserve_layout(outp, processed.astype(np.float64), header, info)
        msg = "FFT saved to " + outp
        if self.checkOpenInSiril.isChecked():
            launched, siril_msg = _maybe_launch_siril(outp)
            msg += "  " + siril_msg
        self.statusLabel.setText(msg)

    def runErosion(self):
        inp = self.erInput.text().strip()
        outp = self.erOutput.text().strip()
        if not inp or not outp:
            self.statusLabel.setText("Provide input and output files.")
            return
        try:
            iters = int(self.erIter.text().strip())
            ksize = int(self.erKernel.text().strip())
        except Exception:
            self.statusLabel.setText("Invalid erosion parameters.")
            return

        data, header, info = load_fits_rgb(inp)

        # Normalization check
        if self.checkInputNormalized.isChecked():
            if not _check_normalized_0_1(data):
                launched = _launch_normalize_gui(inp)
                if launched:
                    self.statusLabel.setText("Input not normalized. Launched normalize_gui.py for input.")
                else:
                    self.statusLabel.setText("Input not normalized and normalize_gui.py not found.")
                return

        if data.ndim == 3 and data.shape[2] == 3:
            res_channels = []
            for c in range(3):
                channel = data[:, :, c].astype(np.float32)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                res_channels.append(cv2.erode(channel, kernel, iterations=iters))
            result = np.stack(res_channels, axis=2)
        elif data.ndim == 2:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            result = cv2.erode(data.astype(np.float32), kernel, iterations=iters)
        else:
            self.statusLabel.setText("Unsupported dimensions in erosion.")
            return
        write_fits_preserve_layout(outp, result.astype(np.float64), header, info)
        msg = "Erosion saved to " + outp
        if self.checkOpenInSiril.isChecked():
            launched, siril_msg = _maybe_launch_siril(outp)
            msg += "  " + siril_msg
        self.statusLabel.setText(msg)

    def runDilation(self):
        inp = self.diInput.text().strip()
        outp = self.diOutput.text().strip()
        if not inp or not outp:
            self.statusLabel.setText("Provide input and output files.")
            return
        try:
            iters = int(self.diIter.text().strip())
            ksize = int(self.diKernel.text().strip())
        except Exception:
            self.statusLabel.setText("Invalid dilation parameters.")
            return

        data, header, info = load_fits_rgb(inp)

        # Normalization check
        if self.checkInputNormalized.isChecked():
            if not _check_normalized_0_1(data):
                launched = _launch_normalize_gui(inp)
                if launched:
                    self.statusLabel.setText("Input not normalized. Launched normalize_gui.py for input.")
                else:
                    self.statusLabel.setText("Input not normalized and normalize_gui.py not found.")
                return

        if data.ndim == 3 and data.shape[2] == 3:
            res_channels = []
            for c in range(3):
                channel = data[:, :, c].astype(np.float32)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                res_channels.append(cv2.dilate(channel, kernel, iterations=iters))
            result = np.stack(res_channels, axis=2)
        elif data.ndim == 2:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            result = cv2.dilate(data.astype(np.float32), kernel, iterations=iters)
        else:
            self.statusLabel.setText("Unsupported dimensions in dilation.")
            return
        write_fits_preserve_layout(outp, result.astype(np.float64), header, info)
        msg = "Dilation saved to " + outp
        if self.checkOpenInSiril.isChecked():
            launched, siril_msg = _maybe_launch_siril(outp)
            msg += "  " + siril_msg
        self.statusLabel.setText(msg)

    def runGaussian(self):
        inp = self.gaInput.text().strip()
        outp = self.gaOutput.text().strip()
        if not inp or not outp:
            self.statusLabel.setText("Provide input and output files.")
            return
        try:
            sigma = float(self.gaSigma.text().strip())
        except Exception:
            self.statusLabel.setText("Invalid sigma.")
            return

        data, header, info = load_fits_rgb(inp)

        # Normalization check
        if self.checkInputNormalized.isChecked():
            if not _check_normalized_0_1(data):
                launched = _launch_normalize_gui(inp)
                if launched:
                    self.statusLabel.setText("Input not normalized. Launched normalize_gui.py for input.")
                else:
                    self.statusLabel.setText("Input not normalized and normalize_gui.py not found.")
                return

        if data.ndim == 3 and data.shape[2] == 3:
            res_channels = []
            for c in range(3):
                channel = data[:, :, c]
                norm = np.interp(channel, (np.nanmin(channel), np.nanmax(channel)), (0, 1)).astype(np.float32)
                res_channels.append(cv2.GaussianBlur(norm, (0, 0), sigmaX=sigma, sigmaY=sigma))
            result = np.stack(res_channels, axis=2)
        elif data.ndim == 2:
            norm = np.interp(data, (np.nanmin(data), np.nanmax(data)), (0, 1)).astype(np.float32)
            result = cv2.GaussianBlur(norm, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:
            self.statusLabel.setText("Unsupported dimensions in Gaussian.")
            return
        write_fits_preserve_layout(outp, result.astype(np.float64), header, info)
        msg = "Gaussian blur saved to " + outp
        if self.checkOpenInSiril.isChecked():
            launched, siril_msg = _maybe_launch_siril(outp)
            msg += "  " + siril_msg
        self.statusLabel.setText(msg)

    def runHpMore(self):
        inp = self.hpInput.text().strip()
        outp = self.hpOutput.text().strip()
        if not inp or not outp:
            self.statusLabel.setText("Provide input and output files for HpMore.")
            return

        data, header, info = load_fits_rgb(inp)

        # Normalization check
        if self.checkInputNormalized.isChecked():
            if not _check_normalized_0_1(data):
                launched = _launch_normalize_gui(inp)
                if launched:
                    self.statusLabel.setText("Input not normalized. Launched normalize_gui.py for input.")
                else:
                    self.statusLabel.setText("Input not normalized and normalize_gui.py not found.")
                return

        # For HpMore we want channel-first internal processing (C, Y, X)
        if data.ndim == 3 and data.shape[2] == 3:
            # convert to (C, Y, X)
            data_cf = np.transpose(data, (2, 0, 1))
        elif data.ndim == 3 and data.shape[0] == 3:
            # perhaps already channel-first - convert to (C, Y, X) by identity
            data_cf = np.array(data)
        else:
            self.statusLabel.setText("Input FITS is not a 3-channel image in HpMore.")
            return

        # Choose kernel strength based on checkbox
        if getattr(self, "hpModerateChk", None) and self.hpModerateChk.isChecked():
            # moderate sharpening kernel (3x3 Laplacian-like)
            hp_kernel = np.array([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=float)
            mode_text = "High Pass (moderate)"
        else:
            # stronger 5x5 kernel (existing aggressive kernel)
            hp_kernel = np.array([
                [-1, -1, -1, -1, -1],
                [-1,  1,  2,  1, -1],
                [-1,  2,  4,  2, -1],
                [-1,  1,  2,  1, -1],
                [-1, -1, -1, -1, -1]
            ], dtype=float)
            mode_text = "High Pass More (strong)"

        filtered_channels = np.empty_like(data_cf)
        for i in range(data_cf.shape[0]):
            channel = data_cf[i]
            # compute highpass response
            highpass = convolve2d(channel, hp_kernel, mode='same', boundary='symm')
            # apply only above threshold to avoid boosting background noise
            threshold = np.percentile(channel, 70)
            mask = (channel > threshold).astype(float)
            enhanced = channel + highpass * mask
            filtered_channels[i] = enhanced

        # convert back to (Y, X, C)
        result = np.transpose(filtered_channels, (1, 2, 0))
        write_fits_preserve_layout(outp, result.astype(np.float64), header, info)
        msg = f"{mode_text} saved to " + outp
        if self.checkOpenInSiril.isChecked():
            launched, siril_msg = _maybe_launch_siril(outp)
            msg += "  " + siril_msg
        self.statusLabel.setText(msg)

    def runLocAdapt(self):
        inp = self.laInput.text().strip()
        outp = self.laOutput.text().strip()
        if not inp or not outp:
            self.statusLabel.setText("Provide input and output files.")
            return
        try:
            neigh = int(self.laNeigh.text().strip())
            contrast = float(self.laContrast.text().strip())
            feather = int(self.laFeather.text().strip())
        except Exception:
            self.statusLabel.setText("Invalid LocAdapt parameters.")
            return

        data, header, info = load_fits_rgb(inp)

        # Normalization check
        if self.checkInputNormalized.isChecked():
            if not _check_normalized_0_1(data):
                launched = _launch_normalize_gui(inp)
                if launched:
                    self.statusLabel.setText("Input not normalized. Launched normalize_gui.py for input.")
                else:
                    self.statusLabel.setText("Input not normalized and normalize_gui.py not found.")
                return

        # Simple local adaptation example: scale local std to target contrast
        if data.ndim == 3 and data.shape[2] == 3:
            out_channels = []
            for c in range(3):
                ch = data[:, :, c]
                # compute local std via gaussian blur of squared deviations
                mean = cv2.GaussianBlur(ch.astype(np.float32), (neigh|1, neigh|1), 0)
                sq = (ch - mean) ** 2
                var = cv2.GaussianBlur(sq.astype(np.float32), (neigh|1, neigh|1), 0)
                std = np.sqrt(np.maximum(var, 1e-12))
                scale = contrast / (std + 1e-12)
                out = ch * scale
                out_channels.append(out)
            result = np.stack(out_channels, axis=2)
        else:
            ch = data if data.ndim == 2 else data[:, :, 0]
            mean = cv2.GaussianBlur(ch.astype(np.float32), (neigh|1, neigh|1), 0)
            sq = (ch - mean) ** 2
            var = cv2.GaussianBlur(sq.astype(np.float32), (neigh|1, neigh|1), 0)
            std = np.sqrt(np.maximum(var, 1e-12))
            scale = contrast / (std + 1e-12)
            result = ch * scale

        write_fits_preserve_layout(outp, result.astype(np.float64), header, info)
        msg = "LocAdapt saved to " + outp
        if self.checkOpenInSiril.isChecked():
            launched, siril_msg = _maybe_launch_siril(outp)
            msg += "  " + siril_msg
        self.statusLabel.setText(msg)


def main():
    app = QApplication(sys.argv)
    window = ImageFiltersWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()