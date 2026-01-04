#!/usr/bin/env python3
"""
pixelmath.py

Standalone PyQt6 GUI for pixel-wise arithmetic on FITS images (Add, Subtract, Multiply, Divide, Max, Min).

Changes:
 - Improved FITS handling for 3-plane RGB preferred files:
     * Accepts both (3, Y, X) and (Y, X, 3) input layouts.
     * Performs arithmetic per-pixel and per-channel for 3-channel images.
     * Preserves/updates the primary header and writes the result with the same channel layout
       as the first input (if first input was channel-first (3,Y,X) the output keeps that).
 - Safer numeric handling (float64 arithmetic) and clearer error messages.
 - Added normalization checks and Siril launch option.
 - NEW: Global checkbox to control output normalization (default ON) (Patch 1)
 - NEW: write_fits_preserve_layout normalizes output and writes NORMED header (Patch 2)
 - UPDATED: write call uses write_fits_preserve_layout and respects checkbox (Patch 3)
 - UPDATED: explicit GUI warning when input is not normalized before launching normalizer (Patch 4)
"""
import sys
import os
import subprocess
from pathlib import Path
import shutil

import numpy as np
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QComboBox, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt


def write_fits_preserve_layout(path, arr_yxc, header, layout_info, normalize_output=True):
    """
    Write arr_yxc back to FITS preserving original layout.
    If normalize_output=True, normalize to [0,1] before saving.
    Adds FITS header keyword NORMED = 1 or 0.
    """
    out = np.array(arr_yxc, copy=False)

    # Optional normalization
    norm_applied = 0
    if normalize_output:
        finite = np.isfinite(out)
        if np.any(finite):
            mn = float(np.nanmin(out))
            mx = float(np.nanmax(out))
            if mx > mn:
                out = (out - mn) / (mx - mn)
            else:
                out = np.zeros_like(out, dtype=np.float64)
        else:
            out = np.zeros_like(out, dtype=np.float64)
        norm_applied = 1

    # Preserve original layout
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

    # Header handling
    hdr = header.copy() if header is not None else fits.Header()
    hdr['BITPIX'] = -32
    hdr['NORMED'] = (norm_applied, "1 if output was normalized to [0,1]")

    fits.writeto(path, out_write.astype(np.float32), hdr, overwrite=True)


class PixelMathWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel Math")
        self.initUI()

    def initUI(self):
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        layout = QGridLayout(centralWidget)

        # Row 0: Operation selector
        layout.addWidget(QLabel("Operation:"), 0, 0)
        self.operationComboBox = QComboBox()
        self.operationComboBox.addItems(["Add", "Subtract", "Multiply", "Divide", "Max", "Min"])
        layout.addWidget(self.operationComboBox, 0, 1, 1, 2)

        # Row 1: First image
        layout.addWidget(QLabel("First Image File:"), 1, 0)
        self.firstImageLineEdit = QLineEdit()
        layout.addWidget(self.firstImageLineEdit, 1, 1)
        self.firstBrowseButton = QPushButton("Browse")
        self.firstBrowseButton.clicked.connect(self.browseFirstImage)
        layout.addWidget(self.firstBrowseButton, 1, 2)

        # Row 2: Second image
        layout.addWidget(QLabel("Second Image File:"), 2, 0)
        self.secondImageLineEdit = QLineEdit()
        layout.addWidget(self.secondImageLineEdit, 2, 1)
        self.secondBrowseButton = QPushButton("Browse")
        self.secondBrowseButton.clicked.connect(self.browseSecondImage)
        layout.addWidget(self.secondBrowseButton, 2, 2)

        # Row 3: Output file
        layout.addWidget(QLabel("Output File:"), 3, 0)
        self.outputLineEdit = QLineEdit()
        layout.addWidget(self.outputLineEdit, 3, 1)
        self.outputBrowseButton = QPushButton("Browse")
        self.outputBrowseButton.clicked.connect(self.browseOutputFile)
        layout.addWidget(self.outputBrowseButton, 3, 2)

        # Row 4: Brightness adjustment
        layout.addWidget(QLabel("Brightness Adjustment (add):"), 4, 0)
        self.brightnessLineEdit = QLineEdit("0")
        layout.addWidget(self.brightnessLineEdit, 4, 1, 1, 2)

        # Row 5: Image1 contrast numerator/denom
        layout.addWidget(QLabel("Image1 Contrast Numerator:"), 5, 0)
        self.img1ContrastNumLineEdit = QLineEdit("1")
        layout.addWidget(self.img1ContrastNumLineEdit, 5, 1)
        layout.addWidget(QLabel("Denom:"), 5, 2)
        self.img1ContrastDenomLineEdit = QLineEdit("1")
        layout.addWidget(self.img1ContrastDenomLineEdit, 5, 3)

        # Row 6: Image2 contrast numerator/denom
        layout.addWidget(QLabel("Image2 Contrast Numerator:"), 6, 0)
        self.img2ContrastNumLineEdit = QLineEdit("1")
        layout.addWidget(self.img2ContrastNumLineEdit, 6, 1)
        layout.addWidget(QLabel("Denom:"), 6, 2)
        self.img2ContrastDenomLineEdit = QLineEdit("1")
        layout.addWidget(self.img2ContrastDenomLineEdit, 6, 3)

        # Row 7: Compute button
        self.computeButton = QPushButton("Compute")
        self.computeButton.clicked.connect(self.computeOperation)
        layout.addWidget(self.computeButton, 7, 1, 1, 2)

        # Row 8: Status
        self.statusLabel = QLabel("")
        layout.addWidget(self.statusLabel, 8, 0, 1, 4)

        # Row 9: Normalization check for first image
        self.checkFirstNormalized = QCheckBox("Check first image normalized to [0,1]")
        self.checkFirstNormalized.setChecked(True)
        layout.addWidget(self.checkFirstNormalized, 9, 0, 1, 4)

        # Row 10: Normalization check for second image
        self.checkSecondNormalized = QCheckBox("Check second image normalized to [0,1]")
        self.checkSecondNormalized.setChecked(True)
        layout.addWidget(self.checkSecondNormalized, 10, 0, 1, 4)

        # Row 11: Offer to open result in Siril
        self.checkOpenInSiril = QCheckBox("Offer to open result in Siril after save")
        self.checkOpenInSiril.setChecked(True)
        layout.addWidget(self.checkOpenInSiril, 11, 0, 1, 4)

        # NEW (Patch 1): Normalize output checkbox
        self.checkNormalizeOutput = QCheckBox("Normalize output FITS to [0,1]")
        self.checkNormalizeOutput.setChecked(True)
        layout.addWidget(self.checkNormalizeOutput, 12, 0, 1, 4)

        # Set minimum size
        self.setMinimumSize(820, 460)

    # File dialogs
    def browseFirstImage(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open First Image", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if filename:
            self.firstImageLineEdit.setText(filename)

    def browseSecondImage(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Second Image", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if filename:
            self.secondImageLineEdit.setText(filename)

    def browseOutputFile(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Output Image", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if filename:
            self.outputLineEdit.setText(filename)

    # Validation
    def validateInputs(self):
        first_file = self.firstImageLineEdit.text().strip()
        second_file = self.secondImageLineEdit.text().strip()
        output_file = self.outputLineEdit.text().strip()
        if not first_file or not second_file or not output_file:
            self.statusLabel.setText("Error: Provide first, second and output file paths.")
            return None

        try:
            brightness = float(self.brightnessLineEdit.text())
            img1_num = float(self.img1ContrastNumLineEdit.text())
            img1_denom = float(self.img1ContrastDenomLineEdit.text())
            img2_num = float(self.img2ContrastNumLineEdit.text())
            img2_denom = float(self.img2ContrastDenomLineEdit.text())
        except ValueError:
            self.statusLabel.setText("Error: Brightness/contrast fields must be numbers.")
            return None

        if img1_denom == 0 or img2_denom == 0:
            self.statusLabel.setText("Error: Contrast denominators must not be zero.")
            return None

        if not os.path.exists(first_file):
            self.statusLabel.setText("Error: First image file does not exist.")
            return None
        if not os.path.exists(second_file):
            self.statusLabel.setText("Error: Second image file does not exist.")
            return None

        return {
            "first_file": first_file,
            "second_file": second_file,
            "output_file": output_file,
            "brightness": brightness,
            "img1_scale": img1_num / img1_denom,
            "img2_scale": img2_num / img2_denom
        }

    # utility: load fits primary and normalize orientation
    def _load_fits_preserve_header(self, path):
        """
        Load primary HDU data and header.
        Return (data_array, header, layout_info)
        layout_info: dict with keys:
            'orig_shape' - original data shape
            'channel_first' - True if input was (3, Y, X)
        The returned data_array will be float64 and in shape:
            - (Y, X) for single-plane
            - (Y, X, 3) for three-channel
        """
        with fits.open(path, memmap=False) as hdul:
            data = hdul[0].data
            header = hdul[0].header

        if data is None:
            raise ValueError("FITS contains no primary data")

        arr = np.array(data)  # keep as-is for detection
        layout_info = {"orig_shape": arr.shape, "channel_first": False}

        # common cases:
        # (3, Y, X) -> transpose to (Y, X, 3)
        # (Y, X, 3) -> keep
        # (Y, X) -> expand to (Y, X, 1) or keep as 2D
        if arr.ndim == 3 and arr.shape[0] == 3:
            # channel-first
            arr = np.transpose(arr, (1, 2, 0))
            layout_info["channel_first"] = True
        elif arr.ndim == 3 and arr.shape[2] == 3:
            layout_info["channel_first"] = False
        elif arr.ndim == 2:
            # single plane -> leave as 2D
            pass
        else:
            # Unexpected but attempt to handle: if small first dimension equals 3, transpose
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
                layout_info["channel_first"] = True
            else:
                raise ValueError(f"Unsupported FITS data shape: {arr.shape}")

        # cast to float64 for arithmetic safety
        arr = arr.astype(np.float64, copy=False)
        return arr, header, layout_info

    def _check_normalized_0_1(self, arr, tol=1e-8):
        """
        Return True if arr min is approximately 0 and max approximately 1 within tol.
        Works for 2D and 3D arrays.
        """
        if arr.size == 0:
            return False
        amin = float(np.nanmin(arr))
        amax = float(np.nanmax(arr))
        if np.isnan(amin) or np.isnan(amax):
            return False
        return (abs(amin - 0.0) <= tol) and (abs(amax - 1.0) <= tol)

    def _warn_and_launch_normalize(self, filepath, which="Input"):
        """Show warning and attempt to launch normalize_gui.py"""
        QMessageBox.warning(
            self,
            f"{which} Not Normalized",
            f"{which} FITS is not normalized to [0,1].\n"
            "normalize_gui.py will be launched."
        )
        launched = self._launch_normalize_gui(filepath)
        if launched:
            self.statusLabel.setText(f"{which} not normalized. Launched normalize_gui.py.")
        else:
            self.statusLabel.setText(f"{which} not normalized and normalize_gui.py not found.")
        return launched

    def _launch_normalize_gui(self, filepath):
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
                self.statusLabel.setText("normalize_gui.py not found in current directory.")
                return False
            # Launch the normalizer with the filename as argument
            subprocess.Popen([sys.executable, str(script), str(filepath)])
            return True
        except Exception as e:
            self.statusLabel.setText(f"Failed to launch normalizer: {e}")
            return False

    # Compute operation
    def computeOperation(self):
        self.statusLabel.setText("")  # clear
        inputs = self.validateInputs()
        if inputs is None:
            return

        op = self.operationComboBox.currentText()

        # Load images first so we can check normalization if requested
        try:
            im1, hdr1, info1 = self._load_fits_preserve_header(inputs["first_file"])
            im2, hdr2, info2 = self._load_fits_preserve_header(inputs["second_file"])
        except Exception as e:
            self.statusLabel.setText(f"Error loading FITS: {e}")
            return

        # Normalization checks with explicit warning (Patch 4)
        try:
            if self.checkFirstNormalized.isChecked():
                if not self._check_normalized_0_1(im1):
                    launched = self._warn_and_launch_normalize(inputs["first_file"], which="First image")
                    return

            if self.checkSecondNormalized.isChecked():
                if not self._check_normalized_0_1(im2):
                    launched = self._warn_and_launch_normalize(inputs["second_file"], which="Second image")
                    return
        except Exception as e:
            self.statusLabel.setText(f"Normalization check failed: {e}")
            return

        # Check dimension compatibility after orientation normalization:
        if im1.shape != im2.shape:
            # allow single-plane vs 3-channel mismatch only if one can be broadcast: expand single plane to 3 channels
            if im1.ndim == 2 and im2.ndim == 3 and im2.shape[2] == 3:
                im1 = np.stack([im1] * 3, axis=-1)
            elif im2.ndim == 2 and im1.ndim == 3 and im1.shape[2] == 3:
                im2 = np.stack([im2] * 3, axis=-1)
            else:
                self.statusLabel.setText(f"Error: Input images do not have the same dimensions ({im1.shape} vs {im2.shape}).")
                return

        # perform scaling
        im1 = im1 * inputs["img1_scale"]
        im2 = im2 * inputs["img2_scale"]

        # Arithmetic (float64)
        try:
            if op == "Add":
                result_image = im1 + im2 + inputs["brightness"]
            elif op == "Subtract":
                result_image = im1 - im2 + inputs["brightness"]
            elif op == "Multiply":
                result_image = im1 * im2 + inputs["brightness"]
            elif op == "Divide":
                # safe divide
                result_image = np.divide(im1, im2, out=np.zeros_like(im1), where=im2 != 0) + inputs["brightness"]
            elif op == "Max":
                result_image = np.maximum(im1, im2) + inputs["brightness"]
            elif op == "Min":
                result_image = np.minimum(im1, im2) + inputs["brightness"]
            else:
                self.statusLabel.setText("Unknown operation selected.")
                return
        except Exception as e:
            self.statusLabel.setText(f"Arithmetic error: {e}")
            return

        # Decide output layout: preserve first input's original layout (channel-first or channel-last)
        out_header = hdr1.copy() if hdr1 is not None else fits.Header()
        # If first input was channel-first originally, transpose result back to (C, Y, X)
        if info1.get("channel_first", False):
            # result_image is (Y, X, C) or (Y, X) -> convert to (C, Y, X)
            if result_image.ndim == 3 and result_image.shape[2] == 3:
                out_arr = np.transpose(result_image, (2, 0, 1))
            elif result_image.ndim == 2:
                out_arr = result_image[np.newaxis, :, :]
            else:
                # unexpected shape - write as-is
                out_arr = result_image
        else:
            # keep as (Y, X, C) or (Y, X)
            out_arr = result_image

        # Ensure output dtype is float64 for precision (user can downcast later if desired)
        out_arr = out_arr.astype(np.float64, copy=False)

        # Update header BITPIX if present: FITS writer will set BITPIX based on data dtype,
        # but if we want to be explicit set to -64 for float64
        try:
            out_header['BITPIX'] = -64
        except Exception:
            pass

        # Write result using centralized writer that can normalize and set NORMED (Patch 3)
        try:
            write_fits_preserve_layout(
                inputs["output_file"],
                out_arr,
                out_header,
                info1,
                normalize_output=self.checkNormalizeOutput.isChecked()
            )
            self.statusLabel.setText("Operation completed; output saved successfully.")
        except Exception as e:
            self.statusLabel.setText(f"Failed writing output FITS: {e}")
            return

        # Offer to open in Siril if requested
        if self.checkOpenInSiril.isChecked():
            try:
                siril_exe = shutil.which("siril")
                if siril_exe:
                    # Launch Siril with the output file path
                    subprocess.Popen([siril_exe, os.path.abspath(inputs["output_file"])])
                    self.statusLabel.setText("Operation completed; output saved and Siril launched.")
                else:
                    self.statusLabel.setText("Operation completed; output saved. Siril executable not found in PATH.")
            except Exception as e:
                self.statusLabel.setText(f"Operation completed; output saved. Failed to launch Siril: {e}")


def main():
    app = QApplication(sys.argv)
    window = PixelMathWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()