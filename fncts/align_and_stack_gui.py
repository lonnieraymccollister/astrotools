#!/usr/bin/env python3
"""
align_and_stack_gui_with_counts.py

PyQt6 GUI for aligning FITS files using RANSAC star matching (astroalign)
and stacking them (mean or median). Robust shape handling, fallback, and
counts of successful/failed alignments. Final output saved as float32.
"""

import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
import astroalign as aa

from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QGridLayout, QTextEdit, QMessageBox, QComboBox, QHBoxLayout
)
from PyQt6.QtCore import Qt


# ----------------------------
# FITS helpers (unchanged)
# ----------------------------
def load_fits(path):
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.array(hdu.data), hdu.header
    raise ValueError(f"No image data in {path}")


def save_fits_float32(path, data, header):
    hdr = header.copy()
    hdr['BITPIX'] = -32
    for k in ('BZERO', 'BSCALE'):
        if k in hdr:
            hdr.pop(k)
    fits.writeto(path, data.astype(np.float32), hdr, overwrite=True)


def normalize_fits_shape(arr):
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] == 3:
        return np.transpose(a, (2, 0, 1))
    if a.ndim == 3 and a.shape[0] == 3:
        return a
    raise ValueError(f"Unsupported FITS shape: {a.shape}")


def to_luma(arr):
    a = np.asarray(arr)
    if a.ndim == 2:
        return a.astype(np.float64)
    if a.ndim == 3 and a.shape[0] == 3:
        return np.mean(a.astype(np.float64), axis=0)
    raise ValueError(f"Unsupported shape for luma: {a.shape}")


def apply_transform_full(data, transform, ref_luma):
    a = np.asarray(data)
    if a.ndim == 2:
        return aa.apply_transform(transform, a, ref_luma)
    if a.ndim == 3 and a.shape[0] == 3:
        out = np.zeros((3,) + ref_luma.shape, dtype=np.float64)
        for i in range(3):
            out[i] = aa.apply_transform(transform, a[i].astype(np.float64), ref_luma)
        return out
    raise ValueError(f"Unsupported shape for transform: {a.shape}")


def apply_shift_full(data, shift):
    a = np.asarray(data)
    if a.ndim == 2:
        return np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(a.astype(np.float64)), shift)))
    if a.ndim == 3 and a.shape[0] == 3:
        out = np.zeros((3,) + a.shape[1:], dtype=np.float64)
        for i in range(3):
            out[i] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(a[i].astype(np.float64)), shift)))
        return out
    raise ValueError(f"Unsupported shape for shift: {a.shape}")


def enforce_color_shape(arr, ref_shape):
    a = np.asarray(arr)
    if len(ref_shape) == 2:
        if a.ndim == 2:
            return a
        if a.ndim == 3 and a.shape[0] == 1:
            return a[0]
        raise ValueError(f"Expected mono frame, got {a.shape}")
    if len(ref_shape) == 3 and ref_shape[0] == 3:
        if a.ndim == 3 and a.shape[0] == 3:
            return a
        if a.ndim == 2:
            return np.stack([a, a, a], axis=0)
        raise ValueError(f"Expected 3-channel frame, got {a.shape}")
    raise ValueError(f"Unexpected reference shape: {ref_shape}")


# ----------------------------
# GUI
# ----------------------------
class AlignStackGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Align & Stack FITS (with counts)")
        self.resize(1020, 580)
        self._build_ui()
        self.ref_path = None
        self.target_paths = []

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout()
        central.setLayout(grid)

        # Reference FITS
        grid.addWidget(QLabel("Reference FITS:"), 0, 0)
        self.ref_edit = QLineEdit()
        grid.addWidget(self.ref_edit, 0, 1, 1, 3)
        btn_ref = QPushButton("Browse")
        btn_ref.clicked.connect(self._browse_ref)
        grid.addWidget(btn_ref, 0, 4)

        # Target FITS
        grid.addWidget(QLabel("Target FITS (multi-select):"), 1, 0)
        self.tgt_edit = QLineEdit()
        grid.addWidget(self.tgt_edit, 1, 1, 1, 3)
        btn_tgt = QPushButton("Browse")
        btn_tgt.clicked.connect(self._browse_targets)
        grid.addWidget(btn_tgt, 1, 4)

        # Output
        grid.addWidget(QLabel("Output stacked FITS:"), 2, 0)
        self.out_edit = QLineEdit("stacked_float32.fits")
        grid.addWidget(self.out_edit, 2, 1, 1, 3)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_out)
        grid.addWidget(btn_out, 2, 4)

        # Stacking method
        grid.addWidget(QLabel("Stacking method:"), 3, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["mean", "median"])
        grid.addWidget(self.method_combo, 3, 1)

        # Run button
        self.run_btn = QPushButton("Align & Stack")
        self.run_btn.clicked.connect(self._on_run)
        grid.addWidget(self.run_btn, 3, 2)

        # Live counters (success / fail)
        counters_layout = QHBoxLayout()
        self.success_label = QLabel("Success: 0")
        self.fail_label = QLabel("Fail: 0")
        counters_layout.addWidget(self.success_label)
        counters_layout.addWidget(self.fail_label)
        counters_layout.addStretch()
        grid.addLayout(counters_layout, 3, 3, 1, 2)

        # Log window
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        grid.addWidget(self.log, 4, 0, 10, 5)

    def _log(self, *args):
        self.log.append(" ".join(str(a) for a in args))
        QApplication.processEvents()

    # ----------------------------
    # File dialogs
    # ----------------------------
    def _browse_ref(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select reference FITS", "", "FITS files (*.fits *.fit)")
        if fn:
            self.ref_edit.setText(fn)

    def _browse_targets(self):
        fns, _ = QFileDialog.getOpenFileNames(self, "Select target FITS", "", "FITS files (*.fits *.fit)")
        if fns:
            self.target_paths = fns
            self.tgt_edit.setText("; ".join(fns))

    def _browse_out(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save stacked FITS", self.out_edit.text(), "FITS files (*.fits *.fit)")
        if fn:
            self.out_edit.setText(fn)

    # ----------------------------
    # Main processing with counters
    # ----------------------------
    def _on_run(self):
        ref_path = self.ref_edit.text().strip()
        out_path = self.out_edit.text().strip()
        method = self.method_combo.currentText()

        if not ref_path or not Path(ref_path).exists():
            QMessageBox.warning(self, "Missing reference", "Select a valid reference FITS.")
            return
        if not self.target_paths:
            QMessageBox.warning(self, "Missing targets", "Select at least one target FITS.")
            return

        # Initialize counters and failed list
        success_count = 0
        fail_count = 0
        failed_files = []

        # Reset live labels
        self.success_label.setText("Success: 0")
        self.fail_label.setText("Fail: 0")

        try:
            # Load and normalize reference
            self._log(f"Loading reference: {ref_path}")
            ref_data_raw, ref_hdr = load_fits(ref_path)
            self._log(f"Raw reference shape: {np.asarray(ref_data_raw).shape}, dtype={ref_data_raw.dtype}")
            ref_data = normalize_fits_shape(ref_data_raw)
            ref_luma = to_luma(ref_data)
            self._log(f"Normalized reference shape: {ref_data.shape}, luma shape: {ref_luma.shape}")

            aligned_frames = []

            # Align each target
            for p in self.target_paths:
                self._log(f"\nLoading target: {p}")
                try:
                    data_raw, hdr = load_fits(p)
                    self._log(f"Raw target shape: {np.asarray(data_raw).shape}, dtype={data_raw.dtype}")

                    data = normalize_fits_shape(data_raw)
                    self._log(f"Normalized target shape: {data.shape}")

                    tgt_luma = to_luma(data)

                    # Try astroalign first
                    aligned = None
                    try:
                        self._log("Attempting astroalign RANSAC transform...")
                        transform, _ = aa.find_transform(tgt_luma, ref_luma)
                        self._log("astroalign transform found.")
                        aligned = apply_transform_full(data, transform, ref_luma)
                    except Exception as e_aa:
                        self._log("astroalign failed:", e_aa)
                        # Fallback to phase correlation shift
                        try:
                            self._log("Falling back to phase correlation shift...")
                            shift, error, diffphase = phase_cross_correlation(ref_luma, tgt_luma, upsample_factor=100)
                            self._log(f"Phase correlation shift (dy,dx): {shift}, error: {error}")
                            aligned = apply_shift_full(data, shift)
                        except Exception as e_shift:
                            self._log("Fallback shift failed:", e_shift)
                            raise RuntimeError(f"Alignment failed for {p}") from e_shift

                    # Ensure dtype and shape consistency
                    aligned = np.asarray(aligned)
                    self._log(f"Aligned shape before enforcement: {aligned.shape}")

                    aligned = enforce_color_shape(aligned, ref_data.shape)
                    self._log(f"Aligned shape after enforcement: {aligned.shape}")

                    # Convert aligned to float64 for stacking precision
                    aligned = aligned.astype(np.float64)

                    # Append and increment success
                    aligned_frames.append(aligned)
                    success_count += 1
                    self.success_label.setText(f"Success: {success_count}")
                    self._log(f"Aligned OK: {p} (success_count={success_count})")

                except Exception as e_target:
                    # Log failure, increment fail_count, continue with next file
                    fail_count += 1
                    failed_files.append(p)
                    self.fail_label.setText(f"Fail: {fail_count}")
                    self._log(f"Failed to align {p}: {e_target}")
                    # continue to next target without raising

            # After loop: report counts
            self._log("\nAlignment summary:")
            self._log(f"  Successful alignments: {success_count}")
            self._log(f"  Failed alignments: {fail_count}")
            if failed_files:
                self._log("  Failed files:")
                for f in failed_files:
                    self._log("   -", f)

            if success_count == 0:
                raise RuntimeError("No frames were successfully aligned; aborting stack.")

            # Final check: all frames same shape
            shapes = [a.shape for a in aligned_frames]
            self._log("All aligned frame shapes:", shapes)
            if not all(s == shapes[0] for s in shapes):
                raise RuntimeError(f"Aligned frames have inconsistent shapes: {shapes}")

            # Stack (compute in float64 for best precision)
            self._log("\nStacking frames...")
            stack_arr = np.stack(aligned_frames, axis=0)

            if method == "mean":
                final = np.mean(stack_arr, axis=0)
            else:
                final = np.median(stack_arr, axis=0)

            # Convert final to float32 for output
            final_float32 = final.astype(np.float32)

            # Save as 32-bit float FITS and update header
            save_fits_float32(out_path, final_float32, ref_hdr)
            self._log(f"\nSaved stacked FITS (float32): {out_path}")

            # Final message includes counts
            summary = (f"Stack complete.\nSuccessful alignments: {success_count}\n"
                       f"Failed alignments: {fail_count}\nSaved: {out_path}")
            if failed_files:
                summary += "\n\nFailed files:\n" + "\n".join(failed_files)
            QMessageBox.information(self, "Done", summary)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._log("Error:", e)
            QMessageBox.critical(self, "Error", f"{e}\n\n{tb}")


# ----------------------------
# Main
# ----------------------------
def main():
    app = QApplication(sys.argv)
    w = AlignStackGUI()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()