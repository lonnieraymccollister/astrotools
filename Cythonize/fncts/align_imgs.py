#!/usr/bin/env python3
# align_imgs.py - standalone GUI to select and align multiple FITS files in tiles

import sys
import os
import copy
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from reproject import reproject_interp  # ensure reproject is installed

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QComboBox
)
from PyQt6.QtCore import Qt

# Tile sizes (tune for memory / speed)
CHUNK_X = 1024
CHUNK_Y = 1024

# -------------------------
# Minimal helper stubs
# -------------------------
def find_optimal_celestial_wcs(dw_list):
    """
    Given dw_list = [(data_array, header), ...] compute an output WCS and output shape.
    This is a placeholder; replace with your own WCS combination logic (mosaic, auto-scale, etc.).
    Current stub returns the first header's WCS and shape equal to that data shape.
    """
    data0, hdr0 = dw_list[0]
    wcs_out = WCS(hdr0) if hdr0 is not None else WCS(naxis=2)
    shape_out = data0.shape if data0.ndim == 2 else data0.shape[-2:]
    return wcs_out, shape_out

def process_chunk(arr_tile):
    """
    Per-tile processing hook. Replace this with any per-tile tweak you need.
    Current implementation returns arr_tile unchanged.
    """
    return arr_tile

# -------------------------
# GUI form
# -------------------------
class AlignImagesForm(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Align images")
        self.resize(760, 640)

        # choose how many images (2–13)
        self.count_combo = QComboBox()
        self.count_combo.addItems([f"{i:02d}" for i in range(2, 14)])
        self.count_combo.currentTextChanged.connect(self._update_fields)

        # input / output groups + layouts
        self.input_group = QGroupBox("Reference FITS files")
        self.output_group = QGroupBox("Aligned FITS files")
        self.input_form = QFormLayout()
        self.output_form = QFormLayout()
        self.input_group.setLayout(self.input_form)
        self.output_group.setLayout(self.output_form)

        # keep parallel lists so we can show/hide rows
        self.in_labels = []
        self.in_containers = []
        self.out_labels = []
        self.out_containers = []

        for i in range(13):
            # build Input row
            lab_in = QLabel(f"Image {i+1}:")
            edit_in = QLineEdit()
            btn_in = QPushButton("Browse…")
            btn_in.clicked.connect(lambda _, idx=i: self._browse_input(idx))
            hbox_in = QHBoxLayout()
            hbox_in.addWidget(edit_in)
            hbox_in.addWidget(btn_in)
            container_in = QWidget()
            container_in.setLayout(hbox_in)
            self.input_form.addRow(lab_in, container_in)

            # build Output row
            lab_out = QLabel(f"Aligned {i+1}:")
            edit_out = QLineEdit()
            btn_out = QPushButton("Browse…")
            btn_out.clicked.connect(lambda _, idx=i: self._browse_output(idx))
            hbox_out = QHBoxLayout()
            hbox_out.addWidget(edit_out)
            hbox_out.addWidget(btn_out)
            container_out = QWidget()
            container_out.setLayout(hbox_out)
            self.output_form.addRow(lab_out, container_out)

            # store references
            self.in_labels.append(lab_in)
            self.in_containers.append(container_in)
            self.out_labels.append(lab_out)
            self.out_containers.append(container_out)

        # Align button
        self.align_button = QPushButton("Align")
        self.align_button.clicked.connect(self._on_align)

        # Layout assembly
        top = QHBoxLayout()
        top.addWidget(QLabel("Number of images:"))
        top.addWidget(self.count_combo)
        top.addStretch()

        main = QVBoxLayout(self)
        main.addLayout(top)
        main.addWidget(self.input_group)
        main.addWidget(self.output_group)
        main.addStretch()
        main.addWidget(self.align_button, alignment=Qt.AlignmentFlag.AlignRight)

        # hide all but the first N rows
        self._update_fields(self.count_combo.currentText())

    def _update_fields(self, txt):
        n = int(txt)
        for i in range(len(self.in_labels)):
            show = (i < n)
            self.in_labels[i].setVisible(show)
            self.in_containers[i].setVisible(show)
            self.out_labels[i].setVisible(show)
            self.out_containers[i].setVisible(show)

    def _browse_input(self, idx):
        fn, _ = QFileDialog.getOpenFileName(self, f"Select reference #{idx+1}", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            le = self.in_containers[idx].findChild(QLineEdit)
            le.setText(fn)

    def _browse_output(self, idx):
        fn, _ = QFileDialog.getSaveFileName(self, f"Save aligned #{idx+1}", "", "FITS Files (*.fits *.fit);;All Files (*)")
        if fn:
            le = self.out_containers[idx].findChild(QLineEdit)
            le.setText(fn)

    def _on_align(self):
        count = int(self.count_combo.currentText())
        inputs = [self.in_containers[i].findChild(QLineEdit).text().strip() for i in range(count)]
        outputs = [self.out_containers[i].findChild(QLineEdit).text().strip() for i in range(count)]

        if any(not p for p in inputs + outputs):
            QMessageBox.warning(self, "Missing", "Fill in all paths.")
            return

        try:
            # 1) Read everyone’s data & header (memmap)
            dw = []
            for fn in inputs:
                hdul = fits.open(fn, memmap=True)
                data = hdul[0].data
                hdr = hdul[0].header
                if data is None:
                    hdul.close()
                    raise RuntimeError(f"No data in primary HDU of {fn}")
                # ensure 2D array for reprojection; for cubes pick first 2D plane or adapt as needed
                if data.ndim > 2:
                    # take first 2D slice (modify if you need full cubes)
                    data2 = data[0].astype(np.float64)
                else:
                    data2 = data.astype(np.float64)
                dw.append((data2, hdr))
                hdul.close()

            # 2) Compute common WCS + output shape
            wcs_out, shape_out = find_optimal_celestial_wcs(dw)
            ny, nx = int(shape_out[0]), int(shape_out[1])

            # 3) For each input -> reproject in tiles -> write memmap
            for idx, (fn, (data, hdr)) in enumerate(zip(inputs, dw)):
                outfn = outputs[idx]
                # prepare an empty memmap'd FITS with WCS header
                header_out = wcs_out.to_header()
                primary = fits.PrimaryHDU(data=np.zeros((ny, nx), dtype=np.float32), header=header_out)
                primary.writeto(outfn, overwrite=True)
                out_hdul = fits.open(outfn, mode='update', memmap=True)
                out_mem = out_hdul[0].data

                # tile over Y/X
                for y0 in range(0, ny, CHUNK_Y):
                    y1 = min(y0 + CHUNK_Y, ny)
                    for x0 in range(0, nx, CHUNK_X):
                        x1 = min(x0 + CHUNK_X, nx)

                        # adjust WCS CRPIX so reproject_interp only fills the tile
                        wcs_tile = copy.deepcopy(wcs_out)
                        # modify CRPIX (1-based WCS pixel coordinates)
                        try:
                            wcs_tile.wcs.crpix[0] -= x0
                            wcs_tile.wcs.crpix[1] -= y0
                        except Exception:
                            pass

                        # do reprojection for this tile
                        arr_tile, _ = reproject_interp(
                            (data, hdr),
                            wcs_tile,
                            shape_out=(y1 - y0, x1 - x0)
                        )

                        # apply per-tile tweak
                        arr_tile = process_chunk(arr_tile)

                        # write it back to memmap
                        out_mem[y0:y1, x0:x1] = arr_tile.astype(out_mem.dtype)

                out_hdul.flush()
                out_hdul.close()

            QMessageBox.information(self, "Done", f"Aligned {count} images.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

# -------------------------
# Entry point
# -------------------------
def main():
    app = QApplication(sys.argv)
    w = AlignImagesForm()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()