#!/usr/bin/env python3
"""
pixelmath.py
Standalone PyQt6 GUI for pixel-wise arithmetic on FITS images (Add, Subtract, Multiply, Divide, Max, Min).
Usage: python pixelmath.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import cv2
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt

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
        layout.addWidget(QLabel("Image1 Brightness Adjustment:"), 4, 0)
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

        # Set minimum size
        self.setMinimumSize(640, 240)

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

    # Compute operation
    def computeOperation(self):
        self.statusLabel.setText("")  # clear
        inputs = self.validateInputs()
        if inputs is None:
            return

        op = self.operationComboBox.currentText()
        try:
            # Open FITS
            with fits.open(inputs["first_file"]) as hdul1, fits.open(inputs["second_file"]) as hdul2:
                data1 = hdul1[0].data
                header1 = hdul1[0].header
                data2 = hdul2[0].data

            if data1 is None or data2 is None:
                self.statusLabel.setText("Error: One of the FITS files contains no data.")
                return

            # Convert to float64 for safe arithmetic
            im1 = data1.astype(np.float64) * inputs["img1_scale"]
            im2 = data2.astype(np.float64) * inputs["img2_scale"]

            # Dimension checks
            if im1.shape != im2.shape:
                self.statusLabel.setText("Error: Input images do not have the same dimensions.")
                return
            if im1.ndim not in (2, 3):
                self.statusLabel.setText("Error: Unsupported image dimensionality (expect 2D mono or 3D color).")
                return

            # Perform selected op
            if op == "Add":
                result_image = im1 + im2 + inputs["brightness"]
            elif op == "Subtract":
                result_image = im1 - im2 + inputs["brightness"]
            elif op == "Multiply":
                result_image = im1 * im2 + inputs["brightness"]
            elif op == "Divide":
                # safe divide: where im2==0 -> set to 0 (or choose alternative)
                result_image = np.divide(im1, im2, out=np.zeros_like(im1), where=im2 != 0) + inputs["brightness"]
            elif op == "Max":
                result_image = np.maximum(im1, im2) + inputs["brightness"]
            elif op == "Min":
                result_image = np.minimum(im1, im2) + inputs["brightness"]
            else:
                self.statusLabel.setText("Unknown operation selected.")
                return

            # Write result with original header where possible
            hdu = fits.PrimaryHDU(result_image.astype(np.float64), header=header1)
            hdu.writeto(inputs["output_file"], overwrite=True)
            self.statusLabel.setText("Operation completed; output saved successfully.")
        except Exception as e:
            self.statusLabel.setText(f"An error occurred: {e}")

def main():
    app = QApplication(sys.argv)
    window = PixelMathWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
    win = PixelMathWindow()
    win.show()