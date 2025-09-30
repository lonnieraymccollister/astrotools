#!/usr/bin/env python3
"""
color_tool.py
Simple PyQt6 GUI for: Split Tricolor, Combine Tricolor, Create Luminance (FITS or common image files).
Save this file and run: python color_tool.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import cv2
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QComboBox, QStackedWidget
)
from PyQt6.QtCore import Qt

class ColorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Color")
        self.initUI()

    def initUI(self):
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        mainLayout = QVBoxLayout(centralWidget)

        # Operation selection
        opLayout = QHBoxLayout()
        opLabel = QLabel("Select Operation:")
        self.opCombo = QComboBox()
        self.opCombo.addItems(["Split Tricolor", "Combine Tricolor", "Create Luminance"])
        self.opCombo.currentIndexChanged.connect(self.operationChanged)
        opLayout.addWidget(opLabel)
        opLayout.addWidget(self.opCombo)
        opLayout.addStretch()
        mainLayout.addLayout(opLayout)

        # Stacked widget for operation parameters
        self.stack = QStackedWidget()
        mainLayout.addWidget(self.stack)

        # Page 0: Split Tricolor
        self.splitWidget = QWidget()
        splitLayout = QGridLayout(self.splitWidget)
        splitLayout.addWidget(QLabel("Input Color Image:"), 0, 0)
        self.splitInputLine = QLineEdit()
        splitLayout.addWidget(self.splitInputLine, 0, 1)
        self.splitInputBrowseBtn = QPushButton("Browse")
        self.splitInputBrowseBtn.clicked.connect(lambda: self.browseFile(self.splitInputLine))
        splitLayout.addWidget(self.splitInputBrowseBtn, 0, 2)
        splitLayout.addWidget(QLabel("Mode:"), 1, 0)
        self.splitModeCombo = QComboBox()
        self.splitModeCombo.addItems(["FITS", "Other"])
        splitLayout.addWidget(self.splitModeCombo, 1, 1)
        splitLayout.addWidget(QLabel("Output Base Name:"), 2, 0)
        self.splitOutputBaseLine = QLineEdit()
        splitLayout.addWidget(self.splitOutputBaseLine, 2, 1)
        self.stack.addWidget(self.splitWidget)

        # Page 1: Combine Tricolor
        self.combineWidget = QWidget()
        combineLayout = QGridLayout(self.combineWidget)
        combineLayout.addWidget(QLabel("Blue Image:"), 0, 0)
        self.combineBlueLine = QLineEdit()
        combineLayout.addWidget(self.combineBlueLine, 0, 1)
        self.combineBlueBrowseBtn = QPushButton("Browse")
        self.combineBlueBrowseBtn.clicked.connect(lambda: self.browseFile(self.combineBlueLine))
        combineLayout.addWidget(self.combineBlueBrowseBtn, 0, 2)
        combineLayout.addWidget(QLabel("Green Image:"), 1, 0)
        self.combineGreenLine = QLineEdit()
        combineLayout.addWidget(self.combineGreenLine, 1, 1)
        self.combineGreenBrowseBtn = QPushButton("Browse")
        self.combineGreenBrowseBtn.clicked.connect(lambda: self.browseFile(self.combineGreenLine))
        combineLayout.addWidget(self.combineGreenBrowseBtn, 1, 2)
        combineLayout.addWidget(QLabel("Red Image:"), 2, 0)
        self.combineRedLine = QLineEdit()
        combineLayout.addWidget(self.combineRedLine, 2, 1)
        self.combineRedBrowseBtn = QPushButton("Browse")
        self.combineRedBrowseBtn.clicked.connect(lambda: self.browseFile(self.combineRedLine))
        combineLayout.addWidget(self.combineRedBrowseBtn, 2, 2)
        combineLayout.addWidget(QLabel("Output Combined Image:"), 3, 0)
        self.combineOutputLine = QLineEdit()
        combineLayout.addWidget(self.combineOutputLine, 3, 1)
        self.combineOutputBrowseBtn = QPushButton("Browse")
        self.combineOutputBrowseBtn.clicked.connect(lambda: self.browseFile(self.combineOutputLine, save=True))
        combineLayout.addWidget(self.combineOutputBrowseBtn, 3, 2)
        combineLayout.addWidget(QLabel("Mode:"), 4, 0)
        self.combineModeCombo = QComboBox()
        self.combineModeCombo.addItems(["FITS", "Other"])
        combineLayout.addWidget(self.combineModeCombo, 4, 1)
        self.stack.addWidget(self.combineWidget)

        # Page 2: Create Luminance
        self.luminanceWidget = QWidget()
        lumLayout = QGridLayout(self.luminanceWidget)
        lumLayout.addWidget(QLabel("Input Color Image:"), 0, 0)
        self.lumInputLine = QLineEdit()
        lumLayout.addWidget(self.lumInputLine, 0, 1)
        self.lumInputBrowseBtn = QPushButton("Browse")
        self.lumInputBrowseBtn.clicked.connect(lambda: self.browseFile(self.lumInputLine))
        lumLayout.addWidget(self.lumInputBrowseBtn, 0, 2)
        lumLayout.addWidget(QLabel("Output Luminance Image:"), 1, 0)
        self.lumOutputLine = QLineEdit()
        lumLayout.addWidget(self.lumOutputLine, 1, 1)
        self.lumOutputBrowseBtn = QPushButton("Browse")
        self.lumOutputBrowseBtn.clicked.connect(lambda: self.browseFile(self.lumOutputLine, save=True))
        lumLayout.addWidget(self.lumOutputBrowseBtn, 1, 2)
        lumLayout.addWidget(QLabel("Mode:"), 2, 0)
        self.lumModeCombo = QComboBox()
        self.lumModeCombo.addItems(["FITS", "Other"])
        lumLayout.addWidget(self.lumModeCombo, 2, 1)
        self.stack.addWidget(self.luminanceWidget)

        # Run button + status
        self.runButton = QPushButton("Run")
        self.runButton.clicked.connect(self.runOperation)
        mainLayout.addWidget(self.runButton)
        self.statusLabel = QLabel("")
        mainLayout.addWidget(self.statusLabel)

    def operationChanged(self, index):
        self.stack.setCurrentIndex(index)

    def browseFile(self, lineEdit, save=False):
        if save:
            fileName, _ = QFileDialog.getSaveFileName(self, "Select File", "", "All Files (*)")
        else:
            fileName, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*)")
        if fileName:
            lineEdit.setText(fileName)

    def runOperation(self):
        op = self.opCombo.currentText()
        if op == "Split Tricolor":
            self.runSplitTricolor()
        elif op == "Combine Tricolor":
            self.runCombineTricolor()
        elif op == "Create Luminance":
            self.runCreateLuminance()
        else:
            self.statusLabel.setText("Unknown operation selected.")

    def runSplitTricolor(self):
        inputFile = self.splitInputLine.text().strip()
        mode = self.splitModeCombo.currentText()
        outputBase = self.splitOutputBaseLine.text().strip()
        if not inputFile:
            self.statusLabel.setText("Input file is required for Split Tricolor.")
            return
        if not outputBase:
            outputBase = os.path.splitext(inputFile)[0]
        try:
            if mode == "FITS":
                hdul = fits.open(inputFile)
                header = hdul[0].header
                data = hdul[0].data.astype(np.float32)
                hdul.close()
                if data.ndim != 3 or data.shape[0] != 3:
                    self.statusLabel.setText("FITS file does not appear to have 3 channels in first dimension.")
                    return
                b = data[2, :, :]
                g = data[1, :, :]
                r = data[0, :, :]
                fits.writeto(f"{outputBase}_channel_b.fits", b, header, overwrite=True)
                fits.writeto(f"{outputBase}_channel_g.fits", g, header, overwrite=True)
                fits.writeto(f"{outputBase}_channel_r.fits", r, header, overwrite=True)
                self.statusLabel.setText("Split Tricolor (FITS) completed successfully.")
            else:
                img = cv2.imread(inputFile, -1)
                if img is None:
                    self.statusLabel.setText("Error reading image in Other mode.")
                    return
                channels = cv2.split(img)
                cv2.imwrite(f"{outputBase}_Blue.png", channels[0])
                cv2.imwrite(f"{outputBase}_Green.png", channels[1])
                cv2.imwrite(f"{outputBase}_Red.png", channels[2])
                self.statusLabel.setText("Split Tricolor (Other) completed successfully.")
        except Exception as e:
            self.statusLabel.setText(f"Error in Split Tricolor: {e}")

    def runCombineTricolor(self):
        blueFile = self.combineBlueLine.text().strip()
        greenFile = self.combineGreenLine.text().strip()
        redFile = self.combineRedLine.text().strip()
        outputFile = self.combineOutputLine.text().strip()
        mode = self.combineModeCombo.currentText()
        if not (blueFile and greenFile and redFile and outputFile):
            self.statusLabel.setText("All input and output files are required for Combine Tricolor.")
            return
        try:
            if mode == "FITS":
                with fits.open(blueFile) as hdul:
                    header = hdul[0].header
                    blue = hdul[0].data.astype(np.float32)
                with fits.open(greenFile) as hdul:
                    green = hdul[0].data.astype(np.float32)
                with fits.open(redFile) as hdul:
                    red = hdul[0].data.astype(np.float32)
                RGB = np.stack((red, green, blue))
                hdu = fits.PrimaryHDU(data=RGB, header=header)
                hdu.writeto(outputFile, overwrite=True)
                self.statusLabel.setText("Combine Tricolor (FITS) completed successfully.")
            else:
                blue = cv2.imread(blueFile, -1)
                green = cv2.imread(greenFile, -1)
                red = cv2.imread(redFile, -1)
                if blue is None or green is None or red is None:
                    self.statusLabel.setText("Error reading one of the input images (Other).")
                    return
                merged = cv2.merge((red, green, blue))
                cv2.imwrite(outputFile, merged)
                self.statusLabel.setText("Combine Tricolor (Other) completed successfully.")
        except Exception as e:
            self.statusLabel.setText(f"Error in Combine Tricolor: {e}")

    def runCreateLuminance(self):
        inputFile = self.lumInputLine.text().strip()
        outputFile = self.lumOutputLine.text().strip()
        mode = self.lumModeCombo.currentText()
        if not (inputFile and outputFile):
            self.statusLabel.setText("Both input and output files are required for Create Luminance.")
            return
        try:
            if mode == "FITS":
                hdul = fits.open(inputFile)
                data = hdul[0].data
                hdul.close()
                if data.ndim == 3 and data.shape[0] == 3:
                    img = np.transpose(data, (1, 2, 0))
                elif data.ndim == 3 and data.shape[2] == 3:
                    img = data
                else:
                    self.statusLabel.setText("Unexpected FITS channel layout for luminance.")
                    return
                R = img[:, :, 0].astype(np.float64)
                G = img[:, :, 1].astype(np.float64)
                B = img[:, :, 2].astype(np.float64)
                luminance = 0.2989 * R + 0.5870 * G + 0.1140 * B
                fits.PrimaryHDU(data=luminance).writeto(outputFile, overwrite=True)
                self.statusLabel.setText("Create Luminance (FITS) completed successfully.")
            else:
                img = cv2.imread(inputFile, -1)
                if img is None:
                    self.statusLabel.setText("Error reading input image (Other).")
                    return
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(outputFile, gray)
                self.statusLabel.setText("Create Luminance (Other) completed successfully.")
        except Exception as e:
            self.statusLabel.setText(f"Error in Create Luminance: {e}")

def main():
    app = QApplication(sys.argv)
    window = ColorWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
    win = ColorWindow()
    win.show()