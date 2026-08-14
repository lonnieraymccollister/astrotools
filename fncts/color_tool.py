#!/usr/bin/env python3
"""
color_tool.py
Simple PyQt6 GUI for: Split Tricolor, Combine Tricolor, Create Luminance (FITS or common image files).
Save this file and run: python color_tool.py

Change:
- In Combine Tricolor page a checkbox labeled "two" (unchecked by default).
  When checked, Blue input may be left empty and Blue will be computed as:
      Blue = 0.5 * R + 0.5 * G
  The combine then proceeds normally using that computed Blue channel.
"""
import sys
import os
from pathlib import Path

import numpy as np
import cv2
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QComboBox, QStackedWidget,
    QCheckBox
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

        # New checkbox "two - Leave Blue entry Empty" (unchecked by default)
        self.two_checkbox = QCheckBox("two - Leave Blue entry Empty")
        self.two_checkbox.setChecked(False)
        combineLayout.addWidget(self.two_checkbox, 5, 0, 1, 3)

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

        # New: Luminance equation dropdown
        lumLayout.addWidget(QLabel("Luminance Equation:"), 3, 0)
        self.lumEqCombo = QComboBox()
        # Add items in the requested order and exact labels
        self.lumEqCombo.addItems([
            "synthetic-luminance = 1 * R + 1 * G + 1 * B",
            "Human per(709) luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B",
            "Bayer-one shot color-luminance = 1 * R + 2 * G + 1 * B"
        ])
        lumLayout.addWidget(self.lumEqCombo, 3, 1, 1, 2)

        self.stack.addWidget(self.luminanceWidget)

        # --- Begin patch: add Narrowband Combine page and logic ---

        # In initUI(), after adding the existing pages, add a new page:
                # Add "Narrowband Combine" to operation selector
        self.opCombo.addItem("Narrowband Combine")

        # Page 3: Narrowband Combine
        self.narrowWidget = QWidget()
        nbLayout = QVBoxLayout(self.narrowWidget)

        # Controls row: Preset mapping and normalize
        controlsLayout = QHBoxLayout()
        controlsLayout.addWidget(QLabel("Preset:"))
        self.nbPresetCombo = QComboBox()
        self.nbPresetCombo.addItems([
            "Custom",
            "Halpha->R, OIII->G, SII->B (default)",
            "Halpha->R, OIII->B, SII->G",
            "Greyscale weighted"
        ])
        controlsLayout.addWidget(self.nbPresetCombo)
        self.nbNormalizeCheck = QCheckBox("Normalize each filter")
        self.nbNormalizeCheck.setChecked(True)
        controlsLayout.addWidget(self.nbNormalizeCheck)
        controlsLayout.addStretch()
        nbLayout.addLayout(controlsLayout)

        # Header labels for the filter rows
        headerLayout = QHBoxLayout()
        headerLayout.addWidget(QLabel("Wavelength (nm)"))
        headerLayout.addWidget(QLabel("File"))
        headerLayout.addWidget(QLabel("Map"))
        headerLayout.addStretch()
        nbLayout.addLayout(headerLayout)

        # Container for dynamic filter rows
        self.nbRowsContainer = QVBoxLayout()
        nbLayout.addLayout(self.nbRowsContainer)

        # Add / Remove buttons
        btnsLayout = QHBoxLayout()
        self.nbAddBtn = QPushButton("Add Filter")
        self.nbAddBtn.clicked.connect(self.nbAddRow)
        btnsLayout.addWidget(self.nbAddBtn)
        self.nbRemoveBtn = QPushButton("Remove Last")
        self.nbRemoveBtn.clicked.connect(self.nbRemoveRow)
        btnsLayout.addWidget(self.nbRemoveBtn)
        btnsLayout.addStretch()
        nbLayout.addLayout(btnsLayout)

        # Output file and mode
        outLayout = QGridLayout()
        outLayout.addWidget(QLabel("Output File:"), 0, 0)
        self.nbOutputLine = QLineEdit()
        outLayout.addWidget(self.nbOutputLine, 0, 1)
        self.nbOutputBrowseBtn = QPushButton("Browse")
        self.nbOutputBrowseBtn.clicked.connect(lambda: self.browseFile(self.nbOutputLine, save=True))
        outLayout.addWidget(self.nbOutputBrowseBtn, 0, 2)
        outLayout.addWidget(QLabel("Mode:"), 1, 0)
        self.nbModeCombo = QComboBox()
        self.nbModeCombo.addItems(["FITS", "Other"])
        outLayout.addWidget(self.nbModeCombo, 1, 1)
        nbLayout.addLayout(outLayout)

        # Add the narrowband page to the stack
        self.stack.addWidget(self.narrowWidget)

        # Keep an internal list of row widgets
        self.nb_rows = []

# End of UI additions

# Add these helper methods to the ColorWindow class:

    def nbAddRow(self):
        """Add a narrowband filter row: wavelength, file, mapping."""
        rowWidget = QWidget()
        layout = QHBoxLayout(rowWidget)
        wlLine = QLineEdit()
        wlLine.setPlaceholderText("e.g., 656.3")
        layout.addWidget(wlLine)
        fileLine = QLineEdit()
        layout.addWidget(fileLine)
        browseBtn = QPushButton("Browse")
        browseBtn.clicked.connect(lambda: self.browseFile(fileLine))
        layout.addWidget(browseBtn)
        mapCombo = QComboBox()
        mapCombo.addItems(["Assign to R", "Assign to G", "Assign to B", "Weighted"])
        layout.addWidget(mapCombo)
        self.nbRowsContainer.addWidget(rowWidget)
        self.nb_rows.append((rowWidget, wlLine, fileLine, mapCombo))

    def nbRemoveRow(self):
        """Remove last narrowband row."""
        if not self.nb_rows:
            return
        rowWidget, wlLine, fileLine, mapCombo = self.nb_rows.pop()
        self.nbRowsContainer.removeWidget(rowWidget)
        rowWidget.setParent(None)
        rowWidget.deleteLater()

# Modify runOperation to handle the new operation:
    def runOperation(self):
        op = self.opCombo.currentText()
        if op == "Split Tricolor":
            self.runSplitTricolor()
        elif op == "Combine Tricolor":
            self.runCombineTricolor()
        elif op == "Create Luminance":
            self.runCreateLuminance()
        elif op == "Narrowband Combine":
            self.runNarrowbandCombine()
        else:
            self.statusLabel.setText("Unknown operation selected.")

# Add the main processing method:

    def runNarrowbandCombine(self):
        """Combine multiple narrowband filter images into an RGB image using mappings/presets."""
        # Collect rows
        rows = []
        for (_, wlLine, fileLine, mapCombo) in self.nb_rows:
            wl_text = wlLine.text().strip()
            file_text = fileLine.text().strip()
            map_text = mapCombo.currentText()
            if not file_text:
                continue
            try:
                wl = float(wl_text) if wl_text else None
            except:
                wl = None
            rows.append({"wl": wl, "file": file_text, "map": map_text})

        if not rows:
            self.statusLabel.setText("Add at least one narrowband filter (file) to combine.")
            return
        outputFile = self.nbOutputLine.text().strip()
        mode = self.nbModeCombo.currentText()
        normalize = self.nbNormalizeCheck.isChecked()
        preset = self.nbPresetCombo.currentText()

        # Preset mapping convenience: if user selected a preset, try to auto-assign by wavelength
        # Common lines: H-alpha ~656.3, OIII ~500.7, SII ~672.4
        def assign_preset(rows, preset):
            if preset.startswith("Halpha->R"):
                for r in rows:
                    if r["wl"] is None:
                        continue
                    wl = r["wl"]
                    if abs(wl - 656.3) < 5:
                        r["map"] = "Assign to R"
                    elif abs(wl - 500.7) < 5:
                        r["map"] = "Assign to G"
                    elif abs(wl - 672.4) < 5:
                        r["map"] = "Assign to B"
            # other presets can be added similarly

        if preset != "Custom":
            assign_preset(rows, preset)

        try:
            # Read all filter images into float arrays and optionally normalize
            arrays = []
            header = None
            for r in rows:
                f = r["file"]
                if mode == "FITS" or f.lower().endswith(".fits"):
                    hdul = fits.open(f)
                    data = hdul[0].data.astype(np.float64)
                    if header is None:
                        header = hdul[0].header
                    hdul.close()
                    # Accept channel-first or channel-last; prefer single-channel
                    if data.ndim == 3:
                        # If channel-first with 3 channels, try to collapse to single channel by taking first
                        if data.shape[0] == 3:
                            arr = data[0, :, :].astype(np.float64)
                        elif data.shape[2] == 3:
                            arr = data[:, :, 0].astype(np.float64)
                        else:
                            arr = data.astype(np.float64)
                    else:
                        arr = data.astype(np.float64)
                else:
                    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        self.statusLabel.setText(f"Error reading file: {f}")
                        return
                    if img.ndim == 3 and img.shape[2] >= 3:
                        # assume BGR; take first channel as intensity if single-filter image
                        arr = img[:, :, 0].astype(np.float64)
                    elif img.ndim == 2:
                        arr = img.astype(np.float64)
                    else:
                        arr = img[:, :, 0].astype(np.float64)

                # Normalize each filter to its max if requested
                if normalize:
                    maxv = np.nanmax(arr)
                    if maxv != 0 and not np.isnan(maxv):
                        arr = arr / maxv
                arrays.append((r, arr))

            # Determine output shape from first array
            base_shape = arrays[0][1].shape
            # Initialize R,G,B accumulators as float64
            R = np.zeros(base_shape, dtype=np.float64)
            G = np.zeros(base_shape, dtype=np.float64)
            B = np.zeros(base_shape, dtype=np.float64)

            # Distribute arrays according to mapping
            for (r, arr) in arrays:
                # If shapes mismatch, try to resize or error out
                if arr.shape != base_shape:
                    self.statusLabel.setText("Input filter images have different shapes; resize externally.")
                    return
                m = r["map"]
                if m == "Assign to R":
                    R += arr
                elif m == "Assign to G":
                    G += arr
                elif m == "Assign to B":
                    B += arr
                elif m == "Weighted":
                    # Weighted: add equally to all channels by default
                    R += arr
                    G += arr
                    B += arr
                else:
                    # fallback: add to all
                    R += arr
                    G += arr
                    B += arr

            # Optional post-normalize to prevent overflow when writing integer images
            # Scale to 0..1 then to dtype range if needed
            if mode == "FITS":
                # Stack as (R,G,B) channel-first to match your FITS convention
                RGB = np.stack((R, G, B))
                if header is None:
                    fits.PrimaryHDU(data=RGB).writeto(outputFile, overwrite=True)
                else:
                    fits.PrimaryHDU(data=RGB, header=header).writeto(outputFile, overwrite=True)
                self.statusLabel.setText("Narrowband Combine (FITS) completed successfully.")
            else:
                # For Other: map to output dtype similar to runCombineTricolor
                # Determine dtype from first input file if possible
                sample_dtype = None
                # try to infer dtype from first input file on disk
                first_file = rows[0]["file"]
                sample_img = cv2.imread(first_file, cv2.IMREAD_UNCHANGED)
                if sample_img is not None:
                    sample_dtype = sample_img.dtype
                else:
                    sample_dtype = np.uint8

                # If float accumulators, scale to dtype range
                if np.issubdtype(sample_dtype, np.integer):
                    info = np.iinfo(sample_dtype)
                    # Normalize combined channels to 0..max
                    max_val = max(np.nanmax(R), np.nanmax(G), np.nanmax(B), 1.0)
                    R_out = np.clip((R / max_val) * info.max, info.min, info.max).astype(sample_dtype)
                    G_out = np.clip((G / max_val) * info.max, info.min, info.max).astype(sample_dtype)
                    B_out = np.clip((B / max_val) * info.max, info.min, info.max).astype(sample_dtype)
                else:
                    # float output
                    R_out = R.astype(np.float32)
                    G_out = G.astype(np.float32)
                    B_out = B.astype(np.float32)

                merged = cv2.merge((B_out, G_out, R_out))
                ok = cv2.imwrite(outputFile, merged)
                if not ok:
                    self.statusLabel.setText("Failed to write narrowband output image (Other).")
                    return
                self.statusLabel.setText("Narrowband Combine (Other) completed successfully.")

        except Exception as e:
            self.statusLabel.setText(f"Error in Narrowband Combine: {e}")

# --- End patch


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
                # OpenCV returns B,G,R order
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
        two_checked = self.two_checkbox.isChecked()

        # Require red, green, output always
        if not (greenFile and redFile and outputFile):
            self.statusLabel.setText("Red, Green and Output files are required for Combine Tricolor.")
            return

        try:
            if mode == "FITS":
                # Read red and green (required)
                with fits.open(redFile) as hdul:
                    red = hdul[0].data.astype(np.float32)
                    header = hdul[0].header
                with fits.open(greenFile) as hdul:
                    green = hdul[0].data.astype(np.float32)

                if two_checked:
                    # Compute blue = 0.5 * R + 0.5 * G
                    blue = 0.5 * red + 0.5 * green
                else:
                    if not blueFile:
                        self.statusLabel.setText("Blue file required when 'two' is not checked.")
                        return
                    with fits.open(blueFile) as hdul:
                        blue = hdul[0].data.astype(np.float32)

                # Stack as (R, G, B) in FITS first-dimension channel order
                RGB = np.stack((red, green, blue))
                hdu = fits.PrimaryHDU(data=RGB, header=header)
                hdu.writeto(outputFile, overwrite=True)
                if two_checked:
                    self.statusLabel.setText("Combine Tricolor (FITS) completed using computed Blue = 0.5*R + 0.5*G.")
                else:
                    self.statusLabel.setText("Combine Tricolor (FITS) completed successfully.")
            else:
                # Other (image) mode
                # Read red and green (required)
                red_img = cv2.imread(redFile, cv2.IMREAD_UNCHANGED)
                green_img = cv2.imread(greenFile, cv2.IMREAD_UNCHANGED)
                if red_img is None or green_img is None:
                    self.statusLabel.setText("Error reading Red or Green input images (Other).")
                    return

                # Extract single-channel intensity arrays for R and G
                def extract_channel(img):
                    if img.ndim == 2:
                        return img.astype(np.float64)
                    elif img.ndim == 3 and img.shape[2] >= 3:
                        # assume BGR; user provided red/green files may be single-channel or color
                        # For red file, prefer channel 2; for green file, prefer channel 1
                        return img[:, :, 0].astype(np.float64) if img.shape[2] == 1 else img[:, :, 0].astype(np.float64)
                    elif img.ndim == 3 and img.shape[2] == 1:
                        return img[:, :, 0].astype(np.float64)
                    else:
                        return img[:, :, 0].astype(np.float64)

                R = extract_channel(red_img)
                G = extract_channel(green_img)

                if two_checked:
                    B = 0.5 * R + 0.5 * G
                else:
                    if not blueFile:
                        self.statusLabel.setText("Blue file required when 'two' is not checked.")
                        return
                    blue_img = cv2.imread(blueFile, cv2.IMREAD_UNCHANGED)
                    if blue_img is None:
                        self.statusLabel.setText("Error reading Blue input image (Other).")
                        return
                    B = extract_channel(blue_img)

                # Prepare merged image in B,G,R order for OpenCV
                # Determine output dtype based on inputs (prefer integer if inputs are integer)
                dtype = red_img.dtype
                # Clip and cast appropriately
                if np.issubdtype(dtype, np.integer):
                    info = np.iinfo(dtype)
                    R_out = np.clip(R, info.min, info.max).astype(dtype)
                    G_out = np.clip(G, info.min, info.max).astype(dtype)
                    B_out = np.clip(B, info.min, info.max).astype(dtype)
                else:
                    R_out = R.astype(np.float32)
                    G_out = G.astype(np.float32)
                    B_out = B.astype(np.float32)

                merged = cv2.merge((B_out, G_out, R_out))
                ok = cv2.imwrite(outputFile, merged)
                if not ok:
                    self.statusLabel.setText("Failed to write output image (Other).")
                    return
                if two_checked:
                    self.statusLabel.setText("Combine Tricolor (Other) completed using computed Blue = 0.5*R + 0.5*G.")
                else:
                    self.statusLabel.setText("Combine Tricolor (Other) completed successfully.")
        except Exception as e:
            self.statusLabel.setText(f"Error in Combine Tricolor: {e}")

    def runCreateLuminance(self):
        inputFile = self.lumInputLine.text().strip()
        outputFile = self.lumOutputLine.text().strip()
        mode = self.lumModeCombo.currentText()
        eq_label = self.lumEqCombo.currentText()
        if not (inputFile and outputFile):
            self.statusLabel.setText("Both input and output files are required for Create Luminance.")
            return

        # Map selected equation to weights (R, G, B)
        if eq_label.startswith("synthetic-luminance"):
            wR, wG, wB = 1.0, 1.0, 1.0
        elif eq_label.startswith("Human per(709)"):
            wR, wG, wB = 0.2126, 0.7152, 0.0722
        elif eq_label.startswith("Bayer-one shot"):
            wR, wG, wB = 1.0, 2.0, 1.0
        else:
            # fallback to standard Rec.709
            wR, wG, wB = 0.2126, 0.7152, 0.0722

        try:
            if mode == "FITS":
                hdul = fits.open(inputFile)
                data = hdul[0].data
                header = hdul[0].header
                hdul.close()
                # Accept either channel-first (3, H, W) or channel-last (H, W, 3)
                if data.ndim == 3 and data.shape[0] == 3:
                    img = np.transpose(data, (1, 2, 0))  # to H,W,3
                elif data.ndim == 3 and data.shape[2] == 3:
                    img = data
                else:
                    self.statusLabel.setText("Unexpected FITS channel layout for luminance.")
                    return

                # Ensure float for computation
                R = img[:, :, 0].astype(np.float64)
                G = img[:, :, 1].astype(np.float64)
                B = img[:, :, 2].astype(np.float64)

                luminance = (wR * R) + (wG * G) + (wB * B)

                # Write luminance as float FITS (preserve header where possible)
                fits.PrimaryHDU(data=luminance, header=header).writeto(outputFile, overwrite=True)
                self.statusLabel.setText(f"Create Luminance (FITS) completed using: {eq_label}")

            else:
                img = cv2.imread(inputFile, cv2.IMREAD_UNCHANGED)
                if img is None:
                    self.statusLabel.setText("Error reading input image (Other).")
                    return

                # If grayscale already, just write it out
                if img.ndim == 2:
                    gray = img
                else:
                    # OpenCV loads as BGR
                    img_f = img.astype(np.float64)
                    # split channels B,G,R
                    if img_f.shape[2] >= 3:
                        B = img_f[:, :, 0]
                        G = img_f[:, :, 1]
                        R = img_f[:, :, 2]
                    else:
                        # unexpected channel count
                        self.statusLabel.setText("Unexpected channel count in input image.")
                        return

                    luminance = (wR * R) + (wG * G) + (wB * B)

                    # To avoid overflow/clipping issues, scale/clamp to original dtype range.
                    dtype = img.dtype
                    if np.issubdtype(dtype, np.integer):
                        # determine max for integer type
                        info = np.iinfo(dtype)
                        # If weights sum > 1, luminance may exceed original range; clip
                        luminance = np.clip(luminance, info.min, info.max)
                        gray = luminance.astype(dtype)
                    else:
                        # float image: keep float32
                        gray = luminance.astype(np.float32)

                ok = cv2.imwrite(outputFile, gray)
                if not ok:
                    self.statusLabel.setText("Failed to write output image (Other).")
                    return
                self.statusLabel.setText(f"Create Luminance (Other) completed using: {eq_label}")

        except Exception as e:
            self.statusLabel.setText(f"Error in Create Luminance: {e}")

def main():
    app = QApplication(sys.argv)
    window = ColorWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()