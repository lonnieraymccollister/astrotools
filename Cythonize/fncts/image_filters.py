#!/usr/bin/env python3
"""
image_filters.py
PyQt6 GUI to run a set of FITS image filters (Unsharp, RL deconv, FFT high-pass, morphology, Gaussian, HpMore, LocAdapt).
This is a cleaned, runnable version of your code.
"""

import sys
import os
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
    QStackedWidget
)
from PyQt6.QtCore import Qt

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

    # Page 6: HpMore (High-pass More)
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
        hdul = fits.open(inp)
        header = hdul[0].header
        data = hdul[0].data.astype(np.float64)
        hdul.close()
        if data.ndim == 2:
            blurred = cv2.GaussianBlur(data, (0, 0), 2.0)
            result = cv2.addWeighted(data, 2.0, blurred, -1.0, 0)
        elif data.ndim == 3:
            res_channels = []
            for c in range(data.shape[2]):
                channel = data[:, :, c]
                blurred = cv2.GaussianBlur(channel, (0, 0), 2.0)
                res_channels.append(cv2.addWeighted(channel, 2.0, blurred, -1.0, 0))
            result = np.stack(res_channels, axis=2)
        else:
            self.statusLabel.setText("Unsupported dimensions for Unsharp Mask.")
            return
        fits.writeto(outp, result.astype(np.float64), header, overwrite=True)
        self.statusLabel.setText("Unsharp Mask saved to " + outp)

    def runLrDeconv(self):
        inp = self.lrInput.text().strip()
        outp = self.lrOutput.text().strip()
        iters = int(self.lrIter.text().strip())
        psf_mode = self.lrPsfMode.currentText()

        hdul = fits.open(inp)
        header = hdul[0].header
        image = hdul[0].data.astype(np.float64)
        hdul.close()

        from astropy.convolution import convolve_fft
        def richardson_lucy(im, psf, iterations):
            im = im.astype(np.float64)
            im_est = im.copy()
            psf_mirror = psf[::-1, ::-1]
            for i in range(iterations):
                conv_est = convolve_fft(im_est, psf, normalize_kernel=True)
                conv_est[conv_est == 0] = 1e-7
                relative_blur = im / conv_est
                correction = convolve_fft(relative_blur, psf_mirror, normalize_kernel=True)
                im_est *= correction
            return im_est

        if psf_mode == "Analytical":
            sigma = 2.0
            ks = 25
            ax = np.linspace(-(ks-1)/2., (ks-1)/2., ks)
            xx, yy = np.meshgrid(ax, ax)
            psf = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
            psf /= psf.sum()
        else:
            x = float(self.lrPsfX.text().strip())
            y = float(self.lrPsfY.text().strip())
            size = int(self.lrPsfSize.text().strip())
            cutout = Cutout2D(image, (x, y), size)
            psf = cutout.data.copy()
            psf -= np.median(psf)
            psf[psf < 0] = 0
            if psf.sum() != 0:
                psf /= psf.sum()

        if image.ndim == 2:
            deconv = richardson_lucy(image, psf, iters)
        elif image.ndim == 3:
            deconv_channels = [richardson_lucy(image[:,:,c], psf, iters) for c in range(image.shape[2])]
            deconv = np.stack(deconv_channels, axis=2)
        else:
            self.statusLabel.setText("Unsupported dimensions for LrDeconv.")
            return

        fits.writeto(outp, deconv.astype(np.float64), header, overwrite=True)
        self.statusLabel.setText("LrDeconv saved to " + outp)

    def runFFT(self):
        inp = self.fftInput.text().strip()
        outp = self.fftOutput.text().strip()
        cutoff = float(self.fftCutoff.text().strip())
        weight = float(self.fftWeight.text().strip())
        denom = float(self.fftDenom.text().strip())
        radius = int(self.fftRadius.text().strip())
        cutoff2 = float(self.fftSecondCutoff.text().strip())
        hdul = fits.open(inp)
        header = hdul[0].header
        data = hdul[0].data.astype(np.float64)
        hdul.close()

        if data.ndim == 3:
            if data.shape[0] != 3 and data.shape[-1] == 3:
                data = np.transpose(data, (2, 0, 1))
        else:
            self.statusLabel.setText("FFT requires a 3-channel image.")
            return

        b, g, r = data[0], data[1], data[2]

        def high_pass_filter(im, cutoff_frac, weight_frac):
            fft = np.fft.fft2(im)
            fft_shift = np.fft.fftshift(fft)
            rows, cols = im.shape
            crow, ccol = rows // 2, cols // 2
            rad = int(cutoff_frac * min(rows, cols))
            mask = np.ones((rows, cols), np.float32)
            mask[max(0, crow-rad):min(rows, crow+rad), max(0, ccol-rad):min(cols, ccol+rad)] = 0
            fft_shift_filtered = fft_shift * mask
            fft_inverse = np.fft.ifftshift(fft_shift_filtered)
            im_filt = np.abs(np.fft.ifft2(fft_inverse))
            im_weighted = cv2.addWeighted(im.astype(np.float32), 1 - weight_frac, im_filt.astype(np.float32), weight_frac, 0)
            return im_weighted

        def feather_image(im, radius_px, distance):
            r_px = max(1, radius_px | 1)  # ensure odd
            im_blur = cv2.GaussianBlur(im, (radius_px, radius_px), 0) if radius_px > 1 else im
            mask = np.full(im.shape, 255, dtype=np.uint8)
            k = max(1, int(distance)*2+1)
            mask_blur = cv2.GaussianBlur(mask, (k, k), 0)
            return (im_blur * (mask_blur / 255.0)).astype(im.dtype)

        def process_channel(ch):
            ch_norm = np.interp(ch, (ch.min(), ch.max()), (0, 1)).astype(np.float32)
            filtered = high_pass_filter(ch_norm, cutoff/denom, weight/denom)
            return feather_image(filtered, radius, int(cutoff2))

        b_proc = process_channel(b)
        g_proc = process_channel(g)
        r_proc = process_channel(r)
        processed = np.stack([b_proc, g_proc, r_proc], axis=0)
        fits.writeto(outp, processed.astype(np.float64), header, overwrite=True)
        self.statusLabel.setText("FFT saved to " + outp)

    def runErosion(self):
        inp = self.erInput.text().strip()
        outp = self.erOutput.text().strip()
        iters = int(self.erIter.text().strip())
        ksize = int(self.erKernel.text().strip())
        hdul = fits.open(inp)
        header = hdul[0].header
        data = hdul[0].data.astype(np.float64)
        hdul.close()
        if data.ndim == 3:
            res_channels = []
            for c in range(data.shape[2]):
                channel = data[:, :, c]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                res_channels.append(cv2.erode(channel, kernel, iterations=iters))
            result = np.stack(res_channels, axis=2)
        elif data.ndim == 2:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            result = cv2.erode(data, kernel, iterations=iters)
        else:
            self.statusLabel.setText("Unsupported dimensions in erosion.")
            return
        fits.writeto(outp, result.astype(np.float64), header, overwrite=True)
        self.statusLabel.setText("Erosion saved to " + outp)

    def runDilation(self):
        inp = self.diInput.text().strip()
        outp = self.diOutput.text().strip()
        iters = int(self.diIter.text().strip())
        ksize = int(self.diKernel.text().strip())
        hdul = fits.open(inp)
        header = hdul[0].header
        data = hdul[0].data.astype(np.float64)
        hdul.close()
        if data.ndim == 3:
            res_channels = []
            for c in range(data.shape[2]):
                channel = data[:, :, c]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                res_channels.append(cv2.dilate(channel, kernel, iterations=iters))
            result = np.stack(res_channels, axis=2)
        elif data.ndim == 2:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            result = cv2.dilate(data, kernel, iterations=iters)
        else:
            self.statusLabel.setText("Unsupported dimensions in dilation.")
            return
        fits.writeto(outp, result.astype(np.float64), header, overwrite=True)
        self.statusLabel.setText("Dilation saved to " + outp)

    def runGaussian(self):
        inp = self.gaInput.text().strip()
        outp = self.gaOutput.text().strip()
        sigma = float(self.gaSigma.text().strip())
        hdul = fits.open(inp)
        header = hdul[0].header
        data = hdul[0].data.astype(np.float64)
        hdul.close()
        if data.ndim == 3:
            res_channels = []
            for c in range(data.shape[2]):
                channel = data[:, :, c]
                norm = np.interp(channel, (channel.min(), channel.max()), (0, 1)).astype(np.float32)
                res_channels.append(cv2.GaussianBlur(norm, (0,0), sigmaX=sigma, sigmaY=sigma))
            result = np.stack(res_channels, axis=2)
        elif data.ndim == 2:
            norm = np.interp(data, (data.min(), data.max()), (0, 1)).astype(np.float32)
            result = cv2.GaussianBlur(norm, (0,0), sigmaX=sigma, sigmaY=sigma)
        else:
            self.statusLabel.setText("Unsupported dimensions in Gaussian.")
            return
        fits.writeto(outp, result.astype(np.float64), header, overwrite=True)
        self.statusLabel.setText("Gaussian blur saved to " + outp)

    def runHpMore(self):
        inp = self.hpInput.text().strip()
        outp = self.hpOutput.text().strip()
        hdul = fits.open(inp)
        data = hdul[0].data.astype(np.float64)
        hdul.close()
        if data.ndim == 3:
            if data.shape[-1] == 3:
                data = np.transpose(data, (2, 0, 1))
            elif data.shape[0] == 3:
                pass
            else:
                self.statusLabel.setText("Unexpected color image shape in HpMore.")
                return
        else:
            self.statusLabel.setText("Input FITS is not a 3-channel image in HpMore.")
            return
        hp_kernel_5x5 = np.array([
            [-1, -1, -1, -1, -1],
            [-1,  1,  2,  1, -1],
            [-1,  2,  4,  2, -1],
            [-1,  1,  2,  1, -1],
            [-1, -1, -1, -1, -1]
        ], dtype=float)
        filtered_channels = np.empty_like(data)
        for i in range(data.shape[0]):
            channel = data[i]
            highpass = convolve2d(channel, hp_kernel_5x5, mode='same', boundary='symm')
            threshold = np.percentile(channel, 70)
            mask = channel > threshold
            filtered_channel = channel.copy()
            filtered_channel[mask] = channel[mask] + highpass[mask]
            filtered_channels[i] = filtered_channel
        fits.writeto(outp, filtered_channels.astype(np.float64), overwrite=True)
        self.statusLabel.setText("HpMore saved to " + outp)

    def runLocAdapt(self):
        inp = self.laInput.text().strip()
        outp = self.laOutput.text().strip()
        neigh = int(self.laNeigh.text().strip())
        target_std = float(self.laContrast.text().strip())
        feather_dist = float(self.laFeather.text().strip())
        hdul = fits.open(inp)
        header = hdul[0].header
        data = hdul[0].data.astype(float)
        hdul.close()
        if data.ndim == 3 and data.shape[0] == 3:
            channel = data[0]
        elif data.ndim == 2:
            channel = data
        else:
            self.statusLabel.setText("Unsupported dimensions for LocAdapt.")
            return

        def compute_optimum_contrast_percentage(image, target_std):
            current_std = np.std(image)
            if current_std == 0:
                return 100.0
            return (target_std / current_std) * 100.0

        def compute_optimum_feather_distance(image, neighborhood_size, factor=1.0):
            adjusted_size = max(1, neighborhood_size - 1)
            kernel = np.ones((adjusted_size, adjusted_size), dtype=np.float32) / (adjusted_size*adjusted_size)
            local_mean = nd_convolve(image, kernel, mode='reflect')
            local_mean_sq = nd_convolve(image**2, kernel, mode='reflect')
            local_std = np.sqrt(np.abs(local_mean_sq - local_mean**2))
            return factor * np.median(local_std)

        def contrast_filter(image, neighborhood_size, contrast_factor, feather_distance):
            adjusted_size = max(1, neighborhood_size - 1)
            kernel = np.ones((adjusted_size, adjusted_size), dtype=float)
            kernel /= kernel.size
            local_mean = nd_convolve(image, kernel, mode='reflect')
            enhanced = (image - local_mean) * contrast_factor + local_mean
            squared = np.square(image)
            local_mean_sq = nd_convolve(squared, kernel, mode='reflect')
            local_std = np.sqrt(np.abs(local_mean_sq - np.square(local_mean)))
            weight = np.clip(local_std / max(1e-9, feather_distance), 0, 1)
            return weight * enhanced + (1 - weight) * image

        opt_contrast = compute_optimum_contrast_percentage(channel, target_std)
        contrast_factor = opt_contrast / 100.0
        opt_feather = compute_optimum_feather_distance(channel, neigh, feather_dist)
        fd = max(1e-9, opt_feather / 100.0)
        enhanced = contrast_filter(channel, neigh, contrast_factor, fd)
        if data.ndim == 3 and data.shape[0] == 3:
            data[0] = enhanced
            result = data
        else:
            result = enhanced
        fits.writeto(outp, result.astype(np.float64), header, overwrite=True)
        self.statusLabel.setText("LocAdapt saved to " + outp)

# Main
def main():
    app = QApplication(sys.argv)
    window = ImageFiltersWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
