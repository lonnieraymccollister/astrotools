#!/usr/bin/env python3
"""
affine_transform.py
Standalone PyQt6 app: Automatic, Manual, FITS modes (loads/saves images, converts 16-bit PNG -> 32-bit FITS).
"""

import sys
from pathlib import Path

import numpy as np
import cv2
from astropy.io import fits

from PyQt6 import QtWidgets, QtGui, QtCore

# ---------- helper ---------------------------------------------------------
def png16_to_fits32(png_filename, fits_filename):
    data = cv2.imread(str(png_filename), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise IOError(f"Could not read image file: {png_filename}")
    if data.dtype != np.uint16:
        raise TypeError(f"Expected a 16-bit image (dtype=np.uint16), but got {data.dtype}")
    if data.ndim == 3 and data.shape[2] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = np.transpose(data, (2, 0, 1))
    data_32 = data.astype(np.float32)
    fits.writeto(str(fits_filename), data_32, overwrite=True)

# ---------- ImageLabel -----------------------------------------------------
class ImageLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(QtCore.QPoint)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.setMouseTracking(True)
        self.points = []
        self.display_image = None

    def setImage(self, cv_img):
        if cv_img is None:
            return
        # Ensure image is BGR uint8/uint16 -> convert to RGB uint8 for display
        img = cv_img
        if img.dtype == np.uint16:
            # scale to uint8 for display (preserve original for processing)
            disp = (img >> 8).astype(np.uint8)
        else:
            disp = img.astype(np.uint8)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB) if disp.ndim == 3 else cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qImg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qImg)
        self.display_image = pix
        self.setPixmap(self.display_image)
        self.points = []

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if self.pixmap() is None:
            return
        point = event.pos()
        self.points.append(point)
        pix = self.pixmap().copy()
        painter = QtGui.QPainter(pix)
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.red)
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawEllipse(point, 6, 6)
        painter.end()
        self.setPixmap(pix)
        self.clicked.emit(point)

# ---------- AutoWidget (unchanged logic but small safety) -------------------
class AutoWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        images_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(images_layout)

        self.refLabel = QtWidgets.QLabel("Reference Image", self)
        self.refLabel.setFixedSize(400, 400)
        self.refLabel.setFrameShape(QtWidgets.QFrame.Shape.Box)
        images_layout.addWidget(self.refLabel)

        self.alignLabel = QtWidgets.QLabel("Alignment Image", self)
        self.alignLabel.setFixedSize(400, 400)
        self.alignLabel.setFrameShape(QtWidgets.QFrame.Shape.Box)
        images_layout.addWidget(self.alignLabel)

        self.resultLabel = QtWidgets.QLabel("Result", self)
        self.resultLabel.setFixedSize(400, 400)
        self.resultLabel.setFrameShape(QtWidgets.QFrame.Shape.Box)
        layout.addWidget(self.resultLabel)

        btn_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_layout)
        self.btnLoadRef = QtWidgets.QPushButton("Load Reference Image")
        self.btnLoadAlign = QtWidgets.QPushButton("Load Alignment Image")
        self.btnCompute  = QtWidgets.QPushButton("Compute Homography")
        self.btnSave     = QtWidgets.QPushButton("Save Result")
        btn_layout.addWidget(self.btnLoadRef)
        btn_layout.addWidget(self.btnLoadAlign)
        btn_layout.addWidget(self.btnCompute)
        btn_layout.addWidget(self.btnSave)

        self.btnLoadRef.clicked.connect(self.loadReferenceImage)
        self.btnLoadAlign.clicked.connect(self.loadAlignmentImage)
        self.btnCompute.clicked.connect(self.computeHomography)
        self.btnSave.clicked.connect(self.saveResult)

        self.referenceImage = None
        self.alignmentImage = None
        self.resultImage = None

    def displayImage(self, label, img):
        if img is None:
            return
        rgb = cv2.cvtColor((img.astype(np.uint8) if img.dtype != np.uint8 else img), cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        qImg = QtGui.QImage(rgb.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        pixmap = pixmap.scaled(label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(pixmap)

    def loadReferenceImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Reference Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)")
        if fileName:
            img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
            if img is None:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to load reference image!")
                return
            self.referenceImage = img
            self.displayImage(self.refLabel, img)

    def loadAlignmentImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Alignment Image", "", "Images (*.png *.jpg *.bmp);;All Files (*)")
        if fileName:
            img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
            if img is None:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to load alignment image!")
                return
            self.alignmentImage = img
            self.displayImage(self.alignLabel, img)

    def computeHomography(self):
        if self.referenceImage is None or self.alignmentImage is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "Load both images first!")
            return
        ref_gray = cv2.cvtColor((self.referenceImage.astype(np.uint8) if self.referenceImage.dtype != np.uint8 else self.referenceImage), cv2.COLOR_BGR2GRAY)
        align_gray = cv2.cvtColor((self.alignmentImage.astype(np.uint8) if self.alignmentImage.dtype != np.uint8 else self.alignmentImage), cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(ref_gray, None)
        kp2, des2 = orb.detectAndCompute(align_gray, None)
        if des1 is None or des2 is None:
            QtWidgets.QMessageBox.warning(self, "Error", "ORB features not detected!")
            return
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 4:
            QtWidgets.QMessageBox.warning(self, "Error", "Not enough matches to compute homography!")
            return
        num_good_matches = max(4, int(len(matches) * 0.9))
        good_matches = matches[:num_good_matches]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Homography computation failed!")
            return
        height, width = self.alignmentImage.shape[:2]
        warped = cv2.warpPerspective(self.referenceImage, homography, (width, height))
        self.resultImage = warped
        self.displayImage(self.resultLabel, warped)

    def saveResult(self):
        if self.resultImage is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Compute the homography first!")
            return
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Result Image", "", "16-bit PNG Files (*.png);;All Files (*)")
        if fileName:
            out = self.resultImage
            if out.dtype == np.uint8:
                out16 = (out.astype(np.uint16)) * 257
            else:
                out16 = out
            ok = cv2.imwrite(fileName, out16)
            if ok:
                fits_filename = str(Path(fileName).with_suffix('.fits'))
                rgb_result = cv2.cvtColor(out16, cv2.COLOR_BGR2RGB)
                fits_data = rgb_result.astype(np.float32)
                fits.writeto(fits_filename, fits_data, overwrite=True)
                QtWidgets.QMessageBox.information(self, "Success", f"Saved:\n{fileName}\n{fits_filename}")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to save image!")

# ---------- ManualWidget ---------------------------------------------------
class ManualWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_image = None
        self.target_image = None
        self.src_points = []
        self.dst_points = []
        self.transformed = None

        self.srcLabel = ImageLabel()
        self.dstLabel = ImageLabel()
        self.resultLabel = QtWidgets.QLabel("Result image will appear here")
        self.resultLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)

        self.srcScrollArea = QtWidgets.QScrollArea()
        self.srcScrollArea.setWidget(self.srcLabel)
        self.srcScrollArea.setWidgetResizable(True)
        self.srcScrollArea.setFixedSize(400,400)

        self.dstScrollArea = QtWidgets.QScrollArea()
        self.dstScrollArea.setWidget(self.dstLabel)
        self.dstScrollArea.setWidgetResizable(True)
        self.dstScrollArea.setFixedSize(400,400)

        self.resultScrollArea = QtWidgets.QScrollArea()
        self.resultScrollArea.setWidget(self.resultLabel)
        self.resultScrollArea.setWidgetResizable(True)
        self.resultScrollArea.setFixedSize(400,400)

        self.loadSrcButton = QtWidgets.QPushButton("Load Original Mask")
        self.loadDstButton = QtWidgets.QPushButton("Load Comparison Image")
        self.transformButton = QtWidgets.QPushButton("Compute Affine Transform")
        self.clearPointsButton = QtWidgets.QPushButton("Clear Points")
        self.saveButton = QtWidgets.QPushButton("Save Result")

        self.outXEdit = QtWidgets.QLineEdit()
        self.outXEdit.setPlaceholderText("Output width (e.g., 800)")
        self.outYEdit = QtWidgets.QLineEdit()
        self.outYEdit.setPlaceholderText("Output height (e.g., 600)")
        self.outFileEdit = QtWidgets.QLineEdit()
        self.outFileEdit.setPlaceholderText("Output file name (e.g., new_mask.jpg)")

        srcGroup = QtWidgets.QGroupBox("Original Mask Image (Click 3 Points)")
        srcLayout = QtWidgets.QVBoxLayout()
        srcLayout.addWidget(self.srcScrollArea)
        srcLayout.addWidget(self.loadSrcButton)
        srcGroup.setLayout(srcLayout)

        dstGroup = QtWidgets.QGroupBox("Comparison Image (Click 3 Points)")
        dstLayout = QtWidgets.QVBoxLayout()
        dstLayout.addWidget(self.dstScrollArea)
        dstLayout.addWidget(self.loadDstButton)
        dstGroup.setLayout(dstLayout)

        resultGroup = QtWidgets.QGroupBox("Transformed Result")
        resultLayout = QtWidgets.QVBoxLayout()
        resultLayout.addWidget(self.resultScrollArea)
        resultGroup.setLayout(resultLayout)

        topLayout = QtWidgets.QHBoxLayout()
        topLayout.addWidget(srcGroup)
        topLayout.addWidget(dstGroup)
        topLayout.addWidget(resultGroup)

        controlLayout = QtWidgets.QHBoxLayout()
        controlLayout.addWidget(self.transformButton)
        controlLayout.addWidget(self.clearPointsButton)
        controlLayout.addWidget(self.outXEdit)
        controlLayout.addWidget(self.outYEdit)
        controlLayout.addWidget(self.outFileEdit)
        controlLayout.addWidget(self.saveButton)

        mainLayout = QtWidgets.QVBoxLayout()
        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(controlLayout)
        self.setLayout(mainLayout)

        self.srcLabel.clicked.connect(self.recordSrcPoint)
        self.dstLabel.clicked.connect(self.recordDstPoint)
        self.loadSrcButton.clicked.connect(self.loadSourceImage)
        self.loadDstButton.clicked.connect(self.loadTargetImage)
        self.transformButton.clicked.connect(self.computeAffine)
        self.clearPointsButton.clicked.connect(self.clearPoints)
        self.saveButton.clicked.connect(self.saveResult)

    def loadSourceImage(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Original Mask Image", "", "Image Files (*.png *.jpg *.bmp)")
        if filename:
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            if img is None:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to load image.")
                return
            self.source_image = img
            self.srcLabel.setImage(img)
            self.src_points = []

    def loadTargetImage(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Comparison Image", "", "Image Files (*.png *.jpg *.bmp)")
        if filename:
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            if img is None:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to load image.")
                return
            self.target_image = img
            self.dstLabel.setImage(img)
            self.dst_points = []

    def recordSrcPoint(self, point):
        x, y = point.x(), point.y()
        self.src_points.append((x, y))
        print("Source point:", x, y)
        if len(self.src_points) > 3:
            self.src_points = self.src_points[:3]

    def recordDstPoint(self, point):
        x, y = point.x(), point.y()
        self.dst_points.append((x, y))
        print("Destination point:", x, y)
        if len(self.dst_points) > 3:
            self.dst_points = self.dst_points[:3]

    def clearPoints(self):
        self.src_points = []
        self.dst_points = []
        if self.source_image is not None:
            self.srcLabel.setImage(self.source_image)
        if self.target_image is not None:
            self.dstLabel.setImage(self.target_image)

    def computeAffine(self):
        if self.source_image is None or self.target_image is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Load both images before computing transformation.")
            return
        if len(self.src_points) < 3 or len(self.dst_points) < 3:
            QtWidgets.QMessageBox.warning(self, "Error", "Click 3 points on each image.")
            return
        pts1 = np.float32(self.src_points[:3])
        pts2 = np.float32(self.dst_points[:3])
        M = cv2.getAffineTransform(pts1, pts2)
        try:
            out_w = int(self.outXEdit.text())
            out_h = int(self.outYEdit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid output dimensions.")
            return
        transformed = cv2.warpAffine(self.source_image, M, (out_w, out_h))
        self.transformed = transformed
        rgb = cv2.cvtColor(transformed.astype(np.uint8), cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qImg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        pixmap = pixmap.scaled(self.resultLabel.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.resultLabel.setPixmap(pixmap)

    def saveResult(self):
        if self.transformed is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Compute the transformation first.")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Result Image", "", "16-bit PNG Files (*.png);;All Files (*)")
        if filename:
            out = self.transformed
            out16 = (out.astype(np.uint16) * 257) if out.dtype == np.uint8 else out
            ok = cv2.imwrite(filename, out16)
            if ok:
                fits_filename = str(Path(filename).with_suffix('.fits'))
                rgb_result = cv2.cvtColor(out16, cv2.COLOR_BGR2RGB)
                fits_data = rgb_result.astype(np.float32)
                fits.writeto(fits_filename, fits_data, overwrite=True)
                QtWidgets.QMessageBox.information(self, "Saved", f"Saved:\n{filename}\n{fits_filename}")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Failed to save the image!")

# ---------- FitsWidget ----------------------------------------------------
class FitsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.btnLoadPNG = QtWidgets.QPushButton("Load 16-bit PNG")
        self.fileLineEdit = QtWidgets.QLineEdit()
        self.fileLineEdit.setReadOnly(True)
        self.btnConvertFITS = QtWidgets.QPushButton("Convert & Save as 32-bit FITS")
        layout.addWidget(self.btnLoadPNG)
        layout.addWidget(self.fileLineEdit)
        layout.addWidget(self.btnConvertFITS)
        self.btnLoadPNG.clicked.connect(self.load_png)
        self.btnConvertFITS.clicked.connect(self.convert_fits)
        self.input_png = None

    def load_png(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select 16-bit PNG", "", "PNG Files (*.png);;All Files (*)")
        if fileName:
            self.input_png = fileName
            self.fileLineEdit.setText(fileName)

    def convert_fits(self):
        if not self.input_png:
            QtWidgets.QMessageBox.warning(self, "Error", "No file loaded!")
            return
        fits_filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save 32-bit FITS", "", "FITS Files (*.fits);;All Files (*)")
        if fits_filename:
            try:
                png16_to_fits32(self.input_png, fits_filename)
                QtWidgets.QMessageBox.information(self, "Success", f"Saved 32-bit FITS file:\n{fits_filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))

# ---------- MainWindow ----------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Transformation: Automatic, Manual, and FITS Modes")
        self.resize(1000, 800)
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        self.modeComboBox = QtWidgets.QComboBox()
        self.modeComboBox.addItem("Automatic")
        self.modeComboBox.addItem("Manual")
        self.modeComboBox.addItem("FITS")
        main_layout.addWidget(self.modeComboBox)
        self.stackedWidget = QtWidgets.QStackedWidget()
        self.autoWidget = AutoWidget()
        self.manualWidget = ManualWidget()
        self.fitsWidget = FitsWidget()
        self.stackedWidget.addWidget(self.autoWidget)
        self.stackedWidget.addWidget(self.manualWidget)
        self.stackedWidget.addWidget(self.fitsWidget)
        main_layout.addWidget(self.stackedWidget)
        self.modeComboBox.currentIndexChanged.connect(self.switchMode)

    def switchMode(self, index):
        self.stackedWidget.setCurrentIndex(index)

# ---------- entry point ---------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()