# some code from copilot
# import required libraries
import fnmatch
from PIL import Image
import cv2, sys, os, subprocess, shutil, ffmpeg, tifffile, copy
import numpy as np
import matplotlib
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits
from astropy.convolution import convolve_fft
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.wcs import WCS
from astropy.coordinates import Angle
from astropy.stats import sigma_clipped_stats
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils.detection import DAOStarFinder
import mpmath as mp
import glob
from skimage.exposure import match_histograms
from scipy.special import erfinv
from scipy.ndimage import zoom 
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from reproject import reproject_interp, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs
from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
print("Qt: v", QT_VERSION_STR, "\tPyQt: v", PYQT_VERSION_STR)
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QLineEdit, QMessageBox, QGroupBox, QScrollArea,
    QComboBox, QGridLayout, QStackedWidget,
    QFormLayout
)
from PyQt6.QtGui import QDoubleValidator
from warpaffinemaskrescale import warp_affine_mask_rescale

def AffineTransform():

  try:
           
      #############################################
      # Function to Convert a 16-bit PNG to 32-bit FITS
      #############################################
      def png16_to_fits32(png_filename, fits_filename):
          """
          Reads a 16-bit PNG image using OpenCV and saves it as a 32-bit FITS file.
          """
          # Read the image as-is, preserving the 16-bit depth.
          data = cv2.imread(png_filename, cv2.IMREAD_UNCHANGED)
          if data is None:
              raise IOError(f"Could not read image file: {png_filename}")
          
          # Verify that the image data is 16-bit
          if data.dtype != np.uint16:
              raise TypeError(f"Expected a 16-bit image (dtype=np.uint16), but got {data.dtype}")
      
          # If the image has multiple channels, assume it is color.
          # OpenCV reads color images in BGR order. We convert them to RGB.
          if len(data.shape) == 3 and data.shape[2] == 3:
              data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
              # Transpose the data so that the color channels come first:
              # (height, width, channels) -> (channels, height, width)
              data = np.transpose(data, (2, 0, 1))
          
          # Convert the 16-bit unsigned integer data into 32-bit float.
          data_32 = data.astype(np.float32)
          
          # Write the 32-bit float array to a FITS file.
          fits.writeto(fits_filename, data_32, overwrite=True)
          print(f"Saved 32-bit FITS file: {fits_filename}")
      
      #############################################
      # Custom ImageLabel for manual mode that records click positions.
      #############################################
      class ImageLabel(QtWidgets.QLabel):
          clicked = QtCore.pyqtSignal(QtCore.QPoint)
          
          def __init__(self, parent=None):
              super().__init__(parent)
              # Use fully qualified flags in PyQt6.
              self.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
              self.setMouseTracking(True)
              self.points = []  # Stores clicked points
              self.display_image = None
      
          def setImage(self, cv_img):
              """
              Set the image from an OpenCV-like BGR NumPy array.
              """
              # If cv2 is present, convert BGR to RGB; otherwise assume it's already RGB.
              if "cv2" in globals():
                  rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
              else:
                  rgb_image = cv_img
              h, w, ch = rgb_image.shape
              bytes_per_line = ch * w
              qImg = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
              self.display_image = QtGui.QPixmap.fromImage(qImg)
              self.setPixmap(self.display_image)
              self.points = []
      
          def mousePressEvent(self, event: QtGui.QMouseEvent):
              if self.pixmap() is None:
                  return
              point = event.pos()
              self.points.append(point)
              # Draw a red circle centered at the click point.
              pix = self.pixmap().copy()
              painter = QtGui.QPainter(pix)
              painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.red, 5))
              painter.drawEllipse(point, 5, 5)
              painter.end()
              self.setPixmap(pix)
              self.clicked.emit(point)
      
      #############################################
      # Automatic Mode Widget (uses OpenCV for ORB/homography)
      #############################################
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
          
          def loadReferenceImage(self):
              options = QtWidgets.QFileDialog.Options()
              fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                  self, "Select Reference Image", "",
                  "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
              if fileName:
                  self.referenceImage = cv2.imread(fileName)
                  if self.referenceImage is None:
                      QtWidgets.QMessageBox.warning(self, "Error", "Failed to load reference image!")
                      return
                  self.displayImage(self.refLabel, self.referenceImage)
          
          def loadAlignmentImage(self):
              options = QtWidgets.QFileDialog.Options()
              fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                  self, "Select Alignment Image", "",
                  "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
              if fileName:
                  self.alignmentImage = cv2.imread(fileName)
                  if self.alignmentImage is None:
                      QtWidgets.QMessageBox.warning(self, "Error", "Failed to load alignment image!")
                      return
                  self.displayImage(self.alignLabel, self.alignmentImage)
          
          def computeHomography(self):
              if self.referenceImage is None or self.alignmentImage is None:
                  QtWidgets.QMessageBox.warning(self, "Warning", "Load both images first!")
                  return
              ref_gray = cv2.cvtColor(self.referenceImage, cv2.COLOR_BGR2GRAY)
              align_gray = cv2.cvtColor(self.alignmentImage, cv2.COLOR_BGR2GRAY)
              orb = cv2.ORB_create(5000)
              kp1, des1 = orb.detectAndCompute(ref_gray, None)
              kp2, des2 = orb.detectAndCompute(align_gray, None)
              if des1 is None or des2 is None:
                  QtWidgets.QMessageBox.warning(self, "Error", "ORB features not detected!")
                  return
              bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
              matches = bf.match(des1, des2)
              matches = sorted(matches, key=lambda x: x.distance)
              num_good_matches = int(len(matches) * 0.9)
              good_matches = matches[:num_good_matches]
              if len(good_matches) < 4:
                  QtWidgets.QMessageBox.warning(self, "Error", "Not enough matches to compute homography!")
                  return
              src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
              dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
              homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
              if homography is None:
                  QtWidgets.QMessageBox.warning(self, "Error", "Homography computation failed!")
                  return
              height, width, _ = self.alignmentImage.shape
              warped = cv2.warpPerspective(self.referenceImage, homography, (width, height))
              self.resultImage = warped
              self.displayImage(self.resultLabel, warped)
          
          def saveResult(self):
              if self.resultImage is None:
                  QtWidgets.QMessageBox.warning(self, "Error", "Compute the homography first!")
                  return
              options = QtWidgets.QFileDialog.Options()
              fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
                  self, "Save Result Image", "",
                  "16-bit PNG Files (*.png);;All Files (*)", options=options)
              if fileName:
                  if self.resultImage.dtype == np.uint8:
                      result16 = (self.resultImage.astype(np.uint16)) * 257
                  else:
                      result16 = self.resultImage
                  if cv2.imwrite(fileName, result16):
                      fits_filename = fileName.rsplit('.', 1)[0] + ".fits"
                      # Convert from BGR to RGB before writing to FITS.
                      rgb_result = cv2.cvtColor(result16, cv2.COLOR_BGR2RGB)
                      fits_data = rgb_result.astype(np.float32)
                      fits.writeto(fits_filename, fits_data, overwrite=True)
                      QtWidgets.QMessageBox.information(self, "Success", f"Saved:\n{fileName}\n{fits_filename}")
                  else:
                      QtWidgets.QMessageBox.warning(self, "Error", "Failed to save image!")
          
          def displayImage(self, label, img):
              rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              height, width, channels = rgb.shape
              bytes_per_line = channels * width
              qImg = QtGui.QImage(rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
              pixmap = QtGui.QPixmap.fromImage(qImg)
              # Use PyQt6 enum for aspect ratio mode:
              pixmap = pixmap.scaled(label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
              label.setPixmap(pixmap)
      
      #############################################
      # Manual Mode Widget (point-click affine transform)
      #############################################
      class ManualWidget(QtWidgets.QWidget):
          def __init__(self, parent=None):
              super().__init__(parent)
              self.source_image = None
              self.target_image = None
              self.src_points = []  # Three clicked points on source
              self.dst_points = []  # Three clicked points on target
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
              filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                  self, "Select Original Mask Image", "",
                  "Image Files (*.png *.jpg *.bmp)")
              if filename:
                  self.source_image = cv2.imread(filename, cv2.IMREAD_COLOR)
                  if self.source_image is None:
                      QtWidgets.QMessageBox.warning(self, "Error", "Failed to load image.")
                      return
                  self.srcLabel.setImage(self.source_image)
                  self.src_points = []
          
          def loadTargetImage(self):
              filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                  self, "Select Comparison Image", "",
                  "Image Files (*.png *.jpg *.bmp)")
              if filename:
                  self.target_image = cv2.imread(filename, cv2.IMREAD_COLOR)
                  if self.target_image is None:
                      QtWidgets.QMessageBox.warning(self, "Error", "Failed to load image.")
                      return
                  self.dstLabel.setImage(self.target_image)
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
              rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
              h, w, ch = rgb.shape
              bytes_per_line = ch * w
              qImg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
              pixmap = QtGui.QPixmap.fromImage(qImg)
              pixmap = pixmap.scaled(self.resultLabel.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
              self.resultLabel.setPixmap(pixmap)
          
          def saveResult(self):
              if self.transformed is None:
                  QtWidgets.QMessageBox.warning(self, "Error", "Compute the transformation first.")
                  return
              filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                  self, "Save Result Image", "", "16-bit PNG Files (*.png);;All Files (*)")
              if filename:
                  if self.transformed.dtype == np.uint8:
                      transformed16 = (self.transformed.astype(np.uint16)) * 257
                  else:
                      transformed16 = self.transformed
                  if cv2.imwrite(filename, transformed16):
                      fits_filename = filename.rsplit('.', 1)[0] + ".fits"
                      rgb_result = cv2.cvtColor(transformed16, cv2.COLOR_BGR2RGB)
                      fits_data = rgb_result.astype(np.float32)
                      fits.writeto(fits_filename, fits_data, overwrite=True)
                      QtWidgets.QMessageBox.information(self, "Saved", f"Saved:\n{filename}\n{fits_filename}")
                  else:
                      QtWidgets.QMessageBox.warning(self, "Error", "Failed to save the image!")
      
      #############################################
      # FITS Mode Widget: Uses Pillow (PIL) to read a 16-bit PNG
      # and then save it as a 32-bit FITS file.
      #############################################
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
              fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                  self, "Select 16-bit PNG", "", "PNG Files (*.png);;All Files (*)")
              if fileName:
                  self.input_png = fileName
                  self.fileLineEdit.setText(fileName)
          
          def convert_fits(self):
              if not self.input_png:
                  QtWidgets.QMessageBox.warning(self, "Error", "No file loaded!")
                  return
              fits_filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                  self, "Save 32-bit FITS", "", "FITS Files (*.fits);;All Files (*)")
              if fits_filename:
                  try:
                      png16_to_fits32(self.input_png, fits_filename)
                      QtWidgets.QMessageBox.information(self, "Success", f"Saved 32-bit FITS file:\n{fits_filename}")
                  except Exception as e:
                      QtWidgets.QMessageBox.critical(self, "Error", str(e))
      
      #############################################
      # Main Window: Contains a drop-down to select mode and a stacked widget.
      #############################################
      class MainWindow(QtWidgets.QMainWindow):
          def __init__(self):
              super().__init__()
              self.setWindowTitle("Image Transformation: Automatic, Manual, and FITS Modes")
              self.resize(1000, 800)
              
              central_widget = QtWidgets.QWidget(self)
              self.setCentralWidget(central_widget)
              main_layout = QtWidgets.QVBoxLayout(central_widget)
              
              # Drop-down for mode.
              self.modeComboBox = QtWidgets.QComboBox()
              self.modeComboBox.addItem("Automatic")
              self.modeComboBox.addItem("Manual")
              self.modeComboBox.addItem("FITS")
              main_layout.addWidget(self.modeComboBox)
              
              # Stacked widget for different mode widgets.
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
      
      #############################################
      # Run the application.
      #############################################
      def main():
          app = QtWidgets.QApplication(sys.argv)
          window = MainWindow()
          window.show()
          app.exec()
      
      if __name__ == '__main__':
          main()
            

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()    


def mask():
  try:
      
      def main():
          
                 sysargv1  = input("Enter the Image1  -->")
                 sysargv3  = input("Enter the Mask for image  white- and black  -->")
                 sysargv4  = input("Enter the filename of the masked image to save  -->")
           
                 image = cv2.imread(sysargv1, -1)
                 mask = cv2.imread(sysargv3, -1)
           
                 # Apply the mask to the image
                 masked_image = cv2.bitwise_and(image, mask)
                 cv2.imwrite(sysargv4, masked_image)
                 
      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def maskinvert():
  try:
      
      def main():
      
                 sysargv1  = input("Enter the mask Image  -->")
                 sysargv3  = input("Enter the output invert Mask -->")
           
                 sysargv7  = input("Enter 0 for fits or 1 for other file -->")
           
                 if sysargv7 == '0':
           
                   # Function to read FITS file and return data
                   with fits.open(sysargv1) as hdul:
                       data = hdul[0].data.astype(np.float64)
                       data_range = np.max(data) - np.min(data)
                       if data_range == 0:
                         normalized_data = np.zeros_like(data)  # or handle differently
                       else:
                           normalized_data = (data - np.min(data)) / data_range
           
                       #Invert the data
                       inverted_data = 1 - normalized_data
           
                       #Create a new HDU with the inverted data
                       hdu = fits.PrimaryHDU(inverted_data)
           
                       #Copy the header from the original HDU
                       hdu.header = hdul[0].header
           
                       #Create a new HDU list and save it to a new FITS file
                       new_hdul = fits.HDUList([hdu])
                       new_hdul.writeto(sysargv3, overwrite=True)
           
                   print("Inverted FITS file saved successfully!")
           
                 if sysargv7 == '1':
                   image = cv2.imread(sysargv1, -1)
                   # Apply the inverted mask to the image
                   masked_image = cv2.bitwise_not(image)
                   cv2.imwrite(sysargv3, masked_image)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def filecount():
  try:
      
      def main():
      
                 sysargv1  = input("Enter the  directory path from explorer  -->")
                 sysargv2  = input("Enter file type as (*.fit)  -->")
                 count = len(fnmatch.filter(os.listdir(sysargv1), sysargv2))
                 print('File Count:', count)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def resize():
  try:

      def main():

                 sysargv1  = input("Enter the Image to be resized(bicubic)(16bit uses .tif/.fit)  -->")
                 sysargv2  = input("Enter the scale(num)(1,2,3, or 4)(***Cubic only picels, gry and (x=y)***)  -->")
                 sysargv2a = input("Enter the scale(denom)(1,2,3, or 4)   -->")
                 sysargv3  = input("Enter the filename of the resized image to be saved(16bit uses .tif/.fit)  -->")    
                 sysargv7  = input("Enter 0 for fits cubic gry float64, 1 for fits LANCZOS4((Note cv2->RGB<-)32bit) or 2 for other file -->")
           
                 if sysargv7 == '0':
                   # Open the FITS file
                   fits_file = sysargv1
                   hdul = fits.open(fits_file)
                   data = hdul[0].data
           
                   # Ensure the data is in float64 format
                   data = data.astype(np.float64)
           
                   # Original dimensions of the image
                   original_height, original_width = data.shape
           
                   # Set the desired width while preserving the aspect ratio
                   target_width = int(sysargv2) / int(sysargv2a)  # Example: New width in pixels
                   scaling_factor = target_width / original_width
           
                   # Apply the same scaling factor to both dimensions
                   resized_data = zoom(data, (scaling_factor, scaling_factor), order=3)  # Cubic interpolation
           
                   # Ensure the resized data is still float64
                   resized_data = resized_data.astype(np.float64)
           
                   # Save the resized image to a new FITS file
                   hdu = fits.PrimaryHDU(resized_data)
                   hdu.writeto( sysargv3, overwrite=True)
           
                   print("Image resized using Cubic Interpolation (float64) and saved successfully!")
           
           
                 if sysargv7 == '1':
           
                   # Function to read FITS file and return data
                   def read_fits(file):
                       hdul = fits.open(file)
                       header = hdul[0].header
                       data = hdul[0].data
                       hdul.close()
                       return data, header
           
                   # Read the FITS files
                   file1 = sysargv1
           
                   # Read the image data from the FITS file
                   image_data, header = read_fits(file1)
           
                   #image_data = np.swapaxes(image_data, 0, 2)
                   #image_data = np.swapaxes(image_data, 0, 1)
                   image_data = np.transpose(image_data, (1, 2, 0))
           
                   # Normalize the image data to the range [0, 65535]
                   image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 65535
                   image_data = image_data.astype(np.float32)
           
                   # Convert the image to BGR format (OpenCV uses BGR by default)
                   image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
           
                   img = cv2.resize(image_bgr,None,fx=int(sysargv2) / int(sysargv2a),fy=int(sysargv2) / int(sysargv2a),interpolation=cv2.INTER_LANCZOS4)
           
                   # Save or display the result
                   image_rgb = np.transpose(img, (2, 0, 1))
           
                   image_data = image_rgb.astype(np.float32)
                   data_range = np.max(image_data) - np.min(image_data)
                   if data_range == 0:
                     normalized_data = np.zeros_like(image_data)  # or handle differently
                   else:
                     normalized_data = (image_data - np.min(image_data)) / data_range
                   image_rgb = normalized_data    
           
                   # Create a FITS HDU
                   hdu = fits.PrimaryHDU(image_rgb, header)
           
                   # Write to FITS file
                   hdu.writeto(sysargv3,  overwrite=True)
            
                 if sysargv7 == '2':
           
                   image = tifffile.imread(sysargv1)      
                   img = cv2.resize(image,None,fx=int(sysargv2) / int(sysargv2a),fy=int(sysargv2) / int(sysargv2a),interpolation=cv2.INTER_LANCZOS4)
                   tifffile.imwrite("INTER_LANCZOS4" + sysargv3, img)
           
      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def multiply2images():

  try:

      def main():

                 sysargv2  = input("Enter the first (default fits Siril) Image  -->")
                 sysargv3  = input("Enter the second (default fits Siril) Image  -->")
           
                 # Load the first FITS file
                 hdul1 = fits.open(sysargv2)
                 image_data1 = hdul1[0].data.astype(np.float64)
               
                 # Load the second FITS file
                 hdul2 = fits.open(sysargv3)
                 image_data2 = hdul2[0].data.astype(np.float64)
               
                 # Check if the image data from both FITS files have the same shape
                 if image_data1.shape != image_data2.shape:
                   print("Error: The input images do not have the same dimensions!")
                   hdul1.close()
                   hdul2.close()
                   return
                 print(image_data1.shape)
                 print(image_data2.shape)
                  #Add the RGB channels from both images
                  #Assuming both images are in the format (height, width, 3) (RGB)
                 if image_data1.ndim == 3 and image_data2.ndim == 3 :
                  # Add the corresponding channels (R, G, B) of both images
                   sysargv4  = input("Enter the filename of the added images to save  -->")
                   sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
                   sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
                   sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
                   sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")
                   sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
                   image_data1_contrastadd = int(sysargv5)
                   image_data1_contrastscale = (int(sysargv6)/int(sysargv7))
                   image_data2_contrastscale = (int(sysargv8)/int(sysargv9))
           
                   result_image = ((image_data1 * image_data1_contrastscale ) * (image_data2 * image_data1_contrastscale)) + image_data1_contrastadd
           
                   #result_image = image_data1 + image_data2
           
                   # Create a new FITS HDU for the result
                   result_hdu = fits.PrimaryHDU(result_image)
                   
                   # Create an HDU list (this only contains the result HDU)
                   hdulist = fits.HDUList([result_hdu])
                   
                   # Save the result as a new FITS file
                   hdulist.writeto(sysargv4, overwrite=True)
                   
                 else:
                   print("Error: The FITS files do not appear to be in the expected RGB format.")
               
                 # Close the FITS files
                   hdul1.close()
                   hdul2.close()

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def divide2images():

  try:

      def main():

                 sysargv2  = input("Enter the first (default fits Siril) Image  -->")
                 sysargv3  = input("Enter the second (default fits Siril) Image  -->")
           
                 # Load the first FITS file
                 hdul1 = fits.open(sysargv2)
                 image_data1 = hdul1[0].data.astype(np.float64)
               
                 # Load the second FITS file
                 hdul2 = fits.open(sysargv3)
                 image_data2 = hdul2[0].data.astype(np.float64)
               
                 # Check if the image data from both FITS files have the same shape
                 if image_data1.shape != image_data2.shape:
                   print("Error: The input images do not have the same dimensions!")
                   hdul1.close()
                   hdul2.close()
                   return
                 print(image_data1.shape)
                 print(image_data2.shape)
                  #Add the RGB channels from both images
                  #Assuming both images are in the format (height, width, 3) (RGB)
                 if image_data1.ndim == 3 and image_data2.ndim == 3 :
                  # Add the corresponding channels (R, G, B) of both images
                   sysargv4  = input("Enter the filename of the added images to save  -->")
                   sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
                   sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
                   sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
                   sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")
                   sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
                   image_data1_contrastadd = int(sysargv5)
                   image_data1_contrastscale = (int(sysargv6)/int(sysargv7))
                   image_data2_contrastscale = (int(sysargv8)/int(sysargv9))
           
                   result_image = ((image_data1 * image_data1_contrastscale ) / (image_data2 * image_data1_contrastscale)) + image_data1_contrastadd
           
                   #result_image = image_data1 + image_data2
           
                   # Create a new FITS HDU for the result
                   result_hdu = fits.PrimaryHDU(result_image)
                   
                   # Create an HDU list (this only contains the result HDU)
                   hdulist = fits.HDUList([result_hdu])
                   
                   # Save the result as a new FITS file
                   hdulist.writeto(sysargv4, overwrite=True)
                   
                 else:
                   print("Error: The FITS files do not appear to be in the expected RGB format.")
               
                 # Close the FITS files
                   hdul1.close()
                   hdul2.close()

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def max2images():

  try:

      def main():

                 sysargv2  = input("Enter the first (default fits Siril) Image  -->")
                 sysargv3  = input("Enter the second (default fits Siril) Image  -->")
           
                 # Load the first FITS file
                 hdul1 = fits.open(sysargv2)
                 image_data1 = hdul1[0].data.astype(np.float64)
               
                 # Load the second FITS file
                 hdul2 = fits.open(sysargv3)
                 image_data2 = hdul2[0].data.astype(np.float64)
               
                 # Check if the image data from both FITS files have the same shape
                 if image_data1.shape != image_data2.shape:
                   print("Error: The input images do not have the same dimensions!")
                   hdul1.close()
                   hdul2.close()
                   return
                 print(image_data1.shape)
                 print(image_data2.shape)
                  #Add the RGB channels from both images
                  #Assuming both images are in the format (height, width, 3) (RGB)
                 if image_data1.ndim == 3 and image_data2.ndim == 3 :
                  # Add the corresponding channels (R, G, B) of both images
                   sysargv4  = input("Enter the filename of the added images to save  -->")
                   sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
                   sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
                   sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
                   sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")
                   sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
                   image_data1_contrastadd = int(sysargv5)
                   image_data1_contrastscale = (int(sysargv6)/int(sysargv7))
                   image_data2_contrastscale = (int(sysargv8)/int(sysargv9))
           
                   result_image = max((image_data1 * image_data1_contrastscale ), (image_data2 * image_data1_contrastscale)) + image_data1_contrastadd
           
                   #result_image = image_data1 + image_data2
           
                   # Create a new FITS HDU for the result
                   result_hdu = fits.PrimaryHDU(result_image)
                   
                   # Create an HDU list (this only contains the result HDU)
                   hdulist = fits.HDUList([result_hdu])
                   
                   # Save the result as a new FITS file
                   hdulist.writeto(sysargv4, overwrite=True)
                   
                 else:
                   print("Error: The FITS files do not appear to be in the expected RGB format.")
               
                 # Close the FITS files
                   hdul1.close()
                   hdul2.close()

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def min2images():

  try:

      def main():

                 sysargv2  = input("Enter the first (default fits Siril) Image  -->")
                 sysargv3  = input("Enter the second (default fits Siril) Image  -->")
           
                 # Load the first FITS file
                 hdul1 = fits.open(sysargv2)
                 image_data1 = hdul1[0].data.astype(np.float64)
               
                 # Load the second FITS file
                 hdul2 = fits.open(sysargv3)
                 image_data2 = hdul2[0].data.astype(np.float64)
               
                 # Check if the image data from both FITS files have the same shape
                 if image_data1.shape != image_data2.shape:
                   print("Error: The input images do not have the same dimensions!")
                   hdul1.close()
                   hdul2.close()
                   return
                 print(image_data1.shape)
                 print(image_data2.shape)
                  #Add the RGB channels from both images
                  #Assuming both images are in the format (height, width, 3) (RGB)
                 if image_data1.ndim == 3 and image_data2.ndim == 3 :
                  # Add the corresponding channels (R, G, B) of both images
                   sysargv4  = input("Enter the filename of the added images to save  -->")
                   sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
                   sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
                   sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
                   sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")
                   sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
                   image_data1_contrastadd = int(sysargv5)
                   image_data1_contrastscale = (int(sysargv6)/int(sysargv7))
                   image_data2_contrastscale = (int(sysargv8)/int(sysargv9))
           
                   result_image = min((image_data1 * image_data1_contrastscale ), (image_data2 * image_data1_contrastscale)) + image_data1_contrastadd
           
                   #result_image = image_data1 + image_data2
           
                   # Create a new FITS HDU for the result
                   result_hdu = fits.PrimaryHDU(result_image)
                   
                   # Create an HDU list (this only contains the result HDU)
                   hdulist = fits.HDUList([result_hdu])
                   
                   # Save the result as a new FITS file
                   hdulist.writeto(sysargv4, overwrite=True)
                   
                 else:
                   print("Error: The FITS files do not appear to be in the expected RGB format.")
               
                 # Close the FITS files
                   hdul1.close()
                   hdul2.close()

      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def splittricolor():
    
  def read_fits(file_path):
      """Read a FITS, add dummy SIP orders if missing, return data & header."""
      hdul   = fits.open(file_path)
      header = hdul[0].header
      # Ensure there are SIP keywords
      if 'A_ORDER' not in header:
          header['A_ORDER'] = (0, 'dummy SIP order - no distortion')
          header['B_ORDER'] = (0, 'dummy SIP order - no distortion')
      data = hdul[0].data
      hdul.close()
      return data, header
      
  class FitsSplitter(QWidget):
      def __init__(self):
          super().__init__()
          self.setWindowTitle("FITS Color‐Channel Splitter")
          self._build_ui()
      
      def _build_ui(self):
          layout = QVBoxLayout()
  
          # Row 1: Input Directory
          row_dir = QHBoxLayout()
          lbl_dir = QLabel("Input Directory:")
          self.dir_edit = QLineEdit()
          btn_dir = QPushButton("Browse…")
          btn_dir.clicked.connect(self._browse_dir)
          row_dir.addWidget(lbl_dir)
          row_dir.addWidget(self.dir_edit, stretch=1)
          row_dir.addWidget(btn_dir)
          layout.addLayout(row_dir)
      
          # Row 2: Filename Pattern
          row_pat = QHBoxLayout()
          lbl_pat = QLabel("Filename Pattern:")
          self.pat_edit = QLineEdit("*.fit")
          row_pat.addWidget(lbl_pat)
          row_pat.addWidget(self.pat_edit, stretch=1)
          layout.addLayout(row_pat)
      
          # Split Button
          btn_split = QPushButton("Split Channels")
          btn_split.clicked.connect(self._split_channels)
          btn_split.setFixedHeight(36)
          layout.addWidget(btn_split, alignment=Qt.AlignmentFlag.AlignCenter)
      
          self.setLayout(layout)
          self.resize(600, 140)
      
      def _browse_dir(self):
          """Open a dialog to select an existing directory."""
          path = QFileDialog.getExistingDirectory(
              self, "Select Input Directory", "", QFileDialog.Option.ShowDirsOnly
          )
          if path:
              self.dir_edit.setText(path)
  
      def _split_channels(self):
          """Find files, split channels, and save into subfolders blue/green/red."""
          input_dir = self.dir_edit.text().strip() or "."
          pattern   = self.pat_edit.text().strip() or "*.fit"
  
          # Gather matching FITS files
          files = glob.glob(os.path.join(input_dir, pattern))
          if not files:
              QMessageBox.warning(
                  self, "No Files Found",
                  f"No FITS files matching:\n{input_dir}\\{pattern}"
              )
              return
   
         # Create subdirectories
          for sub in ("blue", "green", "red"):
              os.makedirs(os.path.join(input_dir, sub), exist_ok=True)
  
          # Process each FITS cube
          for fn in files:
              data, header = read_fits(fn)
              data = data.astype(np.float32)
  
              # Split RGB planes (assumes shape [3, y, x])
              b_plane = data[2, ...]
              g_plane = data[1, ...]
              r_plane = data[0, ...]
  
              base = os.path.splitext(os.path.basename(fn))[0]
              mapping = [
                  (b_plane, "b", "blue"),
                  (g_plane, "g", "green"),
                  (r_plane, "r", "red"),
              ]
  
              for arr, suffix, folder in mapping:
                  out_path = os.path.join(
                      input_dir, folder, f"{base}_{suffix}.fits"
                  )
                  fits.writeto(out_path, arr, header, overwrite=True)
  
          QMessageBox.information(
              self, "Done",
              f"Split {len(files)} file(s) into blue/, green/, red/ subdirs."
          )
  
  if __name__ == "__main__":
      app = QApplication(sys.argv)
      splitter = FitsSplitter()
      splitter.show()
      app.exec() 
      return sysargv1
      menue()

  return sysargv1
  menue()

def combinetricolor():

  try:

      def main():

                 sysargv1  = input("Enter the Blue image to be combined  -->")
                 sysargv2  = input("Enter the Green image to be combined  -->")
                 sysargv3  = input("Enter the Red image to be combined  -->")
                 sysargv4  = input("Enter the RGB file to be created  -->")
                 sysargv7  = input("Enter 0 for fits or 1 for other file -->")
           
                 if sysargv7 == '0':
            
           
                   with fits.open(sysargv1) as old_hdul:
                       # Access the header of the primary HDU
                     old_header = old_hdul[0].header
                     old_data = old_hdul[0].data
               
                   # Function to read FITS file and return data
                   def read_fits(file):
                     with fits.open(file, mode='update') as hdul:#
                       data = hdul[0].data
                       # hdul.close()
                     return data
           
                   # Read the FITS files
                   file1 = sysargv1
                   file2 = sysargv2
                   file3 = sysargv3
           
                   # Read the image data from the FITS file
                   blue = read_fits(file1)
                   green = read_fits(file2)
                   red = read_fits(file3)
           
                   blue = blue.astype(np.float32)
                   green = green.astype(np.float32)
                   red = red.astype(np.float32)
           
                   # Check dimensions
                   print("Data1 shape:", blue.shape)
                   print("Data2 shape:", green.shape)
                   print("Data3 shape:", red.shape)
           
                   #newRGBImage = cv2.merge((red,green,blue))
                   RGB_Image1 = np.stack((red,green,blue))
           
                   # Remove the extra dimension
                   RGB_Image = np.squeeze(RGB_Image1)
           
                   # Create a FITS header with NAXIS = 3
                   header = old_header
                   header['NAXIS'] = 3
                   header['NAXIS1'] = RGB_Image.shape[0]
                   header['NAXIS2'] = RGB_Image.shape[1]
                   header['NAXIS3'] = RGB_Image.shape[2]
           
                   # Ensure the data type is correct 
                   newRGB_Image = RGB_Image.astype(np.float64)
           
                   print("newRGB_Image shape:", newRGB_Image.shape)
           
                   fits.writeto( sysargv4, newRGB_Image, overwrite=True)
                   # Save the RGB image as a new FITS file with the correct header
                   hdu = fits.PrimaryHDU(data=newRGB_Image, header=header)
                   hdu.writeto(sysargv4, overwrite=True)
           
                   # Function to read and verify the saved FITS file
                   def verify_fits(sysargv4):
                     with fits.open(sysargv4) as hdul:
                       data = hdul[0].data
                   return data
           
                   # Verify the saved RGB image
                   verified_image = verify_fits(sysargv4)
                   print("Verified image shape:", verified_image.shape)
           
                 if sysargv7 == '1':
           
                   sysargv1  = input("Enter the Blue image to be combined  -->")
                   blue = cv2.imread(sysargv1, -1)
                   sysargv2  = input("Enter the Green image to be combined  -->")
                   green = cv2.imread(sysargv2, -1)
                   sysargv3  = input("Enter the Red image to be combined  -->")
                   red = cv2.imread(sysargv3, -1)
                   sysargv4  = input("Enter the RGB file to be created  -->")
           
                   # Merge the Blue, Green and Red color channels
                   newRGBImage = cv2.merge((red,green,blue))
                   cv2.imwrite(sysargv4, newRGBImage)
                   return sysargv1

      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def createLuminance():

  try:

      def main():

                 sysargv1  = input("Enter the Color Image  -->")
                 sysargv2  = input("Enter the Luminance image to be created  -->")
                 sysargv7  = input("Enter 0 for fits or 1 for other file -->")
           
                 if sysargv7 == '0':
           
                   # Function to read FITS file and return data
                   hdul = fits.open(sysargv1)
               
                   # Extract the image data from the first HDU (Primary HDU or Image Data HDU)
                   image_data = hdul[0].data
                   image_data = np.transpose(image_data, (1, 2, 0))
           
                   # Extract RGB channels
                   R = image_data[:, :, 0]
                   G = image_data[:, :, 1]
                   B = image_data[:, :, 2]
                   R = R.astype(np.float64)
                   G = G.astype(np.float64)
                   B = B.astype(np.float64)
                   
                   # Calculate the luminance (grayscale) using the standard formula
                   luminance = 0.2989 * R + 0.5870 * G + 0.1140 * B
                   # Create a new FITS HDU with the luminance data
                   luminance_hdu = fits.PrimaryHDU(luminance)
                   
                   # Create an HDU list (this is just the luminance HDU in this case)
                   hdulist = fits.HDUList([luminance_hdu])
                   
                   # Save the luminance image as a new FITS file
                   hdulist.writeto(sysargv2, overwrite=True)
           
                 if sysargv7 == '1':
           
                   img = cv2.imread(sysargv1, -1)
                   # createLuminance not percieved
                   grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                   cv2.imwrite(sysargv2, grayscale_img)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def align2img():

  try:

      def main():

                 # Load the two images
                 sysargv1  = input("Enter the 1st reference image name(WCS/std) -->")
                 sysargv7  = input("Enter 02, 03, 04, 05, 06 for fits or 1 for other file -->")
               
                 if sysargv7 == '02':
                   sysargv3  = input("Enter the 2nd reference image file name(WCS/std) -->")  
             
                   sysargv2  = input("Enter 1st Aligned image name(WCS/std)  -->")
           
                   sysargv4  = input("Enter 2nd Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
           
                   def read_fits(file):
                       hdul = fits.open(file)
                       header = hdul[0].header
                       data = hdul[0].data.astype(np.float64)
                       hdul.close()
                       return data, header
           
                   # Read the FITS files
                   file1 = sysargv1    
                   data1, wcs1 = read_fits(file1)
                   file1 = sysargv3    
                   data2, wcs2 = read_fits(file1)
           
                   # Find the optimal WCS for the reprojected images
                   wcs_out, shape_out = find_optimal_celestial_wcs([(data1, wcs1), (data2, wcs2)])
           
                   # Reproject the images to the new WCS
                   array1, footprint1 = reproject_interp((data1, wcs1), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array1 = np.flipud(array1)
                   if fliplr == '1':  array1 = np.fliplr(array1)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv2, overwrite=True)
                   array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1': array2 = np.flipud(array2)
                   if fliplr == '1': array2 = np.fliplr(array2)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4, overwrite=True)
               
                 if sysargv7 == '03':
                   sysargv3  = input("Enter the 2nd reference image file name(WCS/std) -->")  
                   sysargv3a  = input("Enter the 3rd reference image file name(WCS/std) -->")  
               
                   sysargv2  = input("Enter 1st Aligned image name(WCS/std)  -->")
           
                   sysargv4  = input("Enter 2nd Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   sysargv4a  = input("Enter 3rd Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   def read_fits(file):
                       hdul = fits.open(file)
                       header = hdul[0].header
                       data = hdul[0].data.astype(np.float64)
                       hdul.close()
                       return data, header
           
                   # Read the FITS files
                   file1 = sysargv1    
                   data1, wcs1 = read_fits(file1)
                   file1 = sysargv3    
                   data2, wcs2 = read_fits(file1)
                   file1 = sysargv3a    
                   data3, wcs3 = read_fits(file1)
           
                   # Find the optimal WCS for the reprojected images
                   wcs_out, shape_out = find_optimal_celestial_wcs([(data1, wcs1), (data2, wcs2), (data3, wcs3)])
           
                   # Reproject the images to the new WCS
                   array1, footprint1 = reproject_interp((data1, wcs1), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array1 = np.flipud(array1)
                   if fliplr == '1':  array1 = np.fliplr(array1)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv2, overwrite=True)
                   array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array2 = np.flipud(array2)
                   if fliplr == '1':  array2 = np.fliplr(array2)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4, overwrite=True)
                   array3, footprint3 = reproject_interp((data3, wcs3), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array3 = np.flipud(array3)
                   if fliplr == '1':  array3 = np.fliplr(array3)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array3, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4a, overwrite=True)
           
                 if sysargv7 == '04':
                   sysargv3  = input("Enter the 2nd reference image file name(WCS/std) -->")  
                   sysargv3a  = input("Enter the 3rd reference image file name(WCS/std) -->")  
                   sysargv3b  = input("Enter the 4th reference image file name(WCS/std) -->")  
               
                   sysargv2  = input("Enter 1st Aligned image name(WCS/std)  -->")
           
                   sysargv4  = input("Enter 2nd Aligned image name(WCS/std)  -->")
                   # Open the FITS files
               
                   sysargv4a  = input("Enter 3rd Aligned image name(WCS/std)  -->")
                   # Open the FITS files
               
                   sysargv4b  = input("Enter 4th Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   def read_fits(file):
                       hdul = fits.open(file)
                       header = hdul[0].header
                       data = hdul[0].data.astype(np.float64)
                       hdul.close()
                       return data, header
           
                   # Read the FITS files
                   file1 = sysargv1    
                   data1, wcs1 = read_fits(file1)
                   file1 = sysargv3    
                   data2, wcs2 = read_fits(file1)
                   file1 = sysargv3a    
                   data3, wcs3 = read_fits(file1)
                   file1 = sysargv3b    
                   data4, wcs4 = read_fits(file1)
           
                   # Find the optimal WCS for the reprojected images
                   wcs_out, shape_out = find_optimal_celestial_wcs([(data1, wcs1), (data2, wcs2), (data3, wcs3), (data4, wcs4)])
           
                   # Reproject the images to the new WCS
                   array1, footprint1 = reproject_interp((data1, wcs1), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array1 = np.flipud(array1)
                   if fliplr == '1':  array1 = np.fliplr(array1)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv2, overwrite=True)
                   array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array2 = np.flipud(array2)
                   if fliplr == '1':  array2 = np.fliplr(array2)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4, overwrite=True)
                   array3, footprint3 = reproject_interp((data3, wcs3), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array3 = np.flipud(array3)
                   if fliplr == '1':  array3 = np.fliplr(array3)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array3, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4a, overwrite=True)
                   array4, footprint4 = reproject_interp((data4, wcs4), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array4 = np.flipud(array4)
                   if fliplr == '1':  array4 = np.fliplr(array4)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array4, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4b, overwrite=True)
           
               
                 if sysargv7 == '05':
                   sysargv3  = input("Enter the 2nd reference image file name(WCS/std) -->")  
                   sysargv3a  = input("Enter the 3rd reference image file name(WCS/std) -->")  
                   sysargv3b  = input("Enter the 4th reference image file name(WCS/std) -->")  
                   sysargv3c  = input("Enter the 5th reference image file name(WCS/std) -->") 
                   
                   sysargv2  = input("Enter 1st Aligned image name(WCS/std)  -->")
           
                   sysargv4  = input("Enter 2nd Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   sysargv4a  = input("Enter 3rd Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   sysargv4b  = input("Enter 4th Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   sysargv4c  = input("Enter 5th Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   def read_fits(file):
                       hdul = fits.open(file)
                       header = hdul[0].header
                       data = hdul[0].data.astype(np.float64)
                       hdul.close()
                       return data, header
           
                   # Read the FITS files
                   file1 = sysargv1    
                   data1, wcs1 = read_fits(file1)
                   file1 = sysargv3    
                   data2, wcs2 = read_fits(file1)
                   file1 = sysargv3a    
                   data3, wcs3 = read_fits(file1)
                   file1 = sysargv3b    
                   data4, wcs4 = read_fits(file1)
                   file1 = sysargv3c    
                   data5, wcs5 = read_fits(file1)
           
                   # Find the optimal WCS for the reprojected images
                   wcs_out, shape_out = find_optimal_celestial_wcs([(data1, wcs1), (data2, wcs2), (data3, wcs3), (data4, wcs4)])
           
                   # Reproject the images to the new WCS
                   array1, footprint1 = reproject_interp((data1, wcs1), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array1 = np.flipud(array1)
                   if fliplr == '1':  array1 = np.fliplr(array1)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv2, overwrite=True)
                   array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array2 = np.flipud(array2)
                   if fliplr == '1':  array2 = np.fliplr(array2)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4, overwrite=True)
                   array3, footprint3 = reproject_interp((data3, wcs3), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array3 = np.flipud(array3)
                   if fliplr == '1':  array3 = np.fliplr(array3)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array3, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4a, overwrite=True)
                   array4, footprint4 = reproject_interp((data4, wcs4), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array4 = np.flipud(array4)
                   if fliplr == '1':  array4 = np.fliplr(array4)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array4, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4b, overwrite=True)
                   array5, footprint5 = reproject_interp((data5, wcs5), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array5 = np.flipud(array5)
                   if fliplr == '1':  array5 = np.fliplr(array5)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array5, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4c, overwrite=True)
           
                 if sysargv7 == '06':
                   sysargv3  = input("Enter the 2nd reference image file name(WCS/std) -->")  
                   sysargv3a  = input("Enter the 3rd reference image file name(WCS/std) -->")  
                   sysargv3b  = input("Enter the 4th reference image file name(WCS/std) -->")  
                   sysargv3c  = input("Enter the 5th reference image file name(WCS/std) -->") 
                   sysargv3d  = input("Enter the 6th reference image file name(WCS/std) -->")
                   
                   sysargv2  = input("Enter 1st Aligned image name(WCS/std)  -->")
           
                   sysargv4  = input("Enter 2nd Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   sysargv4a  = input("Enter 3rd Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   sysargv4b  = input("Enter 4th Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   sysargv4c  = input("Enter 5th Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   sysargv4d  = input("Enter 6th Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
           
                   def read_fits(file):
                       hdul = fits.open(file)
                       header = hdul[0].header
                       data = hdul[0].data.astype(np.float64)
                       hdul.close()
                       return data, header
           
                   # Read the FITS files
                   file1 = sysargv1    
                   data1, wcs1 = read_fits(file1)
                   file1 = sysargv3    
                   data2, wcs2 = read_fits(file1)
                   file1 = sysargv3a    
                   data3, wcs3 = read_fits(file1)
                   file1 = sysargv3b    
                   data4, wcs4 = read_fits(file1)
                   file1 = sysargv3c    
                   data5, wcs5 = read_fits(file1)
                   file1 = sysargv3d    
                   data6, wcs6 = read_fits(file1)
           
           
                   # Find the optimal WCS for the reprojected images
                   wcs_out, shape_out = find_optimal_celestial_wcs([(data1, wcs1), (data2, wcs2), (data3, wcs3), (data4, wcs4)])
           
                   # Reproject the images to the new WCS
                   array1, footprint1 = reproject_interp((data1, wcs1), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array1 = np.flipud(array1)
                   if fliplr == '1':  array1 = np.fliplr(array1)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv2, overwrite=True)
                   array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array2 = np.flipud(array2)
                   if fliplr == '1':  array2 = np.fliplr(array2)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4, overwrite=True)
                   array3, footprint3 = reproject_interp((data3, wcs3), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array3 = np.flipud(array3)
                   if fliplr == '1':  array3 = np.fliplr(array3)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array3, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4a, overwrite=True)
                   array4, footprint4 = reproject_interp((data4, wcs4), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array4 = np.flipud(array4)
                   if fliplr == '1':  array4 = np.fliplr(array4)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array4, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4b, overwrite=True)
                   array5, footprint5 = reproject_interp((data5, wcs5), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array5 = np.flipud(array5)
                   if fliplr == '1':  array5 = np.fliplr(array5)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array5, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4c, overwrite=True)
                   array6, footprint6 = reproject_interp((data6, wcs6), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array6 = np.flipud(array6)
                   if fliplr == '1':  array6 = np.fliplr(array6)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array6, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4d, overwrite=True)
           
                 if sysargv7 == '1':
           
                   img1 = cv2.imread( sysargv2, -1 )
                   img2 = cv2.imread( sysargv1, -1 )
           
                   # Convert the images to grayscale
                   gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                   gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
           
                   # Find the keypoints and descriptors with SIFT
                   sift = cv2.SIFT_create()
                   kp1, des1 = sift.detectAndCompute(gray1, None)
                   kp2, des2 = sift.detectAndCompute(gray2, None)
           
                   # Match the descriptors
                   bf = cv2.BFMatcher()
                   matches = bf.knnMatch(des1, des2, k=2)
           
                   # Apply ratio test
                   good = []
                   for m, n in matches:
                       if m.distance < 0.75 * n.distance:
                           good.append(m)
           
                   # Get the coordinates of the matched keypoints
                   src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                   dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
           
                   # Calculate the homography matrix
                   H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
           
                   # Warp the first image to align with the second image
                   aligned_img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
           
                   # Display the aligned image
                   cv2.imshow('Aligned Image', aligned_img)
                   cv2.waitKey(0)
                   cv2.destroyAllWindows()
           
                   cv2.imwrite( sysargv3, aligned_img)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def plotto3d16(sysargv2):

  try:
      
      def main():

                 img = sysargv2
                 #lena = cv2.imread(img, 0)
                 lena = cv2.imread(img, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                 # downscaling has a "smoothing" effect
                 #lena = cv2.resize(lena, (121,121))  # create the x and y coordinate arrays (here we just use pixel indices)
                 xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]
                 # create the figure
                 fig = plt.figure()
                 ax = fig.add_subplot(projection='3d')
                 ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)
                 # show it
                 plt.show()

      if __name__ == "__main__":
          main()                 

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def PNGcreateimage16(sysargv2, sysargv3, sysargv4, sysargv5):

  try:

      def main():

                 radius = 0
                 radiusp1 = 0
                 one = 0
                 two = 0
                 three = 0
                 four = 0
                 diameter = 0
                 diameterp1 = 0
                 imgcrop()
                 radius, radiusp1, diameter, diameterp1, one, two, three, four = imgcoord(radius, radiusp1, diameter, diameterp1, one, two, three, four)
                 my_data = np.array(Image.open('crop.png'))
                 img = np.array(Image.open('crop.png'))
               #  for x in range(121):
               #    for y in range(121):
               #        my_data[x,y]=img[x,y]-min(img[x,y],img[x,120-y],img[120-x,y],img[120-x,120-y])
                 for x in range(diameterp1):
                   for y in range(diameterp1):
                       my_data[x,y]=img[x,y]-min(img[x,y],img[x,diameter-y],img[diameter-x,y],img[diameter-x,diameter-y])   
                 #Rescale to 0-65535 and convert to uint16
                 rescaled = (65535.0 / my_data.max() * (my_data - my_data.min())).astype(np.uint16)
                 im = Image.fromarray(rescaled)
                 symfile = (sysargv2+"_"+sysargv4+"_"+sysargv5+".png")
                 im.save(symfile)
                 image_1 = imread(symfile)
                 # plot raw pixel data
                 blur = cv2.blur(image_1,(3,3)) 
                 pyplot.imshow(blur,cmap='gray')
                 # show the figure
                 pyplot.show()

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def imgcoord(radius, radiusp1, diameter, diameterp1, one, two, three, four):
  radius = int(int(sysargv3))
  radiusp1 = int((int(sysargv3))+1)
  diameter = int(int(sysargv3)*2)
  diameterp1 = int((int(sysargv3)*2)+1)
  one = int((int(sysargv4) - radius))
  two = int((int(sysargv5) - radius))
  three = int((int(sysargv4) + radiusp1))
  four = int((int(sysargv5) + radiusp1))
  #print(radius, radiusp1, diameter, diameterp1, one, two, three, four)
  return radius, radiusp1, diameter, diameterp1, one, two, three, four

def imgcrop():
  radius = 0
  radiusp1 = 0
  one = 0
  two = 0
  three = 0
  four = 0
  diameter = 0
  diameterp1 = 0
  radius, radiusp1, diameter, diameterp1, one, two, three, four = imgcoord(radius, radiusp1, diameter, diameterp1, one, two, three, four)
  #print(radius, radiusp1, diameter, diameterp1, one, two, three, four)
  im = Image.open(sysargv2)
  im1 = im.crop((one, two, three, four))
  im1.save('crop.png')

import numpy as np
import cv2
from astropy.io import fits

def unsharpMask():
    try:
        def main():
            # Data entry instructions (do not change these)
            sysargv1 = input("Enter the Color Image  -->")
            sysargv2 = input("Enter the unsharpMask image to be created  -->")
            
            # Read the input FITS file.
            hdul = fits.open(sysargv1)
            header = hdul[0].header
            data = hdul[0].data.astype(np.float64)
            hdul.close()
            
            # Process based on data dimensions.
            # If the data is 2D (grayscale), process directly;
            # if it is 3D (height, width, channels), process each channel.
            if data.ndim == 2:
                # Apply a Gaussian blur with sigma = 2.0
                blurred = cv2.GaussianBlur(data, (0, 0), 2.0)
                # Compute unsharp mask: result = 2*original - blurred
                unsharp = cv2.addWeighted(data, 2.0, blurred, -1.0, 0)
            elif data.ndim == 3:
                # Assume data stored as (height, width, channels)
                h, w, ch = data.shape
                unsharp_channels = []
                for c in range(ch):
                    channel = data[:, :, c]
                    blurred = cv2.GaussianBlur(channel, (0, 0), 2.0)
                    # Apply unsharp mask to this channel
                    unsharp_channel = cv2.addWeighted(channel, 2.0, blurred, -1.0, 0)
                    unsharp_channels.append(unsharp_channel)
                unsharp = np.stack(unsharp_channels, axis=2)
            else:
                print("Unsupported data dimensions.")
                return
            
            # Write the resulting image to a new FITS file.
            fits.writeto(sysargv2, unsharp.astype(np.float64), header, overwrite=True)
            print("Unsharp mask processing completed. Output saved to", sysargv2)
            
        if __name__ == "__main__":
            main()
            
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Returning to the Main Menue...")
        return sysargv1  # If you have a menue() routine, you may call it here.
        menue()

    return sysargv1
    menue()

def DynamicRescale16():

  try:

      def main():

           # -------------------------------------------------
           # UTILITY FUNCTIONS FOR FITS HANDLING
           # -------------------------------------------------
                 def load_fits(file_path):
                     with fits.open(file_path) as hdul:
                         return hdul[0].data, hdul[0].header
           
                 def split_image(image, tile_size=(600, 600), output_dir="tiles"):
                     os.makedirs(output_dir, exist_ok=True)
                     h, w = image.shape
                     print("Image shape:", image.shape)
                     tile_h, tile_w = tile_size
                     tiles = []
           
                     for i in range(0, h, tile_h):
                         for j in range(0, w, tile_w):
                             # Get the subimage; may be smaller near the right/bottom edges.
                             sub_image = image[i:i+tile_h, j:j+tile_w]
                             sub_h, sub_w = sub_image.shape
           
                             # Create a full-size tile and place the subimage in the top-left corner.
                             padded_tile = np.zeros(tile_size)
                             padded_tile[:sub_h, :sub_w] = sub_image
           
                             # Save the tile with metadata in the filename.
                             tile_file = f"{output_dir}/tile_{i}_{j}_{sub_h}_{sub_w}.fits"
                             fits.writeto(tile_file, padded_tile, overwrite=True)
                             tiles.append(tile_file)
               
                     return tiles
           
                 import re
           
                 def reassemble_image(tiles, original_shape):
                     """
                     Reassemble the full image from a list of processed tile files.
                     The tile filename is expected to include the pattern:
                        tile_i_j_subH_subW
                     (even though additional suffix text is appended).
                     """
                     final_image = np.zeros(original_shape)
               
                     # Regular expression pattern to extract the metadata numbers.
                     pattern = r"tile_(\d+)_(\d+)_(\d+)_(\d+)"
               
                     for tile_file in tiles:
                         base = os.path.basename(tile_file)
                         match = re.search(pattern, base)
                         if not match:
                             print(f"Filename {base} does not match expected pattern.")
                             continue
                         i_str, j_str, sub_h_str, sub_w_str = match.groups()
                         i, j, sub_h, sub_w = map(int, [i_str, j_str, sub_h_str, sub_w_str])
               
                         with fits.open(tile_file) as hdul:
                             data = hdul[0].data
                             # Only take the valid (unpadded) portion from the tile.
                             final_image[i:i+sub_h, j:j+sub_w] = data[:sub_h, :sub_w]
           
                     return final_image
           
                 # -------------------------------------------------
                 # UPDATED TILE PROCESSING FUNCTION
                 # -------------------------------------------------
                 def process_tile(tile_file, width_of_square, bin_value, gamma_value, resize_factor, resize_div):
                     """
                     Processes a given tile file applying dynamic block rescaling,
                     gamma correction, and binning. The parameters are passed in from main.
                     """
                     print(f"\nProcessing tile: {tile_file}")
               
                     # Open the tile file.
                     with fits.open(tile_file) as hdul:
                         header = hdul[0].header
                         image_data = hdul[0].data
           
                     print("Original tile shape:", image_data.shape)
                     print("Data type:", image_data.dtype.name)
               
                     # Normalize the image data to the range [0, 65535] and cast to uint16.
                     norm_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 65535
                     norm_image = norm_image.astype(np.uint16)
               
                     # Resize the image using cv2.resize.
                     resized_image = cv2.resize(norm_image, None, fx=(resize_factor / resize_div), 
                                                fy=(resize_factor / resize_div), interpolation=cv2.INTER_LANCZOS4)

                     # ----------------------------------------------------------------
                     # 2) QUICK PYTHON CAST: make sure your Cython sees float64 inputs
                     # ----------------------------------------------------------------
                     # cast input to float64
                     img64 = resized_image.astype(np.float64)
                     # allocate a same‐shape float64 output buffer
                     out64 = np.empty_like(img64, dtype=np.float64)
                     # (you no longer need your old `my_data = resized_image * 65535`)

                     # 3) call the Cython routine
               
                     # Multiply the resized image to scale it up.
                     my_data = resized_image * 65535
                     img = resized_image * 65535
           
                     # Use the provided square block width for dynamic square processing.
                     block_size = int(width_of_square)
           
                     warp_affine_mask_rescale(img64, out64, block_size)
                     my_data = out64
           
                     # Apply gamma correction.
                     gamma_corrected1 = np.array(65535.0 * (my_data / 65535) ** gamma_value, dtype='float64')
                     gamma_corrected = np.round(gamma_corrected1)
               
                     # Normalize before binning (division factor from original code is 6553500).
                     img_array = np.asarray(gamma_corrected / 6553500, dtype='float64')
               
                     # Calculate new dimensions based on the bin factor.
                     bin_factor = int(bin_value)
                     h_img, w_img = img_array.shape
                     new_height = h_img // bin_factor
                     new_width = w_img // bin_factor
           
                     # Bin the image using summation in non-overlapping blocks.
                     binned_image = np.zeros((new_height, new_width), dtype=img_array.dtype)
                     for y in range(new_height):
                         for x in range(new_width):
                             binned_image[y, x] = np.sum(
                                 img_array[y * bin_factor:(y + 1) * bin_factor,
                                           x * bin_factor:(x + 1) * bin_factor]
                             )
           
                     # Write the processed tile to a new FITS file. The new file name gets the extra suffix.
                     hdu = fits.PrimaryHDU(binned_image, header=header)
                     hdulist = fits.HDUList([hdu])
                     out_filename = tile_file + '_binned_gamma_corrected_drs.fits'
                     hdulist.writeto(out_filename, overwrite=True)
                     print(f"Tile processed and saved to {out_filename}\n")
           
                 # -------------------------------------------------
                 # MAIN EXECUTION
                 # -------------------------------------------------
                 if __name__ == "__main__":
                     sysargv7 = input("Enter 01 to split tile, 02 to process tile files, or 03 to combine tiles --> ")
               
                     if sysargv7 == '01':
                         # Splitting mode.
                         input_file_name = input("Enter the input FITS file name--> ")
                         fits_file = input_file_name
                         image, header = load_fits(fits_file)
                         tile_size = (600, 600)
                         output_dir = input("Enter the output_dir name(tiles)--> ")
                         tiles = split_image(image, tile_size, output_dir)
                         print("Image split into tiles:")
                         for t in tiles:
                             print("  ", t)
               
                     elif sysargv7 == '02':
                         # Processing mode: Collect processing parameters once.
                         output_dir = input("Enter the output_dir name(tiles)--> ")
                         width_of_square = input("Enter the width of square (e.g., 5): ")
                         bin_value = input("Enter the bin value (e.g., 25): ")
                         gamma_value = float(input("Enter gamma (e.g., 0.3981) for 1 magnitude: "))
                         # Fixed values used for resizing.
                         resize_factor = int(25)   # corresponds to sysargv2 = 25 in the original code.
                         resize_div = int(1)       # corresponds to sysargv2a = 1 in the original code.
                   
                         # Get list of tiles from the "tiles" directory.
                         tiles = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".fits")])
                         for tile_file in tiles:
                             process_tile(tile_file, width_of_square, bin_value,
                                          gamma_value, resize_factor, resize_div)
               
                     elif sysargv7 == '03':
                         # Reassemble mode.
                         output_dir = input("Enter the output_dir name(tiles)--> ")
                         input_file_name = input("Enter the input FITS file name--> ")
                         fits_file = input_file_name
                         image, header = load_fits(fits_file)
                         # Select only those processed tile files.
                         tiles = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith("_binned_gamma_corrected_drs.fits")])
                         final_image = reassemble_image(tiles, image.shape)
                         filename = "output_" + output_dir + ".fits"
                         fits.writeto(filename, final_image, header, overwrite=True)
                         # Optional: Cleanup processed tile files.
                         for tile_file in tiles:
                             os.remove(tile_file)
                         print("Processing complete. Final image saved as output_" + output_dir + ".fits "+" .")
               
                     else:
                         print("Invalid option entered.")
           
      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def DynamicRescale16RGB():
  try:
        
      def deletefiles():
                                
          currentDirectory = os.path.abspath(os.getcwd())
          # Define the file to delete
          file_to_delete = "img_enlarged_25x.fit"
          file_to_delete = (os.path.join(currentDirectory, file_to_delete))
          # Check if the file exists
          if os.path.exists(file_to_delete):
              os.remove(file_to_delete)  # Delete the file
              print(f"File '{file_to_delete}' has been deleted.")
          else:
              print(f"File '{file_to_delete}' does not exist.")
      
          currentDirectory = os.path.abspath(os.getcwd())
      
          # Define the file to delete
          file_to_delete = "channel_0_64bit.fits"
          file_to_delete = (os.path.join(currentDirectory, file_to_delete))
          # Check if the file exists
          if os.path.exists(file_to_delete):
              os.remove(file_to_delete)  # Delete the file
              print(f"File '{file_to_delete}' has been deleted.")
          else:
              print(f"File '{file_to_delete}' does not exist.")
      
          # Define the file to delete
          file_to_delete = "channel_1_64bit.fits"
          file_to_delete = (os.path.join(currentDirectory, file_to_delete))
          # Check if the file exists
          if os.path.exists(file_to_delete):
              os.remove(file_to_delete)  # Delete the file
              print(f"File '{file_to_delete}' has been deleted.")
          else:
              print(f"File '{file_to_delete}' does not exist.")
      
          # Define the file to delete
          file_to_delete = "channel_2_64bit.fits"
          file_to_delete = (os.path.join(currentDirectory, file_to_delete))
          # Check if the file exists
          if os.path.exists(file_to_delete):
              os.remove(file_to_delete)  # Delete the file
              print(f"File '{file_to_delete}' has been deleted.")
          else:
              print(f"File '{file_to_delete}' does not exist.")
      
          # Define the file to delete
          file_to_delete = "channel_RGB_64bit_binned_gamma_corrected_drs_B.fit"
          file_to_delete = (os.path.join(currentDirectory, file_to_delete))
          # Check if the file exists
          if os.path.exists(file_to_delete):
              os.remove(file_to_delete)  # Delete the file
              print(f"File '{file_to_delete}' has been deleted.")
          else:
              print(f"File '{file_to_delete}' does not exist.")
      
          # Define the file to delete
          file_to_delete = "channel_RGB_64bit_binned_gamma_corrected_drs_G.fit"
          file_to_delete = (os.path.join(currentDirectory, file_to_delete))
          # Check if the file exists
          if os.path.exists(file_to_delete):
              os.remove(file_to_delete)  # Delete the file
              print(f"File '{file_to_delete}' has been deleted.")
          else:
                print(f"File '{file_to_delete}' does not exist.")
      
          # Define the file to delete
          file_to_delete = "channel_RGB_64bit_binned_gamma_corrected_drs_R.fit"
          file_to_delete = (os.path.join(currentDirectory, file_to_delete))
          # Check if the file exists
          if os.path.exists(file_to_delete):
              os.remove(file_to_delete)  # Delete the file
              print(f"File '{file_to_delete}' has been deleted.")
          else:
              print(f"File '{file_to_delete}' does not exist.")


      
      def main():      

                 sysargv1  = input("Enter the Image to be resized(LANCZOS4)  -->")
                 width_of_square  = input("Enter the the width of square(5)  -->")
                 sysargv2  = int(25)
                 sysargv2a = int(1)
                 sysargv3  = "img_enlarged_25x.fit"
           
                 #################################################################################
                 #################################################################################
           
                   # Function to read FITS file and return data
                 def read_fits(file):
                     hdul = fits.open(file)
                     header = hdul[0].header
                     data = hdul[0].data
                     hdul.close()
                     return data, header
           
                 # Read the FITS files
                 file1 = sysargv1
           
                 # Read the image data from the FITS file
                 image_data, header = read_fits(file1)
           
                 #image_data = np.swapaxes(image_data, 0, 2)
                 #image_data = np.swapaxes(image_data, 0, 1)
                 image_data = np.transpose(image_data, (1, 2, 0))
           
                 # Normalize the image data to the range [0, 65535]
                 image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 65535
                 image_data = image_data.astype(np.uint16)
           
                 # Convert the image to BGR format (OpenCV uses BGR by default)
                 image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
           
                 img = cv2.resize(image_bgr,None,fx=int(sysargv2) / int(sysargv2a),fy=int(sysargv2) / int(sysargv2a),interpolation=cv2.INTER_LANCZOS4)
           
                 # Save or display the result
                 image_rgb = np.transpose(img, (2, 0, 1))
           
                 image_data = image_rgb.astype(np.float64)
                 data_range = np.max(image_data) - np.min(image_data)
                 if data_range == 0:
                   normalized_data = np.zeros_like(image_data)  # or handle differently
                 else:
                   normalized_data = (image_data - np.min(image_data)) / data_range
                 image_rgb = normalized_data    
           
                 # Create a FITS HDU
                 hdu = fits.PrimaryHDU(image_rgb, header)
           
                 # Write to FITS file
                 hdu.writeto(sysargv3,  overwrite=True)
           
                 #################################################################################
                 #################################################################################
           
                 sysargv2  = "img_enlarged_25x.fit"
           
                 # Function to read FITS file and return data
                 def read_fits(file):
                     hdul = fits.open(file)
                     header = hdul[0].header
                     data = hdul[0].data
                     hdul.close()
                     return data, header
           
                 # Read the FITS files
                 file1 = sysargv2
           
                 # Read the image data from the FITS file
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
           
                 # Split the color image into its individual channels
                 #b, g, r = cv2.split(image_data)
                 b, g, r = image_data[2, :, :], image_data[1, :, :], image_data[0, :, :] 
           
           
                 # Save each channel as a separate file
                 fits.writeto(f'channel_0_64bit.fits', b.astype(np.float64), header, overwrite=True)
                 fits.writeto(f'channel_1_64bit.fits', g.astype(np.float64), header, overwrite=True)
                 fits.writeto(f'channel_2_64bit.fits', r.astype(np.float64), header, overwrite=True)
           
                 sysargv1  = "channel_0_64bit.fits"
                 sysargv1a  = "channel_1_64bit.fits"
                 sysargv1b  = "channel_2_64bit.fits"
                 #sysargv2  = input("Enter the the width of square(5)  -->")
                 sysargv2  = width_of_square
                 #sysargv4  = input("Enter the image width in pixels(1000)  -->")
                 #sysargv3  = input("Enter the image height in pixels(1000)  -->")
                 sysargv5  = "channel_RGB_64bit"
                 sysargv6  = "25" #image bin value
                 gamma     = float(".3981")
           
               #################################################################################
                 # Replace 'your_fits_file.fits' with the actual path to your FITS file
                 fits_image_filename = sysargv1
                 # Open the FITS file
                 with fits.open(fits_image_filename) as hdul:
                     # Access the primary HDU (extension 0)
                     header = hdul[0].header
                     image_data = hdul[0].data
                 # Now 'image_data' contains the data from the FITS file as a 2D numpy array
                 hdul.close()
           
                 print(image_data.shape)
                 print(image_data.dtype.name)
                 height, width = image_data.shape
                 sysargv4 = str(width)
                 sysargv3 = str(height)
           
                 my_data = (image_data * 65535)
                 img = (image_data * 65535)
                 #make the Dynamic square loops
                 for xw in range(0, int(sysargv3), int(sysargv2)):
                   for yh in range(0, int(sysargv4), int(sysargv2)): 
                     my_data1 = np.zeros((int(sysargv2), int(sysargv2)))
                     for (x) in range(int(sysargv2)):
                       for (y) in range(int(sysargv2)):
                         my_data1[x,y]=img[(x+xw),(y+yh)]
                     #Rescale to 0-65535 and convert to uint16
                     rescaled1 = ((my_data1.max()+1) * ((my_data1+1) - my_data1.min()) / 65535.0).astype(np.float64)
                     rescaled = (np.round(rescaled1))
                     my_data[xw:(xw+int(sysargv2)), yh:(yh+int(sysargv2))] = rescaled
             
                 for gamma in [float(gamma)]: 
                   # Apply gamma correction. 
                   gamma_corrected1 = np.array(65535.0 *(my_data / 65535) ** gamma, dtype = 'float64') 
                   gamma_corrected = (np.round(gamma_corrected1))
                 #cv2.imwrite(str(sysargv5)+'.tif', gamma_corrected)  
           
                 ##hdu = fits.PrimaryHDU(gamma_corrected)
                 # Create an HDU list and add the primary HDU
                 ##hdulist = fits.HDUList([hdu])
                 # Specify the output FITS file path
                 ##output_fits_filename = sysargv5
                 # Write the HDU list to the FITS file
                 ##hdulist.writeto(str(sysargv5)+'gamma_corrected_drs.fit', overwrite=True)
             
             
                 img_array = np.asarray(gamma_corrected / 6553500, dtype = 'float64')
                 bin_factor = int(sysargv6) 
                 # Get image dimensions
                 height, width = img_array.shape
           
                 # Calculate new dimensions
                 new_height = height // bin_factor
                 new_width = width // bin_factor
           
                 # Bin the image using summation
                 binned_image = np.zeros((new_height, new_width), dtype=img_array.dtype)
                 for y in range(new_height):
                   for x in range(new_width):
                     # Sum pixel values within the bin
                     binned_image[y, x] = np.sum(img_array[y*bin_factor:(y+1)*bin_factor, x*bin_factor:(x+1)*bin_factor])
           
                 hdu = fits.PrimaryHDU(binned_image)
                 # Create an HDU list and add the primary HDU
                 hdulist = fits.HDUList([hdu])
                 # Specify the output FITS file path
                 output_fits_filename = sysargv5
                 # Write the HDU list to the FITS file
                 hdulist.writeto(str(sysargv5)+'_binned_gamma_corrected_drs_B.fit', overwrite=True)
           
               ###########################################################################################
               #################################################################################
                 # Replace 'your_fits_file.fits' with the actual path to your FITS file
                 fits_image_filename = sysargv1a
                 # Open the FITS file
                 with fits.open(fits_image_filename) as hdul:
                     # Access the primary HDU (extension 0)
                     header = hdul[0].header
                     image_data = hdul[0].data
                 # Now 'image_data' contains the data from the FITS file as a 2D numpy array
                 hdul.close()
           
                 print(image_data.shape)
                 print(image_data.dtype.name)
                 height, width = image_data.shape
                 sysargv4 = str(width)
                 sysargv3 = str(height)
           
                 my_data = (image_data * 65535)
                 img = (image_data * 65535)
                 #make the Dynamic square loops
                 for xw in range(0, int(sysargv3), int(sysargv2)):
                   for yh in range(0, int(sysargv4), int(sysargv2)): 
                     my_data1 = np.zeros((int(sysargv2), int(sysargv2)))
                     for (x) in range(int(sysargv2)):
                       for (y) in range(int(sysargv2)):
                         my_data1[x,y]=img[(x+xw),(y+yh)]
                     #Rescale to 0-65535 and convert to uint16
                     rescaled1 = ((my_data1.max()+1) * ((my_data1+1) - my_data1.min()) / 65535.0).astype(np.float64)
                     rescaled = (np.round(rescaled1))
                     my_data[xw:(xw+int(sysargv2)), yh:(yh+int(sysargv2))] = rescaled
             
                 for gamma in [float(gamma)]: 
                   # Apply gamma correction. 
                   gamma_corrected1 = np.array(65535.0 *(my_data / 65535) ** gamma, dtype = 'float64') 
                   gamma_corrected = (np.round(gamma_corrected1))
                 #cv2.imwrite(str(sysargv5)+'.tif', gamma_corrected)  
           
                 ##hdu = fits.PrimaryHDU(gamma_corrected)
                 # Create an HDU list and add the primary HDU
                 ##hdulist = fits.HDUList([hdu])
                 # Specify the output FITS file path
                 ##output_fits_filename = sysargv5
                 # Write the HDU list to the FITS file
                 ##hdulist.writeto(str(sysargv5)+'gamma_corrected_drs.fit', overwrite=True)
             
             
                 img_array = np.asarray(gamma_corrected / 6553500, dtype = 'float64')
                 bin_factor = int(sysargv6) 
                 # Get image dimensions
                 height, width = img_array.shape
           
                 # Calculate new dimensions
                 new_height = height // bin_factor
                 new_width = width // bin_factor
           
                 # Bin the image using summation
                 binned_image = np.zeros((new_height, new_width), dtype=img_array.dtype)
                 for y in range(new_height):
                   for x in range(new_width):
                     # Sum pixel values within the bin
                     binned_image[y, x] = np.sum(img_array[y*bin_factor:(y+1)*bin_factor, x*bin_factor:(x+1)*bin_factor])
           
                 hdu = fits.PrimaryHDU(binned_image)
                 # Create an HDU list and add the primary HDU
                 hdulist = fits.HDUList([hdu])
                 # Specify the output FITS file path
                 output_fits_filename = sysargv5
                 # Write the HDU list to the FITS file
                 hdulist.writeto(str(sysargv5)+'_binned_gamma_corrected_drs_G.fit', overwrite=True)
           
               ###########################################################################################
               #################################################################################
                 # Replace 'your_fits_file.fits' with the actual path to your FITS file
                 fits_image_filename = sysargv1b
                 # Open the FITS file
                 with fits.open(fits_image_filename) as hdul:
                     # Access the primary HDU (extension 0)
                     header = hdul[0].header
                     image_data = hdul[0].data
                 # Now 'image_data' contains the data from the FITS file as a 2D numpy array
                 hdul.close()
           
                 print(image_data.shape)
                 print(image_data.dtype.name)
                 height, width = image_data.shape
                 sysargv4 = str(width)
                 sysargv3 = str(height)
           
                 my_data = (image_data * 65535)
                 img = (image_data * 65535)
                 #make the Dynamic square loops
                 for xw in range(0, int(sysargv3), int(sysargv2)):
                   for yh in range(0, int(sysargv4), int(sysargv2)): 
                     my_data1 = np.zeros((int(sysargv2), int(sysargv2)))
                     for (x) in range(int(sysargv2)):
                       for (y) in range(int(sysargv2)):
                         my_data1[x,y]=img[(x+xw),(y+yh)]
                     #Rescale to 0-65535 and convert to uint16
                     rescaled1 = ((my_data1.max()+1) * ((my_data1+1) - my_data1.min()) / 65535.0).astype(np.float64)
                     rescaled = (np.round(rescaled1))
                     my_data[xw:(xw+int(sysargv2)), yh:(yh+int(sysargv2))] = rescaled
             
                 for gamma in [float(gamma)]: 
                   # Apply gamma correction. 
                   gamma_corrected1 = np.array(65535.0 *(my_data / 65535) ** gamma, dtype = 'float64') 
                   gamma_corrected = (np.round(gamma_corrected1))
                 #cv2.imwrite(str(sysargv5)+'.tif', gamma_corrected)  
           
                 ##hdu = fits.PrimaryHDU(gamma_corrected)
                 # Create an HDU list and add the primary HDU
                 ##hdulist = fits.HDUList([hdu])
                 # Specify the output FITS file path
                 ##output_fits_filename = sysargv5
                 # Write the HDU list to the FITS file
                 ##hdulist.writeto(str(sysargv5)+'gamma_corrected_drs.fit', overwrite=True)
             
             
                 img_array = np.asarray(gamma_corrected / 6553500, dtype = 'float64')
                 bin_factor = int(sysargv6) 
                 # Get image dimensions
                 height, width = img_array.shape
           
                 # Calculate new dimensions
                 new_height = height // bin_factor
                 new_width = width // bin_factor
           
                 # Bin the image using summation
                 binned_image = np.zeros((new_height, new_width), dtype=img_array.dtype)
                 for y in range(new_height):
                   for x in range(new_width):
                     # Sum pixel values within the bin
                     binned_image[y, x] = np.sum(img_array[y*bin_factor:(y+1)*bin_factor, x*bin_factor:(x+1)*bin_factor])
           
                 hdu = fits.PrimaryHDU(binned_image)
                 # Create an HDU list and add the primary HDU
                 hdulist = fits.HDUList([hdu])
                 # Specify the output FITS file path
                 output_fits_filename = sysargv5
                 # Write the HDU list to the FITS file
                 hdulist.writeto(str(sysargv5)+'_binned_gamma_corrected_drs_R.fit', overwrite=True)
           
               ###########################################################################################
           
                 sysargv1  = "channel_RGB_64bit_binned_gamma_corrected_drs_B.fit"
                 sysargv2  = "channel_RGB_64bit_binned_gamma_corrected_drs_G.fit"
                 sysargv3  = "channel_RGB_64bit_binned_gamma_corrected_drs_R.fit"
                 sysargv4  = "channel_RGB_64bit_binned_gamma_corrected_drs_RGB.fit"
           
           
                 with fits.open(sysargv1) as old_hdul:
                     # Access the header of the primary HDU
                   old_header = old_hdul[0].header
                   old_data = old_hdul[0].data
               
                 # Function to read FITS file and return data
                 def read_fits(file):
                   with fits.open(file, mode='update') as hdul:#
                     data = hdul[0].data
                     # hdul.close()
                   return data
           
                 # Read the FITS files
                 file1 = sysargv1
                 file2 = sysargv2
                 file3 = sysargv3
           
                 # Read the image data from the FITS file
                 blue = read_fits(file1)
                 green = read_fits(file2)
                 red = read_fits(file3)
           
                 blue = blue.astype(np.float64)
                 green = green.astype(np.float64)
                 red = red.astype(np.float64)
           
                 # Check dimensions
                 print("Data1 shape:", blue.shape)
                 print("Data2 shape:", green.shape)
                 print("Data3 shape:", red.shape)
           
                 #newRGBImage = cv2.merge((red,green,blue))
                 RGB_Image1 = np.stack((red,green,blue))
           
                 # Remove the extra dimension
                 RGB_Image = np.squeeze(RGB_Image1)
           
                 # Create a FITS header with NAXIS = 3
                 header = old_header
                 header['NAXIS'] = 3
                 header['NAXIS1'] = RGB_Image.shape[0]
                 header['NAXIS2'] = RGB_Image.shape[1]
                 header['NAXIS3'] = RGB_Image.shape[2]
           
                 # Ensure the data type is correct 
                 newRGB_Image = RGB_Image.astype(np.float64)
           
                 print("newRGB_Image shape:", newRGB_Image.shape)
           
                 fits.writeto( sysargv4, newRGB_Image, overwrite=True)
                 # Save the RGB image as a new FITS file with the correct header
                 hdu = fits.PrimaryHDU(data=newRGB_Image, header=header)
                 hdu.writeto(sysargv4, overwrite=True)
           
                 # Function to read and verify the saved FITS file
                 def verify_fits(sysargv4):
                   with fits.open(sysargv4) as hdul:
                     data = hdul[0].data
                 return data
           
                 # Verify the saved RGB image
                 verified_image = verify_fits(sysargv4)
                 print("Verified image shape:", verified_image.shape)
           
                 #################################################################################
                 #################################################################################

      if __name__ == "__main__":
          main()           

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()
      deletefiles()

  return sysargv1
  menue()

def gaussian():

  try:

      def main():

                 sysargv2  = input("Enter the Color Image for GB -->")
           
                 # Function to read FITS file and return data
                 def read_fits(file):
                     hdul = fits.open(file)
                     header = hdul[0].header
                     data = hdul[0].data
                     hdul.close()
                     return data, header
           
                 # Read the FITS files
                 file1 = sysargv2
                 # Read the image data from the FITS file
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
           
                 # Split the color image into its individual channels
                 #b, g, r = cv2.split(image_data)
                 b, g, r = image_data[2, :, :], image_data[1, :, :], image_data[0, :, :] 
           
           
                 # Save each channel as a separate file
                 fits.writeto(f'channel_0_64bit.fits', b.astype(np.float64), header, overwrite=True)
                 fits.writeto(f'channel_1_64bit.fits', g.astype(np.float64), header, overwrite=True)
                 fits.writeto(f'channel_2_64bit.fits', r.astype(np.float64), header, overwrite=True)
           
                 sysargv2  = "channel_0_64bit.fits"
                 sysargv2g  = "channel_1_64bit.fits"
                 sysargv2r  = "channel_2_64bit.fits"
                 sysargv4a  = input("Enter the gausian blur 1.313, 2 etc.  -->")
                 sysargv4 = float(sysargv4a)
           
                 def apply_gaussian_blur(image, sigma):
                   # Apply Gaussian blur with specified sigma
                   blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
                   return blurred_image
           
                 def read_fits(file):
                       hdul = fits.open(file)
                       header = hdul[0].header
                       data = hdul[0].data
                       hdul.close()
                       return data, header
           
                   # Load the FITS blue file
                 file1 = sysargv2
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
           
                 # Normalize the image data
                 image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)
           
                 # Apply the Gaussian blur
                 sigma = sysargv4  # Standard deviation in X and Y directions
                 blurred_image = apply_gaussian_blur(image_data, sigma)
           
                 # Save the final image as a FITS file
                 fits.writeto(f'channel_0_64bitGB.fits', blurred_image.astype(np.float64), header, overwrite=True)
                 # Load the FITS green file
                 file1 = sysargv2g
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
           
                 # Normalize the image data
                 image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)
           
                 # Apply the Gaussian blur
                 sigma = sysargv4  # Standard deviation in X and Y directions
                 blurred_image = apply_gaussian_blur(image_data, sigma)
           
                 # Save the final image as a FITS file
                 fits.writeto(f'channel_1_64bitGB.fits', blurred_image.astype(np.float64), header, overwrite=True)    # Load the FITS red file
                 file1 = sysargv2r
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
                 # Normalize the image data
                 image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)
           
                 # Apply the Gaussian blur
                 sigma = sysargv4  # Standard deviation in X and Y directions
                 blurred_image = apply_gaussian_blur(image_data, sigma)
           
                 # Save the final image as a FITS file
                 fits.writeto(f'channel_2_64bitGB.fits', blurred_image.astype(np.float64), header, overwrite=True)
           
                 sysargv1  = "channel_0_64bitGB.fits"
                 sysargv2  = "channel_1_64bitGB.fits"
                 sysargv3  = "channel_2_64bitGB.fits"
                 sysargv4  = "channel_RGB_64bitGB.fits"
           
                 old_data, old_header = read_fits(file1)
                 old_data = old_data.astype(np.float64)
              
                 # Function to read FITS file and return data
                 def read_fits(file):
                   with fits.open(file, mode='update') as hdul:#
                     data = hdul[0].data
                     hdul.close()
                   return data
           
                 # Read the FITS files
                 file1 = sysargv1
                 file2 = sysargv2
                 file3 = sysargv3
           
                 # Read the image data from the FITS file
                 blue = read_fits(file1)
                 green = read_fits(file2)
                 red = read_fits(file3)
           
                 blue = blue.astype(np.float64)
                 green = green.astype(np.float64)
                 red = red.astype(np.float64)
           
                 # Check dimensions
                 print("Data1 shape:", blue.shape)
                 print("Data2 shape:", green.shape)
                 print("Data3 shape:", red.shape)
           
                 #newRGBImage = cv2.merge((red,green,blue))
                 RGB_Image1 = np.stack((red,green,blue))
           
                 # Remove the extra dimension
                 RGB_Image = np.squeeze(RGB_Image1)
           
                 # Create a FITS header with NAXIS = 3
                 header = old_header
                 header['NAXIS'] = 3
                 header['NAXIS1'] = RGB_Image.shape[0]
                 header['NAXIS2'] = RGB_Image.shape[1]
                 header['NAXIS3'] = RGB_Image.shape[2]
           
                 # Ensure the data type is correct 
                 newRGB_Image = RGB_Image.astype(np.float64)
           
                 print("newRGB_Image shape:", newRGB_Image.shape)
           
                 sysargv4 = "channel_RGB_64bitGB.fits"
                 fits.writeto( sysargv4, newRGB_Image, overwrite=True)
                 # Save the RGB image as a new FITS file with the correct header
                 hdu = fits.PrimaryHDU(data=newRGB_Image, header=header)
                 hdu.writeto(sysargv4, overwrite=True)
           
                 # Read and verify the saved FITS file
                 with fits.open(sysargv4) as hdul:
                   data = hdul[0].data
           
                 # Verify the saved RGB image
                 print("Verified image shape:", data.shape)
           
                 currentDirectory = os.path.abspath(os.getcwd())
           
                 # Define the file to delete
                 file_to_delete = "channel_0_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_1_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_2_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_0_64bitGB.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_1_64bitGB.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                       print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_2_64bitGB.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")

      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def FFT():

  try:

      def main():

                 sysargv2  = input("Enter the Color Image for FFT  -->")
           
                 # Function to read FITS file and return data
                 def read_fits(file):
                     hdul = fits.open(file)
                     header = hdul[0].header
                     data = hdul[0].data
                     hdul.close()
                     return data, header
           
                 # Read the FITS files
                 file1 = sysargv2
                 # Read the image data from the FITS file
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
           
                 # Split the color image into its individual channels
                 #b, g, r = cv2.split(image_data)
                 b, g, r = image_data[2, :, :], image_data[1, :, :], image_data[0, :, :] 
           
           
                 # Save each channel as a separate file
                 fits.writeto(f'channel_0_64bit.fits', b.astype(np.float64), header, overwrite=True)
                 fits.writeto(f'channel_1_64bit.fits', g.astype(np.float64), header, overwrite=True)
                 fits.writeto(f'channel_2_64bit.fits', r.astype(np.float64), header, overwrite=True)
           
                 sysargv2  = "channel_0_64bit.fits"
                 sysargv2g  = "channel_1_64bit.fits"
                 sysargv2r  = "channel_2_64bit.fits"
                 sysargv4  = input("Enter the cutoff(25)(NUM)  -->")
                 sysargv5  = input("Enter the weight(50)(NUM)   -->")
                 sysargv6  = input("Enter the Denominator(100)  -->")
                 sysargv7  = input("Enter the radius(1))  -->")
                 sysargv8  = input("Enter the cutoff(10))  -->")
           
                 #copilot output
                 def high_pass_filter(image, cutoff=(int(sysargv4)/int(sysargv6)), weight=(int(sysargv5)/int(sysargv6))):
                     # Perform FFT
                     fft = np.fft.fft2(image)
                     fft_shift = np.fft.fftshift(fft)
           
                     # Create a high-pass filter mask
                     rows, cols = image.shape
                     crow, ccol = rows // 2, cols // 2
                     mask = np.ones((rows, cols), np.float32)
                     r = int(cutoff * min(rows, cols))
                     center = (crow, ccol)
                     mask[center[0]-r:center[0]+r, center[1]-r:center[1]+r] = 0
           
                     # Apply the mask to the FFT shift
                     fft_shift_filtered = fft_shift * mask
             
                     # Perform inverse FFT
                     fft_inverse_shift = np.fft.ifftshift(fft_shift_filtered)
                     image_filtered = np.fft.ifft2(fft_inverse_shift)
                     image_filtered = np.abs(image_filtered)
           
                     # Weight the filtered image
                     image_weighted = cv2.addWeighted(image, 1 - weight, image_filtered.astype(np.float32), weight, 0)
           
                     return image_weighted
           
                 def feather_image(image, radius=int(sysargv7), distance=int(sysargv8)):
                     # Reduce radius
                     image_blurred = cv2.GaussianBlur(image, (radius, radius), 0)
               
                     # Create a mask with feathered edges covering the entire image
                     mask = np.zeros(image.shape, dtype=np.uint8)
                     mask[:,:] = 255  # Set all mask values to 255 (white)
               
                     # Apply distance feathering
                     mask_blurred = cv2.GaussianBlur(mask, (distance*2+1, distance*2+1), 0)
               
                     # Apply the mask to the image
                     result_image = cv2.bitwise_and(image_blurred, image_blurred, mask=mask_blurred)
           
                     return result_image
           
                 def read_fits(file):
                       hdul = fits.open(file)
                       header = hdul[0].header
                       data = hdul[0].data
                       hdul.close()
                       return data, header
           
                   # Load the FITS blue file
                 file1 = sysargv2
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
           
                 # Normalize the image data
                 image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)
           
                 # Apply high-pass filter
                 filtered_image = high_pass_filter(image_data, cutoff=(int(sysargv4)/int(sysargv6)), weight=(int(sysargv5)/int(sysargv6)))
           
                 # Apply feathering
                 final_image = feather_image(filtered_image, radius=int(sysargv7), distance=int(sysargv8))
           
                 # Save the final image as a FITS file
                 fits.writeto(f'channel_0_64bitfft.fits', final_image.astype(np.float64), header, overwrite=True)
                 # Load the FITS green file
                 file1 = sysargv2g
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
           
                 # Normalize the image data
                 image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)
           
                 # Apply high-pass filter
                 filtered_image = high_pass_filter(image_data, cutoff=(int(sysargv4)/int(sysargv6)), weight=(int(sysargv5)/int(sysargv6)))
           
                 # Apply feathering
                 final_image = feather_image(filtered_image, radius=int(sysargv7), distance=int(sysargv8))
           
                 # Save the final image as a FITS file
                 fits.writeto(f'channel_1_64bitfft.fits', final_image.astype(np.float64), header, overwrite=True)    # Load the FITS red file
                 file1 = sysargv2r
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
                 # Normalize the image data
                 image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)
           
                 # Apply high-pass filter
                 filtered_image = high_pass_filter(image_data, cutoff=(int(sysargv4)/int(sysargv6)), weight=(int(sysargv5)/int(sysargv6)))
           
                 # Apply feathering
                 final_image = feather_image(filtered_image, radius=int(sysargv7), distance=int(sysargv8))
           
                 # Save the final image as a FITS file
                 fits.writeto(f'channel_2_64bitfft.fits', final_image.astype(np.float64), header, overwrite=True)
           
                 sysargv1  = "channel_0_64bitfft.fits"
                 sysargv2  = "channel_1_64bitfft.fits"
                 sysargv3  = "channel_2_64bitfft.fits"
                 sysargv4  = "channel_RGB_64bitfft.fits"
           
                 old_data, old_header = read_fits(file1)
                 old_data = old_data.astype(np.float64)
              
                 # Function to read FITS file and return data
                 def read_fits(file):
                   with fits.open(file, mode='update') as hdul:#
                     data = hdul[0].data
                     hdul.close()
                   return data
           
                 # Read the FITS files
                 file1 = sysargv1
                 file2 = sysargv2
                 file3 = sysargv3
           
                 # Read the image data from the FITS file
                 blue = read_fits(file1)
                 green = read_fits(file2)
                 red = read_fits(file3)
           
                 blue = blue.astype(np.float64)
                 green = green.astype(np.float64)
                 red = red.astype(np.float64)
           
                 # Check dimensions
                 print("Data1 shape:", blue.shape)
                 print("Data2 shape:", green.shape)
                 print("Data3 shape:", red.shape)
           
                 #newRGBImage = cv2.merge((red,green,blue))
                 RGB_Image1 = np.stack((red,green,blue))
           
                 # Remove the extra dimension
                 RGB_Image = np.squeeze(RGB_Image1)
           
                 # Create a FITS header with NAXIS = 3
                 header = old_header
                 header['NAXIS'] = 3
                 header['NAXIS1'] = RGB_Image.shape[0]
                 header['NAXIS2'] = RGB_Image.shape[1]
                 header['NAXIS3'] = RGB_Image.shape[2]
           
                 # Ensure the data type is correct 
                 newRGB_Image = RGB_Image.astype(np.float64)
           
                 print("newRGB_Image shape:", newRGB_Image.shape)
           
                 sysargv4 = "channel_RGB_64bitfft.fits"
                 fits.writeto( sysargv4, newRGB_Image, overwrite=True)
                 # Save the RGB image as a new FITS file with the correct header
                 hdu = fits.PrimaryHDU(data=newRGB_Image, header=header)
                 hdu.writeto(sysargv4, overwrite=True)
           
                 # Read and verify the saved FITS file
                 with fits.open(sysargv4) as hdul:
                   data = hdul[0].data
           
                 # Verify the saved RGB image
                 print("Verified image shape:", data.shape)
           
                 currentDirectory = os.path.abspath(os.getcwd())
           
                 # Define the file to delete
                 file_to_delete = "channel_0_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_1_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_2_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_0_64bitfft.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_1_64bitfft.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                       print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_2_64bitfft.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def  LrDeconv():

  try:
 
      def richardson_lucy(image, psf, iterations=30):
          """
          Perform Richardson–Lucy deconvolution.
    
          Parameters:
            image : 2D ndarray
                The observed (blurred) image.
            psf : 2D ndarray
                The point-spread function; should be normalized.
            iterations : int
                Number of iterations to perform.
    
          Returns:
            deconv : 2D ndarray
                The deconvolved image.
          """
          image = image.astype(np.float64)
          im_est = image.copy()  # initial estimate
          psf_mirror = psf[::-1, ::-1]  # mirror PSF

          for i in range(iterations):
              conv_est = convolve_fft(im_est, psf, normalize_kernel=True)
              conv_est[conv_est == 0] = 1e-7  # Avoid divide-by-zero
              relative_blur = image / conv_est
              correction = convolve_fft(relative_blur, psf_mirror, normalize_kernel=True)
              im_est *= correction
          return im_est

      def extract_psf(image, position, size):
          """
          Extract a PSF from the image by making a cutout around a bright source.
    
          Parameters:
            image : 2D ndarray
                Input image.
            position : tuple
                (x, y) coordinates (in pixels) of the center of the bright star.
            size : int or tuple
                Size of the cutout; can be an integer or a (width, height) tuple.
    
          Returns:
            psf : 2D ndarray
                The normalized PSF extracted from the image.
          """
          cutout = Cutout2D(image, position, size)
          psf = cutout.data.copy()
          psf -= np.median(psf)
          psf[psf < 0] = 0
          psf /= psf.sum()
          return psf

      def main():
          sysargv1  = input("Enter the Image name  -->")
          sysargv2  = input("Enter the name of deconvoluted image name  -->")

          # Load the observed image from a FITS file.
          input_file = sysargv1  # Replace with your actual FITS file.
          with fits.open(input_file) as hdul:
              image = hdul[0].data

          # Check if the image is color.
          # Since we assume a color image has shape (3, height, width)
          if image.ndim == 3 and image.shape[0] == 3:
              is_color = True
          else:
              is_color = False

          # Visualization of the image to help pick the PSF region.
          if not is_color:
              norm = simple_norm(image, 'sqrt', percent=99)
              plt.figure(figsize=(6, 5))
              plt.imshow(image, norm=norm, origin='lower', cmap='gray')
              plt.title("Input Grayscale Image")
              plt.colorbar()
              plt.show()
          else:
              plt.figure(figsize=(6, 5))
              # Rearrange image from (3, height, width) to (height, width, 3) for display.
              plt.imshow(np.moveaxis(image, 0, -1), origin='lower')
              plt.title("Input Color Image")
              plt.show()

          # Choose PSF extraction method:
          #   Option A: Use an analytical PSF (Gaussian 2D kernel)
          #   Option B: Extract the PSF from the image.
          use_analytical_psf = input("Use analytical PSF [g] or extract from image [e]? ").strip().lower()
    
          if use_analytical_psf == 'g':
              sigma = 2.0           # Adjust sigma as needed
              kernel_size = 25      # Size of the PSF kernel
              gauss_kernel = Gaussian2DKernel(sigma, x_size=kernel_size, y_size=kernel_size)
              psf = gauss_kernel.array
              psf /= psf.sum()      # Normalize the PSF
              print("Using an analytical Gaussian PSF.")
    
          elif use_analytical_psf == 'e':
              x = float(input("Enter x coordinate of the PSF center: "))
              y = float(input("Enter y coordinate of the PSF center: "))
              size = float(input("Enter the size of the PSF cutout (in pixels): "))
              # For color images in shape (3, height, width), extract from the first channel.
              if is_color:
                  psf = extract_psf(image[0, :, :], (x, y), size)
              else:
                  psf = extract_psf(image, (x, y), size)
              print("Using the PSF extracted from the image.")
        
              plt.figure(figsize=(4, 4))
              plt.imshow(psf, origin='lower', cmap='viridis')
              plt.title("Extracted PSF")
              plt.colorbar()
              plt.show()
          else:
              print("Invalid option selected. Exiting.")
              return

          # Perform Richardson–Lucy deconvolution with the chosen PSF.
          iterations = int(input("Enter the number of iterations (e.g., 30): "))
    
          if not is_color:
              deconv_image = richardson_lucy(image, psf, iterations=iterations)
              output_file = 'deconvolved_image.fits'
              fits.writeto(output_file, deconv_image, overwrite=True)
              print("Deconvolved image saved to", output_file)
    
              plt.figure(figsize=(12, 5))
              plt.subplot(1, 2, 1)
              plt.imshow(image, origin='lower', cmap='gray', norm=simple_norm(image, 'sqrt', percent=99))
              plt.title("Original Grayscale Image")
              plt.colorbar()
    
              plt.subplot(1, 2, 2)
              plt.imshow(deconv_image, origin='lower', cmap='gray', norm=simple_norm(deconv_image, 'sqrt', percent=99))
              plt.title("Deconvolved Image")
              plt.colorbar()
              plt.tight_layout()
              plt.show()
    
          else:
              # For color images assuming shape (3, height, width)
              deconv_image = np.empty_like(image)
              print("Processing each channel (assuming image shape is (3, height, width)):")
              # Iterate over axis 0 (channels)
              for c in range(image.shape[0]):
                  print(f"Deconvolving channel {c}...")
                  deconv_image[c, :, :] = richardson_lucy(image[c, :, :], psf, iterations=iterations)
              output_file = sysargv2
              fits.writeto(output_file, deconv_image, overwrite=True)
              print("Deconvolved color image saved to", output_file)
    
              plt.figure(figsize=(12, 5))
              plt.subplot(1, 2, 1)
              # Display original color image by moving axis 0 to the end.
              plt.imshow(np.moveaxis(image, 0, -1), origin='lower')
              plt.title("Original Color Image")
    
              plt.subplot(1, 2, 2)
              plt.imshow(np.moveaxis(deconv_image, 0, -1), origin='lower')
              plt.title("Deconvolved Color Image")
              plt.tight_layout()
              plt.show()

      if __name__ == '__main__':
          main()
  
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()


  return sysargv1
  menue()

def erosion():
    try:
        def main():
            # Do not change these data entry instructions
            sysargv2  = input("Enter the input file name --> ")
            sysargv3  = input("Enter number of iterations example 3,5,7 --> ")
            sysargv4  = input("Enter (Kernel)structuring element of radius example 3,5,7 --> ")
            
            # Read the input FITS file
            hdul = fits.open(sysargv2)
            header = hdul[0].header
            data = hdul[0].data.astype(np.float64)
            hdul.close()
            
            # Assume the data are stored as (height, width, channels)
            if data.ndim == 3:
                height, width, channels = data.shape
                processed_channels = []
                for ch in range(channels):
                    channel_data = data[:, :, ch]
                    ksize = int(sysargv4)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                    iterations = int(sysargv3)
                    eroded = cv2.erode(channel_data, kernel, iterations=iterations)
                    processed_channels.append(eroded)
                result = np.stack(processed_channels, axis=2)
            elif data.ndim == 2:
                ksize = int(sysargv4)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                iterations = int(sysargv3)
                result = cv2.erode(data, kernel, iterations=iterations)
            else:
                print("Unsupported data dimensions.")
                return
            
            # Write the processed (eroded) image to a new FITS file
            output_file = "erosion_out.fits"
            fits.writeto(output_file, result.astype(np.float64), header, overwrite=True)
            print("Erosion completed. Output saved to", output_file)
        
        if __name__ == "__main__":
            main()
            
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Returning to the Main Menue...")
        return sysargv1
        menue()
    
    return sysargv1
    menue()

def jpgcomp():

  try:

      def main():

                 sysargv2  = input("Enter the input file name --> ")
                 sysargv3  = input("Enter number percent to compress to (10) ")
                 sysargv4  = input("Enter the output file name --> ")
           
                 # Read the image as grayscale
                 image = cv2.imread(sysargv2)
                 cv2.imwrite(sysargv4, image, [cv2.IMWRITE_JPEG_QUALITY, int(sysargv3)])

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()


def dilation():

  try:

      def main():

                 sysargv2  = input("Enter the input file name --> ")
                 sysargv3  = input("Enter number of iterations example 3,5,7 --> ")
                 sysargv4  = input("Enter (Kernel)structuring element of radius example 3,5,7 --> ")
           
                 # Create a disk-shaped structuring element of radius 5
                 # Read the image as grayscale
                 img = (io.imread(sysargv2))
                 # Define a kernel (a matrix of odd size) for the erosion operation
                 # You can choose different shapes and sizes for the kernel
                 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ((int(sysargv4)), (int(sysargv4))))
           
                 # Apply the erosion operation using cv2.erode()
                 # You can adjust the number of iterations for more or less erosion
                 img_dilated = cv2.dilate(img, kernel, iterations=(int(sysargv3)))
           
                 cv2.imshow('output', img)
                 cv2.imshow('Dilation', img_dilated)
           
                 # Wait for a key press to exit
                 cv2.waitKey(0)
                 # close the window 
                 cv2.destroyAllWindows() 

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def imgcrop1():

  try:

      def main():

                 sysargv1  = input("Enter the Image to crop  -->")
                 sysargv3  = input("Enter the filename of the image to save  -->")
           
                 # Opens a image in RGB mode
                 img = cv2.imread(sysargv1)
           
                 # Size of the image in pixels (size of original image)
                 # (This is not mandatory)
                 height, width = img.shape[:2]
                 print("width", width)
                 print("height", height)
           
                 # Setting the points for cropped image
                 sysargv4  = input("Enter x point of new image -->")
                 sysargv5  = input("Enter y point of new image   -->")
                 sysargv6  = input("Enter width of new image   -->")
                 sysargv7  = input("Enter height of new image  -->")
                 x = int(sysargv4)
                 y = int(sysargv5)
           
                 cropped_image = img[y:y+int(height), x:x+int(width)]
           
                 # Cropped image of above dimension
                 # (It will not change original image)
                 cv2.imshow('image',cropped_image)
                 cv2.waitKey(0)  
                 cv2.imwrite(sysargv3, cropped_image)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def imghiststretch():

  try:

      # ------------------------------
      # Histogram Specification Functions
      # ------------------------------

      def rayleigh_specification(image, sigma):
          """
          Transform the image so its histogram matches a Rayleigh distribution.
    
          Inverse Rayleigh CDF: F⁻¹(p; σ) = σ * sqrt(-2 * ln(1-p))
          """
          vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
          cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
          epsilon = 1e-10
          safe_vals = np.clip(1 - cdf, epsilon, None)
          new_vals = sigma * np.sqrt(-2 * np.log(safe_vals))
          spec_img = new_vals[inv_idx].reshape(image.shape)
          return spec_img

      def gaussian_specification(image, mu, sigma):
          """
          Transform the image so its histogram matches a Gaussian distribution.
    
          Inverse Gaussian CDF: F⁻¹(p; μ,σ) = μ + σ * sqrt(2) * erfinv(2p - 1)
          """
          vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
          cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
          epsilon = 1e-10
          safe_cdf = np.clip(cdf, epsilon, 1 - epsilon)
          new_vals = mu + sigma * np.sqrt(2) * erfinv(2 * safe_cdf - 1)
          spec_img = new_vals[inv_idx].reshape(image.shape)
          return spec_img

      def uniform_specification(image, lower, upper):
          """
          Transform the image so its histogram is uniformly distributed.
    
          Inverse Uniform CDF: F⁻¹(p) = lower + p * (upper - lower)
          """
          vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
          cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
          new_vals = lower + cdf * (upper - lower)
          spec_img = new_vals[inv_idx].reshape(image.shape)
          return spec_img

      def exponential_specification(image, lamb):
          """
          Transform the image so its histogram matches an exponential distribution.
    
          Inverse Exponential CDF: F⁻¹(p; λ) = - (1/λ) * ln(1 - p)
          """
          vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
          cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
          epsilon = 1e-10
          safe_vals = np.clip(1 - cdf, epsilon, None)
          new_vals = - (1 / lamb) * np.log(safe_vals)
          spec_img = new_vals[inv_idx].reshape(image.shape)
          return spec_img

      def lognormal_specification(image, mu, sigma_ln):
          """
          Transform the image so its histogram matches a lognormal distribution.
    
          Inverse Lognormal CDF: 
             F⁻¹(p; μ,σ) = exp( μ + σ * sqrt(2) * erfinv(2p - 1) )
          """
          vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
          cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
          epsilon = 1e-10
          safe_cdf = np.clip(cdf, epsilon, 1 - epsilon)
          new_vals = np.exp(mu + sigma_ln * np.sqrt(2) * erfinv(2 * safe_cdf - 1))
          spec_img = new_vals[inv_idx].reshape(image.shape)
          return spec_img

      # ------------------------------
      # Main Routine
      # ------------------------------

      def main():
          # Load the input FITS image (update the path as needed)
          sysargv2  = input("Enter the image(fits Siril)  -->")
          input_fits_file = sysargv2
          with fits.open(input_fits_file) as hdul:
              image = hdul[0].data.astype(np.float64)
    
          # Choose the transformation type.
          prompt = (
              "Choose histogram specification type:\n"
              "  (r) Rayleigh\n"
              "  (g) Gaussian\n"
              "  (u) Uniform\n"
              "  (e) Exponential\n"
              "  (l) Lognormal\n"
              "Enter one of r, g, u, e, l: "
          )
          transform_type = input(prompt).strip().lower()
    
          # Compute parameters from the image or ask the user.
          if transform_type == 'r':
              sigma_val = np.std(image)
              print(f"Applying Rayleigh specification with sigma = {sigma_val:.3f}")
              specified_image = rayleigh_specification(image, sigma_val)
              output_fits_file = 'image_rayleigh_specified.fits'
    
          elif transform_type == 'g':
              mu_val = np.mean(image)
              sigma_val = np.std(image)
              print(f"Applying Gaussian specification with mu = {mu_val:.3f} and sigma = {sigma_val:.3f}")
              specified_image = gaussian_specification(image, mu_val, sigma_val)
              output_fits_file = 'image_gaussian_specified.fits'
    
          elif transform_type == 'u':
              # For uniform, specify lower and upper limits.
              # For example, map to the range of the original image.
              lower = float(input("Enter lower bound (e.g., 0): "))
              upper = float(input("Enter upper bound (e.g., 1): "))
              print(f"Applying Uniform specification with lower = {lower} and upper = {upper}")
              specified_image = uniform_specification(image, lower, upper)
              output_fits_file = 'image_uniform_specified.fits'
    
          elif transform_type == 'e':
              lamb = float(input("Enter lambda (e.g., 0.1): "))
              print(f"Applying Exponential specification with lambda = {lamb}")
              specified_image = exponential_specification(image, lamb)
              output_fits_file = 'image_exponential_specified.fits'
    
          elif transform_type == 'l':
              mu_val = float(input("Enter mu for lognormal (e.g., 0): "))
              sigma_ln = float(input("Enter sigma for lognormal (e.g., 0.5): "))
              print(f"Applying Lognormal specification with mu = {mu_val} and sigma = {sigma_ln}")
              specified_image = lognormal_specification(image, mu_val, sigma_ln)
              output_fits_file = 'image_lognormal_specified.fits'
    
          else:
              print("Invalid choice. Exiting.")
              return
    
          # Save the specified image to a new FITS file.
          fits.writeto(output_fits_file, specified_image, overwrite=True)
          print(f"Specified image saved as {output_fits_file}")
    
          # Plot histograms for comparison
          fig, axes = plt.subplots(1, 2, figsize=(12, 5))
          axes[0].hist(image.ravel(), bins=256, color='blue', histtype='step')
          axes[0].set_title("Original Image Histogram")
          axes[0].set_xlabel("Intensity")
          axes[0].set_ylabel("Frequency")
    
          axes[1].hist(specified_image.ravel(), bins=256, color='red', histtype='step')
    
          title_dict = {
              'r': "Rayleigh Specified Histogram",
              'g': "Gaussian Specified Histogram",
              'u': "Uniform Specified Histogram",
              'e': "Exponential Specified Histogram",
              'l': "Lognormal Specified Histogram"
          }
          axes[1].set_title(title_dict.get(transform_type, "Specified Histogram"))
          axes[1].set_xlabel("Intensity")
          axes[1].set_ylabel("Frequency")
    
          plt.tight_layout()
          plt.show()

      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def gif():

  try:

      def main():

                 sysargv1  = input("jpg only Enter Image width  -->")
                 sysargv2  = input("jpg only Enter Image height  -->")
                 sysargv3  = input("Enter Gif to save -->")
                 sysargv4  = input("Enter (*.jpg)etc to use for Gif -->")
                 sysargv5  = input("Enter  image duration in millisconds -->")
                 fGIF = sysargv3
                 W = int(sysargv1)
                 H = int(sysargv2)
                 # Create the frames
                 frames = []
                 images = glob.glob(sysargv4)
           
                 for i in images:
                     newImg = Image.open(i)
                     frames.append(newImg)
            
           
                 # Save into a GIF file that loops forever: duration is in milli-second
                 frames[0].save(fGIF, format='GIF', optimize=False, append_images=frames[1:],
                     save_all=True, duration=int(sysargv5), loop=0)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def video():

  try:

      def main():

                 sysargv3  = input("Enter file name to save mp4 Video -->")
                 sysargv3a = input("Enter file name to save gif Video -->")
                 sysargv3b = input("Enter scale (800) pixels size to save gif Video -->")
                 sysargv4  = input("Enter (*.jpg)etc to use for Video -->")
                 sysargv5  = input("Enter frames per second -->")
           
                 img_array = []
                 for filename in glob.glob(sysargv4):
                   img = cv2.imread(filename)
                   height, width, layers = img.shape
                   size = (width,height)
                   img_array.append(img)
           
                 # Save into a video file duration is in fps
                 out = cv2.VideoWriter(sysargv3,cv2.VideoWriter_fourcc(*'mp4v'), int(sysargv5), size)
             
                 for i in range(len(img_array)):
                   out.write(img_array[i])
                 out.release()
           
                 ffmpeg.input(sysargv3).filter('scale', sysargv3b, -1).output(sysargv3a).run()

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()
  
  return sysargv1
  menue()

def alingimg():

  try:

      def main():

                 sysargv2  = input("jpg enter reference Image  -->")
                 sysargv3  = input("jpg enter alignment Image  -->")
                 sysargv4  = input("Enter Image to save -->")
           
                 # Open the image files.
                 img1_color = cv2.imread(sysargv3)  # Image to be aligned.
                 img2_color = cv2.imread(sysargv2)    # Reference image.
           
                 # Convert to grayscale.
                 img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
                 img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
                 height, width = img2.shape
           
                 # Create ORB detector with 5000 features.
                 orb_detector = cv2.ORB_create(5000)
           
                 # Find keypoints and descriptors.
                 # The first arg is the image, second arg is the mask
                 #  (which is not required in this case).
                 kp1, d1 = orb_detector.detectAndCompute(img1, None)
                 kp2, d2 = orb_detector.detectAndCompute(img2, None)
           
                 # Match features between the two images.
                 # We create a Brute Force matcher with 
                 # Hamming distance as measurement mode.
                 matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
           
                 # Match the two sets of descriptors.
                 matches = matcher.match(d1, d2)
           
                 # Sort matches on the basis of their Hamming distance.
                 matches.sort(key = lambda x: x.distance)
           
                 # Take the top 90 % matches forward.
                 matches = matches[:int(len(matches)*0.9)]
                 no_of_matches = len(matches)
           
                 # Define empty matrices of shape no_of_matches * 2.
                 p1 = np.zeros((no_of_matches, 2))
                 p2 = np.zeros((no_of_matches, 2))
           
                 for i in range(len(matches)):
                   p1[i, :] = kp1[matches[i].queryIdx].pt
                   p2[i, :] = kp2[matches[i].trainIdx].pt
           
                 # Find the homography matrix.
                 homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
           
                 # Use this matrix to transform the
                 # colored image wrt the reference image.
                 transformed_img = cv2.warpPerspective(img1_color,
                                 homography, (width, height))
           
                 # Save the output.
                 cv2.imwrite(sysargv4, transformed_img)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def gamma():

  try:

      def main():

                 sysargv1  = input("Enter the image(fits Siril)  -->")
                 sysargv5  = input("Enter the final image name progrm will output a .fit   -->") 
                 gamma     = float(input("Enter gamma(.3981) for 1 magnitude  -->"))
                 # Replace 'your_fits_file.fits' with the actual path to your FITS file
                 fits_image_filename = sysargv1
                 # Open the FITS file
                 with fits.open(fits_image_filename) as hdul:
                     # Access the primary HDU (extension 0)
                     image_data = hdul[0].data
                 # Now 'image_data' contains the data from the FITS file as a 2D numpy array
                 hdul.close()
           
                 print(image_data.shape)
                 print(image_data.dtype.name)
           
                 my_data = (image_data * 65535)
             
                 for gamma in [float(gamma)]: 
                   # Apply gamma correction. 
                   gamma_corrected1 = gamma_corrected = np.array(((65535.0 *(my_data / 65535) ** gamma/65535)/100), dtype = 'float64') 
                   #gamma_corrected = (np.round(gamma_corrected1))
                 #cv2.imwrite(str(sysargv5)+'gamma_corrected'+'.tif', gamma_corrected)  
           
                 hdu = fits.PrimaryHDU(gamma_corrected)
                 # Create an HDU list and add the primary HDU
                 hdulist = fits.HDUList([hdu])
                 # Specify the output FITS file path
                 output_fits_filename = sysargv5
                 # Write the HDU list to the FITS file
                 hdulist.writeto(str(sysargv5)+'gamma_corrected'+'.fit', overwrite=True)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def add2images():

  try:

      def main():

                 sysargv2  = input("Enter the first (default fits Siril) Image  -->")
                 sysargv3  = input("Enter the second (default fits Siril) Image  -->")
           
                 # Load the first FITS file
                 hdul1 = fits.open(sysargv2)
                 image_data1 = hdul1[0].data.astype(np.float64)
               
                 # Load the second FITS file
                 hdul2 = fits.open(sysargv3)
                 image_data2 = hdul2[0].data.astype(np.float64)
               
                 # Check if the image data from both FITS files have the same shape
                 if image_data1.shape != image_data2.shape:
                   print("Error: The input images do not have the same dimensions!")
                   hdul1.close()
                   hdul2.close()
                   return
                 print(image_data1.shape)
                 print(image_data2.shape)
                  #Add the RGB channels from both images
                  #Assuming both images are in the format (height, width, 3) (RGB)
                 if image_data1.ndim == 3 and image_data2.ndim == 3 :
                  # Add the corresponding channels (R, G, B) of both images
                   sysargv4  = input("Enter the filename of the added images to save  -->")
                   sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
                   sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
                   sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
                   sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")
                   sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
                   image_data1_contrastadd = int(sysargv5)
                   image_data1_contrastscale = (int(sysargv6)/int(sysargv7))
                   image_data2_contrastscale = (int(sysargv8)/int(sysargv9))
           
                   result_image = ((image_data1 * image_data1_contrastscale ) + (image_data2 * image_data1_contrastscale)) + image_data1_contrastadd
           
                   #result_image = image_data1 + image_data2
           
                   # Create a new FITS HDU for the result
                   result_hdu = fits.PrimaryHDU(result_image)
                   
                   # Create an HDU list (this only contains the result HDU)
                   hdulist = fits.HDUList([result_hdu])
                   
                   # Save the result as a new FITS file
                   hdulist.writeto(sysargv4, overwrite=True)
                   
                 else:
                   print("Error: The FITS files do not appear to be in the expected RGB format.")
               
                 # Close the FITS files
                   hdul1.close()
                   hdul2.close()

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def subtract2images():

  try:

      def main():

                 sysargv2  = input("Enter the first (default fits Siril) Image  -->")
                 sysargv3  = input("Enter the second (default fits Siril) Image  -->")
           
                 # Load the first FITS file
                 hdul1 = fits.open(sysargv2)
                 image_data1 = hdul1[0].data.astype(np.float64)
               
                 # Load the second FITS file
                 hdul2 = fits.open(sysargv3)
                 image_data2 = hdul2[0].data.astype(np.float64)
               
                 # Check if the image data from both FITS files have the same shape
                 if image_data1.shape != image_data2.shape:
                   print("Error: The input images do not have the same dimensions!")
                   hdul1.close()
                   hdul2.close()
                   return
                 print(image_data1.shape)
                 print(image_data2.shape)
                  #Add the RGB channels from both images
                  #Assuming both images are in the format (height, width, 3) (RGB)
                 if image_data1.ndim == 3 and image_data2.ndim == 3 :
                  # Add the corresponding channels (R, G, B) of both images
                   sysargv4  = input("Enter the filename of the added images to save  -->")
                   sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
                   sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
                   sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
                   sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")
                   sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
                   image_data1_contrastadd = int(sysargv5)
                   image_data1_contrastscale = (int(sysargv6)/int(sysargv7))
                   image_data2_contrastscale = (int(sysargv8)/int(sysargv9))
           
                   result_image = ((image_data1 * image_data1_contrastscale ) - (image_data2 * image_data1_contrastscale)) + image_data1_contrastadd
           
                   #result_image = image_data1 + image_data2
           
                   # Create a new FITS HDU for the result
                   result_hdu = fits.PrimaryHDU(result_image)
                   
                   # Create an HDU list (this only contains the result HDU)
                   hdulist = fits.HDUList([result_hdu])
                   
                   # Save the result as a new FITS file
                   hdulist.writeto(sysargv4, overwrite=True)
                   
                 else:
                   print("Error: The FITS files do not appear to be in the expected RGB format.")
               
                 # Close the FITS files
                   hdul1.close()
                   hdul2.close()

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def clahe():

  try:

      def main():

                 sysargv2  = input("Enter file name of color image to enter(16bit tif/png/fit) -->")
                 sysargv3  = input("Enter clip limit (3) -->")
                 sysargv4  = input("Enter tile Grid Size (8) -->")
                 sysargv5  = input("Enter output filename -->")
                 sysargv7  = input("Enter 0 for fits or 1 for other file -->")
           
                 if sysargv7 == '0':
           
                   # Function to read FITS file and return data
                   # Read the FITS file
                   hdulist = fits.open(sysargv2)
                   header = hdulist[0].header
                   image_data = hdulist[0].data
                   hdulist.close()
           
                   #image_data = np.swapaxes(image_data, 0, 2)
                   #image_data = np.swapaxes(image_data, 0, 1)
                   image_data = np.transpose(image_data, (1, 2, 0))
           
                   # Normalize the image data to the range [0, 65535]
                   image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 65535
                   image_data = image_data.astype(np.uint16)
           
                   # Convert the image to BGR format (OpenCV uses BGR by default)
                   image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
           
                   # Apply CLAHE to each channel
                   clahe = cv2.createCLAHE(clipLimit=int(sysargv3), tileGridSize=(int(sysargv4), int(sysargv4)))
                   channels = cv2.split(image_bgr)
                   clahe_channels = [clahe.apply(channel) for channel in channels]
                   clahe_image = cv2.merge(clahe_channels)
           
                   # Save or display the result
                   image_rgb = np.transpose(clahe_image, (2, 0, 1))
           
                   # Create a FITS HDU
                   hdu = fits.PrimaryHDU(image_rgb, header)
           
                   # Write to FITS file
                   hdu.writeto(sysargv5,  overwrite=True)
               
                   # Save or display the result
                   #cv2.imshow('CLAHE Image', clahe_image)
                   #cv2.waitKey(0)
                   #cv2.destroyAllWindows()
           
           
                 if sysargv7 == '1':
           
                   colorimage = cv2.imread(sysargv2, -1) 
                   clahe_model = cv2.createCLAHE(clipLimit=int(sysargv3), tileGridSize=(int(sysargv4),int(sysargv4)))
                   colorimage_b = clahe_model.apply(colorimage[:,:,0])
                   colorimage_g = clahe_model.apply(colorimage[:,:,1])
                   colorimage_r = clahe_model.apply(colorimage[:,:,2])
                   colorimage_clahe = np.stack((colorimage_b,colorimage_g,colorimage_r), axis=2)
                   cv2.imwrite(sysargv5, colorimage_clahe)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()
  
  return sysargv1
  menue()

def pm_vector_line():

  try:

      def main():

                 sysargv2  = input("Enter file name of color image to enter -->")
                 sysargv3  = input("Enter starting point(x) -->")
                 sysargv4  = input("Enter starting point(y) -->")
                 sysargv5  = input("Enter ending point(mas_x) -->")
                 sysargv6  = input("Enter ending point(mas_y) -->")
                 sysargv7  = input("Enter color b val(255) -->")
                 sysargv8  = input("Enter color g val(255) -->")
                 sysargv9  = input("Enter color r val(255) -->")
                 sysargv10  = input("Enter thickness(1) -->")
                 sysargv11  = input("Enter file name of color image to save -->")
           
                 # Start coordinate, here (0, 0)
                 # represents the top left corner of image
                 start_point = (int(sysargv3), int(sysargv4))
           
                 # End coordinate, here (int(sysargv5), int(sysargv6))
                 # represents the bottom right corner of image
                 end_point = (int(sysargv3) + (int(sysargv5) * -1), int(sysargv4) + (int(sysargv6) * -1))
           
                 # Green color in BGR
                 color = (int(sysargv7), int(sysargv8), int(sysargv9))
           
                 # Line thickness of sysargv10 px
                 thickness = int(sysargv10)
           
             
                 colorimage = cv2.imread(sysargv2, -1) 
                 # Using cv2.line() method
                 # Draw a diagonal green line with thickness of 9 px
                 image = cv2.line(colorimage, start_point, end_point, color, thickness)
                 cv2.imwrite(sysargv11, image)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()
  
  return sysargv1
  menue()

def hist_match():

  try:

      def main():

                 sysargv1  = input("Enter the reference Image  -->")
                 sysargv3  = input("Enter the Image  -->")
                 sysargv4  = input("Enter the filename of the added images to save  -->")
           
                 # Load example images
                 reference = cv2.imread(sysargv1, -1)
                 image = cv2.imread(sysargv3, -1)
           
                 # Perform histogram matching
                 matched = match_histograms(image, reference, channel_axis=-1)
           
                 cv2.imwrite(sysargv4, matched)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def distance():

  try:

      def main():

                 sysargv1  = float(input("parallax angle in milliarcseconds  -->"))
           
                 distancepar = 1 / (sysargv1 / 1000)
                 distanceltyr = 3.26 * distancepar
                 print('distance parsecs', distancepar)
                 print('distance light year', distanceltyr)
           
      if __name__ == "__main__":
          main()           
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def edgedetect():

  try:

      def main():

                 sysargv1  = input("Enter the filename of the 3x(jpg) Image  -->")
                 sysargv2  = input("Enter the filename of Edge sbl/cny to save  -->")
                 sysargv3  = input("Enter the lower threshold(100)  -->")
                 sysargv4  = input("Enter the upper threshold(200)  -->")
           
                 # Read the original image
                 img = cv2.imread(sysargv1, -1) 
           
                 # Convert to graycsale
                 img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                 # Blur the image for better edge detection
                 img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
            
                 # Sobel Edge Detection
                 sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
                 sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
                 sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection 
                 img_sobel_xy = cv2.merge([sobelxy, sobelxy, sobelxy])
                 cv2.imwrite(sysargv2 + "sbl.jpg", img_sobel_xy)
                 imgcny = cv2.Canny(img_blur, int(sysargv3), int(sysargv4))
                 img_cny = cv2.merge([imgcny, imgcny, imgcny])
                 cv2.imwrite(sysargv2 + "cny.jpg", img_cny)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def mosaic():

  try:

      def main():

                 sysargv2  = input("Enter file name of image 1 -->")
                 sysargv3  = input("Enter file name of image 2 -->")
                 sysargv4  = input("Enter file name of image 3 -->")
                 sysargv5  = input("Enter file name of image 4 -->")
                 sysargv6  = input("Enter file name of color image to save -->")
                 sysargv7  = input("Enter 0 for fits or 1 for other file -->")
           
                 if sysargv7 == '0':
           
                   # Function to read FITS file and return data
                   def read_fits(file):
                       hdul = fits.open(file)
                       data = hdul[0].data
                       hdul.close()
                       return data
           
                   # Read the FITS files
                   file1 = sysargv2
                   file2 = sysargv3
                   file3 = sysargv4
                   file4 = sysargv5
           
                   data1 = read_fits(file1)
                   data2 = read_fits(file2)
                   data3 = read_fits(file3)
                   data4 = read_fits(file4)
           
                   # Check dimensions
                   print("Data1 shape:", data1.shape)
                   print("Data2 shape:", data2.shape)
                   print("Data3 shape:", data3.shape)
                   print("Data4 shape:", data4.shape)
           
                   # Combine the images into a mosaic (2x2 grid)
                   mosaic = np.block([[data1, data2], [data3, data4]])
           
                   # Save the mosaic to a new FITS file
                   hdu = fits.PrimaryHDU(mosaic)
                   hdul = fits.HDUList([hdu])
                   hdul.writeto(sysargv6, overwrite=True)
           
                   print("Mosaic FITS file saved successfully!")
           
                 if sysargv7 == '1':
           
                   # Read the four images
                   img1 = cv2.imread(sysargv2)
                   img2 = cv2.imread(sysargv3)
                   img3 = cv2.imread(sysargv4)
                   img4 = cv2.imread(sysargv5)
           
                   # Get the size of the images
                   height, width, _ = img1.shape
           
                   # Create a new image with double the width and height
                   mosaic = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
           
                   # Place the images in the mosaic
                   mosaic[0:height, 0:width] = img1
                   mosaic[0:height, width:width*2] = img2
                   mosaic[height:height*2, 0:width] = img3
                   mosaic[height:height*2, width:width*2] = img4
           
                   # Save the mosaic
                   cv2.imwrite( sysargv6, mosaic)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def imgqtr():

  try:

      def main():

                 sysargv2  = input("Enter file name of image(.tif/.fit) -->")
                 sysargv7  = input("Enter 0 for fits or 1 for other file -->")
           
                 if sysargv7 == '0':
           
                   # Function to read FITS file and return data
                   def read_fits(file1):
                     with fits.open(file1) as hdul:
                       data = hdul[0].data
                       header = hdul[0].header
                       hdul.close()
                       return data
           
                   # Read the FITS files
                   file1 = sysargv2
           
                   data1 = read_fits(file1)
           
                   image_data = data1.astype(np.float64)
                   data_range = np.max(image_data) - np.min(image_data)
                   if data_range == 0:
                     normalized_data = np.zeros_like(image_data)  # or handle differently
                   else:
                     normalized_data = (image_data - np.min(image_data)) / data_range
                   data1 = normalized_data
           
                   # Get the dimensions of the image
                   channels, height, width = data1.shape
           
                   # Calculate the dimensions of each quarter
                   quarter_width = width // 2
                   quarter_height = height // 2
           
                   # Crop the image into four quarters for each channel
                   top_left = data1[:, :quarter_height, :quarter_width]
                   fits.writeto('mosaic_top_left_1.fits', top_left, overwrite=True)
                   top_right = data1[:, :quarter_height, quarter_width:]
                   fits.writeto('mosaic_top_right_2.fits', top_right, overwrite=True)
           
                   # Get the dimensions of the image
                   channels, height, width = data1.shape
           
                   # Calculate the dimensions of each quarter
                   quarter_width = width // 2
                   quarter_height = height // 2
           
                   # Crop the image into four quarters for each channel
                   bottom_left = data1[:, quarter_height:, :quarter_width]
                   fits.writeto('mosaic_bottom_left_3.fits', bottom_left, overwrite=True)
                   bottom_right = data1[:, quarter_height:, quarter_width:]
                   fits.writeto('mosaic_bottom_right_4.fits', bottom_right, overwrite=True)
           
                 if sysargv7 == '1':
           
                   image = cv2.imread(sysargv2)
           
                   # Get the dimensions of the image
                   height, width = image.shape[:2]
           
                   # Calculate the dimensions of each quarter
                   quarter_width = width // 2
                   quarter_height = height // 2
           
                   # Crop the image into four quarters
                   top_left = image[0:quarter_height, 0:quarter_width]
                   top_right = image[0:quarter_height, quarter_width:width]
                   bottom_left = image[quarter_height:height, 0:quarter_width]
                   bottom_right = image[quarter_height:height, quarter_width:width]
           
                   # Save the quarters as separate images
                   cv2.imwrite(sysargv2 + 'mosaictop_left.tif', top_left)
                   cv2.imwrite(sysargv2 + 'mosaictop_right.tif', top_right)
                   cv2.imwrite(sysargv2 + 'mosaicbottom_left.tif', bottom_left)
                   cv2.imwrite(sysargv2 + 'mosaicbottom_right.tif', bottom_right)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()


def CpyOldHdr():

  def print_and_patch_wcs(header, bin_factor):
      """
      Update header in place to add both CD-matrix + explicit CDELT/CUNIT.
      """
      w = WCS(header)

      # compute new CRPIX, CDELT
      orig_crpix = w.wcs.crpix.copy()
      orig_cdelt = w.wcs.cdelt.copy()
      new_crpix  = (orig_crpix - 0.5) / bin_factor + 0.5
      new_cdelt  = orig_cdelt * bin_factor

      # print for user
      print("Original CRPIX:", orig_crpix)
      print("Binned   CRPIX:", new_crpix)
      print("Original CDELT:", orig_cdelt)
      print("Binned   CDELT:", new_cdelt)

      # update WCS object
      w.wcs.crpix = new_crpix
      w.wcs.cdelt = new_cdelt

      # build two partial headers
      hdr_cd    = w.to_header(relax=True)   # CD1_1…CD2_2
      hdr_cdelt = fits.Header()
      hdr_cdelt["CDELT1"]  = (new_cdelt[0], "deg/pix")
      hdr_cdelt["CDELT2"]  = (new_cdelt[1], "deg/pix")
      hdr_cdelt["CUNIT1"]  = ("deg", "units of CRVAL and CDELT")
      hdr_cdelt["CUNIT2"]  = ("deg", "units of CRVAL and CDELT")
      hdr_cdelt["RADESYS"] = (w.wcs.radesys or "ICRS", "frame")
      hdr_cdelt["EQUINOX"] = (w.wcs.equinox,            "equinox")

      # merge into one header
      out = header.copy()
      for k in hdr_cd:
          out[k] = hdr_cd[k]
      for k in hdr_cdelt:
          out[k] = hdr_cdelt[k]

      return out

  def main():
      try:
          # 1) prompt for files + bin
          old_header_file = input("Old-header FITS file: ").strip()
          new_image_file  = input("New image FITS file: ").strip()
          output_file     = input("Output FITS file: ").strip()
          bin_val         = int(input("Binning factor (0 to skip): ").strip())

          # 2) load old header
          with fits.open(old_header_file) as h:
              old_header = h[0].header.copy()

          # 3) if bin>0, patch WCS
          if bin_val > 0:
              header_to_write = print_and_patch_wcs(old_header, bin_val)
          else:
              header_to_write = old_header

          # 4) load new image data
          with fits.open(new_image_file) as h:
              new_data = h[0].data

          # 5) write out exactly once
          fits.writeto(output_file, new_data, header_to_write, overwrite=True)
          print(f"Success: wrote {output_file}")

      except Exception as e:
          print(f"Error: {e}")
          # call your menu or cleanup here if needed
          # menue()

  if __name__ == "__main__":
      main()

  return sysargv1
  menue()


def binimg():

  try:

      def main():
  
                 sysargv2  = input("Enter file name of input color image fits  -->")
                 sysargv3  = input("Enter file name of binned color image  -->")
                 sysargv4  = input("Enter binning_factor(25) -->")
                 bin_size  = int(sysargv4)
                 with fits.open(sysargv2) as hdul:
                     data = hdul[0].data
                   
                     # Check if the image is color (3D array)
                     if data.ndim == 3:
                       binned_data = np.zeros((data.shape[0], data.shape[1] // bin_size, data.shape[2] // bin_size), dtype=data.dtype)
                       for i in range(data.shape[0]):  # Iterate through color channels
                         for y in range(0, data.shape[1], bin_size):
                           for x in range(0, data.shape[2], bin_size):
                             binned_data[i, y // bin_size, x // bin_size] = np.mean(data[i, y:y + bin_size, x:x + bin_size])
                     else:
                     # Handle grayscale or other formats if needed
                       raise ValueError("Only color images are supported. The image must be a 3D array.")
           
                     # Create a new HDU with the binned data
                     hdu = fits.PrimaryHDU(binned_data)
                     # Copy header from original
                     hdu.header = hdul[0].header.copy()
                     # Save the binned image to a new FITS file
                     hdu.writeto(sysargv3, overwrite=True) 

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()


def autostr():

  try:

      def smh_stretch(data, lower_percent=0.5, upper_percent=99.5):
        """
        Perform a shadows/midtones/highlights (SMH) style histogram stretch.
    
        This function computes:
          - low: the shadow threshold (at the lower_percent percentile),
          - high: the highlight threshold (at the upper_percent percentile),
          - med: the midtone (50th percentile),
        and computes gamma such that the normalized midtone maps to 0.5, i.e.,
             ( (med - low)/(high - low) )^gamma = 0.5.
    
        Then, the data is normalized and the gamma correction applied.
    
        Parameters:
          data          : NumPy array containing the image pixel values.
          lower_percent : Lower percentile for shadows (default is 0.5).
          upper_percent : Upper percentile for highlights (default is 99.5).
    
        Returns:
          stretched : The processed image (values in [0, 1]).
          low       : The computed shadow threshold.
          med       : The computed midtone value.
          high      : The computed highlight threshold.
          gamma     : The gamma value applied.
        """
        # Compute shadow and highlight thresholds from percentiles.
        low = np.percentile(data, lower_percent)
        high = np.percentile(data, upper_percent, method='higher')
    
        # Compute the median (midtone) of the image.
        med = np.percentile(data, 50)
        # Ensure the midtone lies within [low, high]
        med = np.clip(med, low, high)
    
        # Avoid division by zero in case high equals low.
        if high == low:
          return np.zeros_like(data), low, med, high, 1.0
    
        # Normalize the midtone between 0 and 1.
        m = (med - low) / (high - low)
        # Avoid a zero or extremely small value for m.
        if m <= 0:
            m = 0.001
    
        # Determine gamma such that m^gamma = 0.5.
        gamma = np.log(0.5) / np.log(m)
    
        # Normalize data to the range [0, 1] based on computed thresholds.
        normalized = (data - low) / (high - low)
        normalized = np.clip(normalized, 0, 1)
    
        # Apply the gamma correction.
        stretched = normalized ** gamma
    
        return stretched, low, med, high, gamma

      def main():
                        
                 sysargv3  = input("Enter file name of image to auto_str  -->")
                 sysargv4  = input("Enter file name of output image -->")
                 lower_percent  = 0.5
                 upper_percent  = 99.999
           
                 with fits.open(sysargv3) as hdul:
                     hdu = hdul[0]
                     data = hdu.data.astype(np.float64)
                     header = hdu.header
           
                 # Apply the SMH stretch.
                 stretched, low, med, high, gamma = smh_stretch(data, lower_percent, upper_percent)
               
                 # Print out the computed parameters.
                 print(f"Shadow threshold (lower {lower_percent}th percentile): {low}")
                 print(f"Midtone (50th percentile): {med}")
                 print(f"Highlight threshold (upper {upper_percent}th percentile): {high}")
                 print(f"Applied gamma: {gamma}\n")
               
                 # Update the header to record stretch information.
                 header['STRETCH'] = ('SMH', 'Shadows/Midtones/Highlights histogram stretch')
                 header['LOWPCT'] = (lower_percent, 'Lower percentile for shadow threshold')
                 header['HIGPCT'] = (upper_percent, 'Upper percentile for highlight threshold')
                 header['SHADOW'] = (low, 'Shadow threshold value')
                 header['MIDTONE'] = (med, 'Midtone (median) value')
                 header['HIGHLT'] = (high, 'Highlight threshold value')
                 header['GAMMA'] = (gamma, 'Gamma value used')
               
                 # Save the stretched image as a new FITS file.
                 hdu_new = fits.PrimaryHDU(data=stretched, header=header)
                 hdu_new.writeto(sysargv4, overwrite=True)
               
                 print(f"Stretched FITS image has been saved to: {sysargv4}")

           
      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()


def LocAdapt():

  try:
      
      def main():
          
                 sysargv2  = input("Enter the Color Image for LA -->")
                 sysargv6  = input("Enter neighborhood_size(15) -->")
                 sysargv6int  = int(sysargv6)
           
                 # Function to read FITS file and return data
                 def read_fits(file):
                     hdul = fits.open(file)
                     header = hdul[0].header
                     data = hdul[0].data
                     hdul.close()
                     return data, header
           
                 # Read the FITS files
                 file1 = sysargv2
                 # Read the image data from the FITS file
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(float)
           
                 # Split the color image into its individual channels
                 #b, g, r = cv2.split(image_data)
                 b, g, r = image_data[2, :, :], image_data[1, :, :], image_data[0, :, :] 
           
           
                 # Save each channel as a separate file
                 fits.writeto(f'channel_0_64bit.fits', b.astype(float), header, overwrite=True)
                 fits.writeto(f'channel_1_64bit.fits', g.astype(float), header, overwrite=True)
                 fits.writeto(f'channel_2_64bit.fits', r.astype(float), header, overwrite=True)
           
                 sysargv2  = "channel_0_64bit.fits"
                 sysargv2g  = "channel_1_64bit.fits"
                 sysargv2r  = "channel_2_64bit.fits"
           
                 sysargv4a  = input("Enter the Contrast as (50) with no decimal  -->")
                 sysargv4 = int(sysargv4a)
                 sysargv4b  = input("Enter the feather_distance as (5) with no decimal  -->")
                 sysargv4c = int(sysargv4b)
           
                 def compute_optimum_contrast_percentage(image, target_std):
                   """
                   Compute an optimum contrast percentage based on the image's current standard deviation.
               
                   Parameters:
                   image (numpy.ndarray): The input image as a 2D NumPy array.
                   target_std (float): A chosen target standard deviation.
                                      The optimum contrast factor is computed as target_std / current_std.
               
                   Returns:
                       float: The computed optimum contrast percentage.
                       For example, a return value of 150 means 150% (i.e., a factor of 1.5).
                   """
                   current_std = np.std(image)
                   if current_std == 0:
                   # For a flat image, no change is applied.
                     return 100.0
                   contrast_factor = target_std / current_std
                   contrast_percentage = contrast_factor * 100.0
                   return contrast_percentage
           
                 def compute_optimum_feather_distance(image, neighborhood_size, factor=1.0):
                   """
                   Compute an optimum feather distance based on the local standard deviation of the image.
                
                   This function computes the local mean and local mean of the squared image within a window,
                   then derives the local standard deviation. The median of these local standard deviations
                   is taken as the baseline optimum feather distance, which can be adjusted by a scaling factor.
                       Parameters:
                       image (numpy.ndarray): The input image as a 2D NumPy array.
                       neighborhood_size (int): The window size used for calculating local statistics.
                       factor (float): A multiplicative factor to adjust the optimum feather distance (default is 1.0).
               
                   Returns:
                       float: The computed optimum feather distance.
                   """
                   # Adjust the window size by reducing it by 1 (ensuring an effective kernel size of at least 1)
                   adjusted_size = max(1, neighborhood_size - 1)
                   kernel = np.ones((adjusted_size, adjusted_size), dtype=np.float32) / (adjusted_size * adjusted_size)
               
                   # Compute the local mean using a convolution
                   local_mean = convolve(image, kernel, mode='reflect')
               
                   # Compute the local mean of the squared image
                   local_mean_sq = convolve(image**2, kernel, mode='reflect')
               
                   # Calculate local standard deviation: sqrt(local_mean_sq - local_mean**2)
                   local_std = np.sqrt(np.abs(local_mean_sq - local_mean**2))
               
                   # Use the median of the local standard deviations as the baseline, then scale.
                   optimum_feather = factor * np.median(local_std)
                   return optimum_feather
           
               #--------------------------------------------------------------------------------------
                 def contrast_filter(image, neighborhood_size, contrast_factor, feather_distance):
                   """
                   Apply a local adaptive contrast filter with feathering.
               
                   This function computes a local mean (using an adjusted neighborhood size) and then
                   enhances the contrast by scaling the deviation from the local mean. Feathering is implemented 
                   by blending the enhanced image with the original image in areas where the local standard deviation 
                   is below a specified feather_distance threshold.
                     
                   Parameters:
                       image (ndarray): 2D array of the input image.
                       neighborhood_size (int): Size of the window for computing local statistics.
                       contrast_factor (float): Factor to scale the local contrast.
                       feather_distance (float): Threshold (in intensity units) that defines the feathering effect.
                                             Regions with local standard deviation lower than this value are blended
                                             with the original image.
               
                   Returns:
                         final_image (ndarray): The contrast-enhanced image.
                   """
                   # Adjust the kernel size (reducing the provided neighborhood size by 1, but never below 1)
                   adjusted_size = max(1, neighborhood_size - 1)
                   kernel = np.ones((adjusted_size, adjusted_size), dtype=float)
                   kernel /= kernel.size
           
                   # Compute the local mean using convolution
                   local_mean = convolve(image, kernel, mode='reflect')
               
                   # Compute the enhanced image using the standard contrast adjustment formula:
                   #   enhanced = (image - local_mean) * contrast_factor + local_mean
                   enhanced_image = (image - local_mean) * contrast_factor + local_mean
           
                   # For feathering, compute the local standard deviation:
                   # First, get the local mean of the squared image.
                   squared_image = np.square(image)
                   local_mean_squared = convolve(squared_image, kernel, mode='reflect')
                   # Then compute the standard deviation (making sure to safeguard against small negative values)
                   local_std = np.sqrt(np.abs(local_mean_squared - np.square(local_mean)))
               
                   # Create a feathering weight: when local_std is below feather_distance the weight is <1.
                   # This weight is 1 if local_std >= feather_distance.
                   weight = np.clip(local_std / feather_distance, 0, 1)
               
                   # Blend the original and enhanced images using the weight map:
                   #   In low-contrast regions (low local_std) the enhanced effect is partially reduced.
                   final_image = weight * enhanced_image + (1 - weight) * image
               
                   return final_image
           
                 # -------------------------
                 # Hard-coded Parameters
                 # -------------------------
                 input_fits  = 'channel_0_64bit.fits'   # Name/path for the input FITS file.
                 output_fits = 'channel_0_64bitLA.fits'  # Name/path for the output FITS file.
                 neighborhood_size = sysargv6int      # Size of the region used to compute local statistics.
                 target_std = int(sysargv4) 
                 # Factor by which to scale the local deviations.
                 feather_distance = int(sysargv4c)       # Feather threshold: lower values produce more feathering in smooth areas.
                 feather_factor = feather_distance
           
                 # -------------------------
                 # Read the FITS File
                 # -------------------------
                 hdulist = fits.open(input_fits)
                 # Convert image data to float32 for accurate arithmetic and convolution
                 image_data = hdulist[0].data.astype(float)
                 header = hdulist[0].header
           
                 optimum_contrast_percentage = compute_optimum_contrast_percentage(image_data, target_std)
                 contrast_factor = optimum_contrast_percentage * 100
                 optimum_feather_distance = compute_optimum_feather_distance(image_data, neighborhood_size, feather_factor)
                 feather_distance = optimum_feather_distance / 100
           
                 # -------------------------
                 # Apply the Contrast Filter with Feathering
                 # -------------------------
                 enhanced_data = contrast_filter(image_data, neighborhood_size, contrast_factor, feather_distance)
           
                 # -------------------------
                 # Write the Enhanced Image to a New FITS File
                 # -------------------------
                 hdu = fits.PrimaryHDU(data=enhanced_data, header=header)
                 hdu.writeto(output_fits, overwrite=True)
                 hdulist.close()
           
                 print(f"Enhanced FITS file with feathering saved as {output_fits}")
           
                 # -------------------------
                 # Hard-coded Parameters
                 # -------------------------
                 input_fits  = 'channel_1_64bit.fits'   # Name/path for the input FITS file.
                 output_fits = 'channel_1_64bitLA.fits'  # Name/path for the output FITS file.
                 neighborhood_size = sysargv6int       # Size of the region used to compute local statistics.
                 target_std = int(sysargv4) 
                 # Factor by which to scale the local deviations.
                 feather_distance = int(sysargv4c)       # Feather threshold: lower values produce more feathering in smooth 
                 feather_factor = feather_distance
           
                 # -------------------------
                 # Read the FITS File
                 # -------------------------
                 hdulist = fits.open(input_fits)
                 # Convert image data to float64 for accurate arithmetic and convolution
                 image_data = hdulist[0].data.astype(float)
                 header = hdulist[0].header
           
                 optimum_contrast_percentage = compute_optimum_contrast_percentage(image_data, target_std)
                 contrast_factor = optimum_contrast_percentage * 100
                 optimum_feather_distance = compute_optimum_feather_distance(image_data, neighborhood_size, feather_factor)
                 feather_distance = optimum_feather_distance / 100
           
                 # -------------------------
                 # Apply the Contrast Filter with Feathering
                 # -------------------------
                 enhanced_data = contrast_filter(image_data, neighborhood_size, contrast_factor, feather_distance)
           
                 # -------------------------
                 # Write the Enhanced Image to a New FITS File
                 # -------------------------
                 hdu = fits.PrimaryHDU(data=enhanced_data, header=header)
                 hdu.writeto(output_fits, overwrite=True)
                 hdulist.close()
           
                 print(f"Enhanced FITS file with feathering saved as {output_fits}")
           
                 # -------------------------
                 # Hard-coded Parameters
                 # -------------------------
                 input_fits  = 'channel_2_64bit.fits'   # Name/path for the input FITS file.
                 output_fits = 'channel_2_64bitLA.fits'  # Name/path for the output FITS file.
                 neighborhood_size = sysargv6int       # Size of the region used to compute local statistics.
                 target_std = int(sysargv4) 
                 # Factor by which to scale the local deviations.
                 feather_distance = int(sysargv4c)       # Feather threshold: lower values produce more feathering in smooth 
                 feather_factor = feather_distance
           
                 # -------------------------
                 # Read the FITS File
                 # -------------------------
                 hdulist = fits.open(input_fits)
                 # Convert image data to float64 for accurate arithmetic and convolution
                 image_data = hdulist[0].data.astype(float)
                 header = hdulist[0].header
           
                 optimum_contrast_percentage = compute_optimum_contrast_percentage(image_data, target_std)
                 contrast_factor = optimum_contrast_percentage * 100
                 optimum_feather_distance = compute_optimum_feather_distance(image_data, neighborhood_size, feather_factor)
                 feather_distance = optimum_feather_distance / 100
           
                 # -------------------------
                 # Apply the Contrast Filter with Feathering
                 # -------------------------
                 enhanced_data = contrast_filter(image_data, neighborhood_size, contrast_factor, feather_distance)
           
                 # -------------------------
                 # Write the Enhanced Image to a New FITS File
                 # -------------------------
                 hdu = fits.PrimaryHDU(data=enhanced_data, header=header)
                 hdu.writeto(output_fits, overwrite=True)
                 hdulist.close()
           
                 print(f"Enhanced FITS file with feathering saved as {output_fits}")
           
               #--------------------------------------------------------------------------------------
           
                 sysargv1  = "channel_0_64bitLA.fits"
                 sysargv2  = "channel_1_64bitLA.fits"
                 sysargv3  = "channel_2_64bitLA.fits"
                 sysargv4  = "channel_RGB_64bitLA.fits"
           
                 old_data, old_header = read_fits(file1)
                 old_data = old_data.astype(float)
              
                 # Function to read FITS file and return data
                 def read_fits(file):
                   with fits.open(file, mode='update') as hdul:#
                     data = hdul[0].data
                     hdul.close()
                   return data
           
                 # Read the FITS files
                 file1 = sysargv1
                 file2 = sysargv2
                 file3 = sysargv3
           
                 # Read the image data from the FITS file
                 blue = read_fits(file1)
                 green = read_fits(file2)
                 red = read_fits(file3)
           
                 blue = blue.astype(float)
                 green = green.astype(float)
                 red = red.astype(float)
           
                 # Check dimensions
                 print("Data1 shape:", blue.shape)
                 print("Data2 shape:", green.shape)
                 print("Data3 shape:", red.shape)
           
                 #newRGBImage = cv2.merge((red,green,blue))
                 RGB_Image1 = np.stack((red,green,blue))
           
                 # Remove the extra dimension
                 RGB_Image = np.squeeze(RGB_Image1)
           
                 # Create a FITS header with NAXIS = 3
                 header = old_header
                 header['NAXIS'] = 3
                 header['NAXIS1'] = RGB_Image.shape[0]
                 header['NAXIS2'] = RGB_Image.shape[1]
                 header['NAXIS3'] = RGB_Image.shape[2]
           
                 # Ensure the data type is correct 
                 newRGB_Image = RGB_Image.astype(float)
           
                 print("newRGB_Image shape:", newRGB_Image.shape)
           
                 sysargv4 = "channel_RGB_64bitLA.fits"
                 fits.writeto( sysargv4, newRGB_Image, overwrite=True)
                 # Save the RGB image as a new FITS file with the correct header
                 hdu = fits.PrimaryHDU(data=newRGB_Image, header=header)
                 hdu.writeto(sysargv4, overwrite=True)
           
                 # Read and verify the saved FITS file
                 with fits.open(sysargv4) as hdul:
                   data = hdul[0].data
           
                 # Verify the saved RGB image
                 print("Verified image shape:", data.shape)
           
                 currentDirectory = os.path.abspath(os.getcwd())
           
                 # Define the file to delete
                 file_to_delete = "channel_0_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_1_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_2_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_0_64bitLA.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_1_64bitLA.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                       print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_2_64bitLA.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")


      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def WcsOvrlay():

  try:

      class FitsWcsPlotter(QWidget):
          def __init__(self):
              super().__init__()
              self.setWindowTitle("FITS WCS RGB Plotter")
              self._build_ui()

          def _build_ui(self):
              layout = QVBoxLayout()

              # Row 1: FITS input
              row1 = QHBoxLayout()
              lbl1 = QLabel("FITS WCS File:")
              self.fits_edit = QLineEdit()
              btn1 = QPushButton("Browse…")
              btn1.clicked.connect(self._browse_fits)
              row1.addWidget(lbl1)
              row1.addWidget(self.fits_edit, stretch=1)
              row1.addWidget(btn1)
              layout.addLayout(row1)

              # Row 2: Plot title
              row2 = QHBoxLayout()
              lbl2 = QLabel("Plot Title:")
              self.title_edit = QLineEdit()
              row2.addWidget(lbl2)
              row2.addWidget(self.title_edit, stretch=1)
              layout.addLayout(row2)

              # Row 3: Output filename
              row3 = QHBoxLayout()
              lbl3 = QLabel("Save Plot As:")
              self.out_edit = QLineEdit()
              btn3 = QPushButton("Browse…")
              btn3.clicked.connect(self._browse_output)
              row3.addWidget(lbl3)
              row3.addWidget(self.out_edit, stretch=1)
              row3.addWidget(btn3)
              layout.addLayout(row3)

              # Plot button
              plot_btn = QPushButton("Plot & Save")
              plot_btn.clicked.connect(self._plot_and_save)
              plot_btn.setFixedHeight(36)
              layout.addWidget(plot_btn, alignment=Qt.AlignmentFlag.AlignCenter)

              self.setLayout(layout)
              self.resize(600, 180)

          def _browse_fits(self):
              path, _ = QFileDialog.getOpenFileName(
                  self, "Select FITS WCS File", "", "FITS Files (*.fits *.fit)"
              )
              if path:
                  self.fits_edit.setText(path)

          def _browse_output(self):
              path, _ = QFileDialog.getSaveFileName(
                  self, "Save Plot As…", "", "PNG Image (*.png);;JPEG Image (*.jpg)"
              )
              if path:
                  self.out_edit.setText(path)

          def _plot_and_save(self):
              try:
                  fits_path = self.fits_edit.text().strip()
                  title     = self.title_edit.text().strip()
                  out_path  = self.out_edit.text().strip()

                  if not fits_path:
                      raise ValueError("Please select a FITS WCS file.")
                  if not title:
                      raise ValueError("Please enter a plot title.")
                  if not out_path:
                      raise ValueError("Please choose an output filename.")

                  # Load FITS and WCS
                  hdul = fits.open(fits_path)
                  data = hdul[0].data
                  wcs  = WCS(hdul[0].header, naxis=2)

                  # Extract RGB channels
                  red   = data[0] / np.max(data[0])
                  green = data[1] / np.max(data[1])
                  blue  = data[2] / np.max(data[2])
                  rgb   = np.stack((red, green, blue), axis=-1)

                  # Plot with world coordinates
                  plt.figure(figsize=(8, 8))
                  ax = plt.subplot(projection=wcs)
                  ax.imshow(rgb, origin='lower')
                  ax.set_xlabel("Right Ascension")
                  ax.set_ylabel("Declination")
                  ax.coords.grid(True, color="white", ls="dotted")
                  plt.title(title)

                  # Save then show
                  plt.savefig(out_path, dpi=150, bbox_inches="tight")
                  plt.show()

                  QMessageBox.information(self, "Success", f"Plot saved to:\n{out_path}")

              except Exception as e:
                  QMessageBox.critical(self, "Error", str(e))

      if __name__ == "__main__":
          app = QApplication(sys.argv)
          window = FitsWcsPlotter()
          window.show()
          sys.exit(app.exec())

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

def Stacking():
                  
               def AlignImgsByDir():
                   class AlignImagesForm(QWidget):
                       def __init__(self):
                           super().__init__()
                           self.setWindowTitle("Batch Align Images (glob mode)")
                           self.resize(500, 150)

                           # Input directory chooser
                           self.in_dir_le = QLineEdit()
                           btn_in_dir = QPushButton("Browse Input Dir…")
                           btn_in_dir.clicked.connect(self._browse_input_dir)
                           h1 = QHBoxLayout()
                           h1.addWidget(QLabel("Input folder:"))
                           h1.addWidget(self.in_dir_le)
                           h1.addWidget(btn_in_dir)

                           # Output directory chooser
                           self.out_dir_le = QLineEdit()
                           btn_out_dir = QPushButton("Browse Output Dir…")
                           btn_out_dir.clicked.connect(self._browse_output_dir)
                           h2 = QHBoxLayout()
                           h2.addWidget(QLabel("Output folder:"))
                           h2.addWidget(self.out_dir_le)
                           h2.addWidget(btn_out_dir)

                           # Align button
                           self.align_button = QPushButton("Align All FITS")
                           self.align_button.clicked.connect(self._on_align)

                           layout = QVBoxLayout(self)
                           layout.addLayout(h1)
                           layout.addLayout(h2)
                           layout.addWidget(self.align_button, alignment=Qt.AlignmentFlag.AlignRight)

                       def _browse_input_dir(self):
                           d = QFileDialog.getExistingDirectory(self, "Select input folder with FITS")
                           if d:
                               self.in_dir_le.setText(d)

                       def _browse_output_dir(self):
                           d = QFileDialog.getExistingDirectory(self, "Select output folder")
                           if d:
                               self.out_dir_le.setText(d)

                       def _on_align(self):
                           in_dir  = self.in_dir_le.text().strip()
                           out_dir = self.out_dir_le.text().strip()

                           if not in_dir or not out_dir:
                               QMessageBox.warning(self, "Missing", "Please select both folders.")
                               return

                           # ----- glob all .fits in input dir -----
                           pattern = os.path.join(in_dir, "*.fit*")
                           inputs = sorted(glob.glob(pattern))
                           if not inputs:
                               QMessageBox.critical(self, "No FIT*", f"No .fit* files found in {in_dir}")
                               return

                           # ----- auto-generate outputs in out_dir -----
                           outputs = [
                               os.path.join(out_dir, "aligned_" + os.path.basename(fn))
                               for fn in inputs
                           ]
                           try:
                               # load all data+hdr
                               dw = []
                               for fn in inputs:
                                   with fits.open(fn) as hd:
                                       dw.append((hd[0].data.astype(np.float64), hd[0].header))

                               # compute common WCS & shape
                               wcs_out, shape_out = find_optimal_celestial_wcs(dw)

                               # reproject & save each
                               for (data, hdr), outfn in zip(dw, outputs):
                                   arr, _ = reproject_interp((data, hdr), wcs_out, shape_out=shape_out)
                                   hdu = fits.PrimaryHDU(arr, header=wcs_out.to_header())
                                   hdu.writeto(outfn, overwrite=True)
               
                               QMessageBox.information(
                                   self, "Done", f"Aligned {len(inputs)} images to:\n{out_dir}"
                               )
                           except Exception as e:
                               QMessageBox.critical(self, "Error", str(e))

                   app = QApplication(sys.argv)
                   w = AlignImagesForm()
                   w.show()
                   app.exec()

               if __name__ == "__main__":
                   try:
                       AlignImgsByDir()
                   except Exception as e:
                       print("Fatal error in AlignImgs():", e)
                       # fall back to main menu, etc.
                       return sysargv1
                       menue()

def combinelrgb():

  try:
               
      class FitsCombiner(QWidget):
          def __init__(self):
              super().__init__()
              self.setWindowTitle("FITS LRGB Combine")
              self._build_ui()

          def _build_ui(self):
              layout = QVBoxLayout()

              # Create rows for the 4 inputs + 1 output
              self.paths = {}
              for label_text, key, mode in [
                  ("Luminance FITS:", "lum", "open"),
                  ("Blue FITS:     ", "blue", "open"),
                  ("Green FITS:    ", "green", "open"),
                  ("Red FITS:      ", "red", "open"),
                  ("Save As:       ", "output", "save"),
              ]:
                  row = QHBoxLayout()
                  lbl = QLabel(label_text)
                  edit = QLineEdit()
                  btn = QPushButton("Browse…")
                  btn.clicked.connect(lambda _, k=key, m=mode: self._browse(k, m))
                  row.addWidget(lbl)
                  row.addWidget(edit, stretch=1)
                  row.addWidget(btn)
                  layout.addLayout(row)
                  self.paths[key] = edit

              # Combine button
              combine_btn = QPushButton("Combine")
              combine_btn.clicked.connect(self.combine_fits)
              combine_btn.setFixedHeight(40)
              layout.addWidget(combine_btn, alignment=Qt.AlignmentFlag.AlignCenter)

              self.setLayout(layout)
              self.resize(600, 250)

          def _browse(self, key, mode):
              """Open file dialog to select input or output path."""
              if mode == "open":
                  path, _ = QFileDialog.getOpenFileName(
                      self, f"Select {key} file", "", "FITS Files (*.fits *.fit)"
                  )
              else:
                  path, _ = QFileDialog.getSaveFileName(
                      self, f"Save combined as…", "", "FITS Files (*.fits *.fit)"
                  )
              if path:
                  self.paths[key].setText(path)

          def combine_fits(self):
              try:
                  # Read file paths from UI
                  lum_file    = self.paths["lum"].text().strip()
                  blue_file   = self.paths["blue"].text().strip()
                  green_file  = self.paths["green"].text().strip()
                  red_file    = self.paths["red"].text().strip()
                  output_file = self.paths["output"].text().strip()

                  # Validate
                  for key, p in [("Luminance", lum_file), ("Blue", blue_file),
                                 ("Green", green_file), ("Red", red_file),
                                 ("Output", output_file)]:
                      if not p:
                          raise ValueError(f"{key} path is empty.")

                  # Helper to read FITS data
                  def read_fits(path):
                      with fits.open(path) as hdul:
                          return hdul[0].data.astype(np.float64), hdul[0].header

                  # Load images
                  lum, header = read_fits(lum_file)
                  blue, _     = read_fits(blue_file)
                  green, _    = read_fits(green_file)
                  red, _      = read_fits(red_file)

                  # Normalize function
                  def normalize(arr):
                      return (arr - arr.min()) / (arr.max() - arr.min())

                  lum   = normalize(lum)
                  blue  = normalize(blue)
                  green = normalize(green)
                  red   = normalize(red)

                  # Build RGB cube (z=3, y, x)
                  rgb_cube = np.zeros((3, lum.shape[0], lum.shape[1]), dtype=np.float64)
                  rgb_cube[0] = red   * lum  # Red plane
                  rgb_cube[1] = green * lum  # Green plane
                  rgb_cube[2] = blue  * lum  # Blue plane

                  # Update header for 3D cube
                  header['NAXIS']  = 3
                  header['NAXIS1'] = lum.shape[1]
                  header['NAXIS2'] = lum.shape[0]
                  header['NAXIS3'] = 3
                  header['FILTER'] = 'L+R+G+B'

                  # Write out
                  hdu = fits.PrimaryHDU(data=rgb_cube, header=header)
                  hdu.writeto(output_file, overwrite=True)

                  QMessageBox.information(self, "Done", f"Combined saved to:\n{output_file}")

              except Exception as e:
                  QMessageBox.critical(self, "Error", str(e))


      if __name__ == "__main__":
          app = QApplication(sys.argv)
          win = FitsCombiner()
          win.show()
          sys.exit(app.exec())
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def MxdlAstap():

  try:

      def main():

                 sysargv0  = input("Enter MaxDl fits file for Astap  -->")
           
                   # Set mpmath precision to 50 decimal places
                 mp.dps = 50
           
                 # Open the FITS file in update mode so we can modify its header.
                 filename = sysargv0    # Replace with your FITS file name
                 hdul = fits.open(filename, mode="update")
                 hdr = hdul[0].header
           
                 # --- Step 1. Read the astrometry.net keywords ---
                 # These keywords come from the astrometry.net solution in the header.
                 objctra = hdr.get("OBJCTRA")      # e.g., '12 30 50.3'
                 objctdec = hdr.get("OBJCTDEC")     # e.g., '+12 23 13'
                 cdelt1  = hdr.get("CDELT1")        # e.g., 1.466251413027E-04
                 cdelt2  = hdr.get("CDELT2")        # e.g., 1.466251413027E-04
           
                 if objctra is None or objctdec is None or cdelt1 is None or cdelt2 is None:
                     raise ValueError("Required keywords OBJCTRA, OBJCTDEC, or CDELT* are missing.")
           
                 # --- Step 2. Convert center coordinates to degrees ---
                 # OBJCTRA is in "HH MM SS.s" [hourangle] and OBJCTDEC is in "DD MM SS".
                 ra_deg  = Angle(objctra, unit="hourangle").degree
                 dec_deg = Angle(objctdec, unit="deg").degree
           
                 # --- Step 3. Determine the reference pixel (typically the image center) ---
                 naxis1 = hdr.get("NAXIS1")
                 naxis2 = hdr.get("NAXIS2")
                 if naxis1 is None or naxis2 is None:
                     raise ValueError("Image dimensions (NAXIS1, NAXIS2) are missing.")
           
                 # Following the common convention, the reference pixel is placed at the center.
                 crpix1 = (naxis1 + 1) / 2.0
                 crpix2 = (naxis2 + 1) / 2.0
           
                 # --- Step 4. Calculate the new WCS parameters with high precision ---
                 # Convert cdelt1 to a high-precision mpmath float:
                 cdelt1_mp = mp.mpf(cdelt1)
                 cdelt2_mp = mp.mpf(cdelt2)
           
                 # Mimic a solved CD1_1 value (from an astrometric solution)
                 solved_cd1_1 = mp.mpf(-3.542246e-4)
                 # Compute the factor by which the effective scale changes:
                 scale_factor = abs(solved_cd1_1) / cdelt1_mp  # e.g., ~2.416
                 # New effective pixel scale (in deg/pixel) – it should be nearly the absolute value of solved_cd1_1.
                 scale = scale_factor * cdelt1_mp
           
                 # Assume a (solved) image rotation, e.g., 179.7°
                 rotation_deg = mp.mpf(179.7)
                 # Convert degrees to radians using mpmath:
                 theta = rotation_deg * (mp.pi / 180)
           
                 # Calculate the CD matrix components using high-precision mpmath functions:
                 cd1_1 = -scale * mp.cos(theta)
                 cd1_2 =  scale * mp.sin(theta)
                 cd2_1 =  scale * mp.sin(theta)
                 cd2_2 =  scale * mp.cos(theta)
           
                 # --- Step 5. Write the computed WCS keywords into the FITS header ---
                 hdr["CRPIX1"]   = crpix1
                 hdr["CRPIX2"]   = crpix2
                 hdr["CRVAL1"]   = ra_deg    # Fitted RA center in degrees
                 hdr["CRVAL2"]   = dec_deg   # Fitted DEC center in degrees
                 hdr["CROTA1"]   = float(rotation_deg)  # Convert high-precision number to float
                 hdr["CROTA2"]   = float(rotation_deg)  # Often nearly the same value as CROTA1
                 hdr["CD1_1"]    = float(cd1_1)
                 hdr["CD1_2"]    = float(cd1_2)
                 hdr["CD2_1"]    = float(cd2_1)
                 hdr["CD2_2"]    = float(cd2_2)
                 hdr["PLTSOLVD"] = True   # Flag indicating that the plate solution has been applied
                 # Additional WCS keywords
                 hdr["CTYPE1"]  = "RA---TAN"   # first parameter RA, projection TANgential   
                 hdr["CTYPE2"]  = "DEC--TAN"   # second parameter DEC, projection TANgential   
                 hdr["CUNIT1"]  = "deg"        # Unit of coordinates                            
                 hdr["EQUINOX"] = 2000.0       # Equinox of coordinates                         
           
           
                 # Save (flush) the changes and close the file.
                 hdul.flush()
                 hdul.close()
           
                 print("WCS keywords calculated and written to the FITS header.")


      if __name__ == "__main__":
          main()           

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()


def CentRatio():

  try:

      def main():

                 sysargv0  = input("Enter star1_image1 x  -->")
                 sysargv1  = input("Enter star1_image1 y  -->")
                 sysargv2  = input("Enter star2_image1 x  -->")
                 sysargv3  = input("Enter star2_image1 y  -->")
                 sysargv4  = input("Enter star1_image2 x  -->")
                 sysargv5  = input("Enter star1_image2 y  -->")
                 sysargv6  = input("Enter star2_image2 x  -->")
                 sysargv7  = input("Enter star2_image2 y  -->")
           
           
                 def euclidean_distance(x1, y1, x2, y2):
                   """
                   Calculate the Euclidean distance between two points.
                   """
                   return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
           
                 # Image One coordinates (in pixels)
                 star1_image1 = (float(sysargv0), float(sysargv1))
                 star2_image1 = (float(sysargv2), float(sysargv3))
           
                 # Calculate distance1 for image one
                 distance1 = euclidean_distance(star1_image1[0], star1_image1[1], star2_image1[0], star2_image1[1])
           
                 # Image Two coordinates (in pixels)
                 star1_image2 = (float(sysargv4), float(sysargv5))
                 star2_image2 = (float(sysargv6), float(sysargv7))
           
                 # Calculate distance2 for image two
                 distance2 = euclidean_distance(star1_image2[0], star1_image2[1], star2_image2[0], star2_image2[1])
           
                 # Calculate the ratio of distances
                 ratio = distance1 / distance2
           
                 # Print out the results
                 print(f"Distance in Image One: {distance1:.5f} pixels")
                 print(f"Distance in Image Two: {distance2:.5f} pixels")
                 print(f"Ratio (distance1 / distance2): {ratio:.5f}")

      if __name__ == "__main__":
          main()           

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def HpMore():

  try:

      def main():

                 sysargv0  = input("Enter fits image  -->")
                 sysargv1  = input("Enter Hp output fits file  -->")
           
                 # Step 1: Read the color FITS file.
                 with fits.open(sysargv0) as hdul:
                     data = hdul[0].data
           
                 print("Original data shape:", data.shape)
           
                 # Ensure the data is in a 3D array representing three color channels.
                 # If the data shape is (ny, nx, 3), transpose it so that channels come first: (3, ny, nx).
                 if data.ndim == 3:
                     if data.shape[-1] == 3:  # shape is likely (ny, nx, 3)
                         data = np.transpose(data, (2, 0, 1))
                         print("Transposed data shape:", data.shape)
                     elif data.shape[0] == 3:
                         # Already in (3, ny, nx) order.
                         print("Data already in (channels, ny, nx) format.")
                     else:
                         raise ValueError("Unexpected color image shape. Expected channel count of 3.")
                 else:
                     raise ValueError("Input FITS does not appear to be a color (3 channel) image.")
           
                 # Step 2: Define the 5x5 high-pass (Laplacian-style) kernel.
                 # This kernel is designed to sum to zero: the negative weights in the outer parts subtract the local average,
                 # while the positive weights at and near the center emphasize high-frequency details.
                 hp_kernel_5x5 = np.array([
                     [-1, -1, -1, -1, -1],
                     [-1,  1,  2,  1, -1],
                     [-1,  2,  4,  2, -1],
                     [-1,  1,  2,  1, -1],
                     [-1, -1, -1, -1, -1]
                 ], dtype=float)
           
                 print("5x5 High-Pass Kernel:\n", hp_kernel_5x5)
           
                 # Step 3: Initialize an output array for the filtered data.
                 # We'll process each channel separately.
                 filtered_channels = np.empty_like(data)
           
                 # Loop over each channel and apply the range-restricted high-pass filter.
                 # For each channel, we compute the 5x5 convolution and then add its values back only where pixel values exceed a threshold.
                 for i in range(data.shape[0]):
                     channel = data[i]
               
                     # Apply the high-pass filter using 2D convolution.
                     # 'same' ensures the output matches the input dimensions,
                     # and 'symm' uses symmetric boundary handling.
                     highpass = convolve2d(channel, hp_kernel_5x5, mode='same', boundary='symm')
               
                     # Create a range restriction mask by choosing pixels above the 70th percentile.
                     threshold = np.percentile(channel, 70)
                     mask = channel > threshold
           
                     # Combine: add the high-pass detail to the original image only in pixels where the mask is True.
                     filtered_channel = channel.copy()
                     filtered_channel[mask] = channel[mask] + highpass[mask]
               
                     filtered_channels[i] = filtered_channel
           
                 # Step 4: Convert the filtered result back for visualization.
                 # Many plotting functions expect color images in (ny, nx, 3) order.
                 filtered_rgb = np.transpose(filtered_channels, (1, 2, 0))
                 original_rgb = np.transpose(data, (1, 2, 0))
           
           
                 # Step 6: Write the filtered image to a new FITS file.
                 # Here we save in the original channel-first ordering (3, ny, nx).
                 fits.writeto( sysargv1, filtered_channels, overwrite=True)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def CombBgrAlIm():
  try:

      def main():

                 input_dir = input("Enter directory path → ")   # e.g. "C:/data/fits"
                 file_pat  = input("Enter file pattern   → ").strip()   # e.g. "*.fit" or "*.fits"
           
                 # List the filenames of your reprojected FITS images.
                 # It is assumed that each image is on the same canvas and black (0) indicates no data.
           
           
                 # Collect all files with names starting with 'reproj' and ending with '.fits'
                 #filenames = sorted(glob.glob(sysargv0))
                 pattern   = os.path.join(input_dir, file_pat)          # e.g. "C:/data/fits/*.fit"
                 filenames = sorted(glob.glob(pattern))
           
                 print("Files found:", filenames)
           
           
                 # Open the first image to get the shape of the common canvas.
                 with fits.open(filenames[0]) as hdul:
                     shape = hdul[0].data.shape
           
                 # Initialize two arrays with the same shape:
                 #   composite: to accumulate the pixel values
                 #   weights: to count how many images contributed at each pixel
                 composite = np.zeros(shape, dtype=np.float32)
                 weights = np.zeros(shape, dtype=np.float32)
           
                 # Loop over each reprojected image
                 for fname in filenames:
                     with fits.open(fname) as hdul:
                         # Convert the image data to float32 for precision.
                         data = hdul[0].data.astype(np.float32)
               
                     # Define a mask for valid data; here we assume that a pixel value > 0 is a valid contribution.
                     # (If your images might have valid zero values, you might need to use a quality mask instead.)
                     mask = data > 0
               
                     # Only add the data where the image actually contributes.
                     composite[mask] += data[mask]
                     weights[mask] += 1
           
                 # Determine the maximum weight on the canvas—we assume this is the maximum number of overlapping images.
                 max_weight = np.max(weights)
                 print(f"Maximum contributions across the canvas: {max_weight}")
           
                 # Now compute the final composite image.
                 # For each pixel, first get the average (composite value divided by the weight) and then
                 # boost it to the level of maximum overlap.
                 # That is, for every pixel: final_pixel = (composite / weight) * sqrt(max_weight)
                 #
                 # This means that if a pixel is observed only once (weight==1) it will get multiplied by sqrt(max_weight),
                 # whereas a pixel observed max_weight times would be scaled by √(max_weight)/√(max_weight) = 1.
                 final = np.zeros_like(composite)
                 valid = weights > 0  # to avoid any division by zero issues
                 final[valid] = (composite[valid] / weights[valid]) * np.sqrt(max_weight)
           
                 # Save out the final image to a new FITS file.
                 hdu = fits.PrimaryHDU(final)
                 hdu.writeto('final_wgted_Al_Img.fits', overwrite=True)
           
                 print("Finished writing final_wgted_Al_Img.fits")

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

def pixelmath():

  try:
               
      class PixelMathWindow(QMainWindow):
          def __init__(self):
              super(PixelMathWindow, self).__init__()
              self.setWindowTitle("Pixel Math")
              self.initUI()
      
          def initUI(self):
              # Create a central widget and set a grid layout.
              centralWidget = QWidget()
              self.setCentralWidget(centralWidget)
              layout = QGridLayout(centralWidget)
      
              # Row 0: Drop-down for selecting the operation
              layout.addWidget(QLabel("Operation:"), 0, 0)
              self.operationComboBox = QComboBox()
              self.operationComboBox.addItems(["Add", "Subtract", "Multiply", "Divide", "Max", "Min"])
              layout.addWidget(self.operationComboBox, 0, 1, 1, 2)
      
              # Row 1: Input for the first image file, with Browse button.
              layout.addWidget(QLabel("First Image File:"), 1, 0)
              self.firstImageLineEdit = QLineEdit()
              layout.addWidget(self.firstImageLineEdit, 1, 1)
              self.firstBrowseButton = QPushButton("Browse")
              self.firstBrowseButton.clicked.connect(self.browseFirstImage)
              layout.addWidget(self.firstBrowseButton, 1, 2)
      
              # Row 2: Input for the second image file, with Browse button.
              layout.addWidget(QLabel("Second Image File:"), 2, 0)
              self.secondImageLineEdit = QLineEdit()
              layout.addWidget(self.secondImageLineEdit, 2, 1)
              self.secondBrowseButton = QPushButton("Browse")
              self.secondBrowseButton.clicked.connect(self.browseSecondImage)
              layout.addWidget(self.secondBrowseButton, 2, 2)
      
              # Row 3: Output file name input, with Browse button.
              layout.addWidget(QLabel("Output File:"), 3, 0)
              self.outputLineEdit = QLineEdit()
              layout.addWidget(self.outputLineEdit, 3, 1)
              self.outputBrowseButton = QPushButton("Browse")
              self.outputBrowseButton.clicked.connect(self.browseOutputFile)
              layout.addWidget(self.outputBrowseButton, 3, 2)
      
              # Row 4: Image1 brightness adjustment entry.
              layout.addWidget(QLabel("Image1 Brightness Adjustment:"), 4, 0)
              self.brightnessLineEdit = QLineEdit("0")
              layout.addWidget(self.brightnessLineEdit, 4, 1, 1, 2)
      
              # Row 5: Image1 contrast parameters.
              layout.addWidget(QLabel("Image1 Contrast Numerator:"), 5, 0)
              self.img1ContrastNumLineEdit = QLineEdit("1")
              layout.addWidget(self.img1ContrastNumLineEdit, 5, 1)
              layout.addWidget(QLabel("Denom:"), 5, 2)
              self.img1ContrastDenomLineEdit = QLineEdit("1")
              layout.addWidget(self.img1ContrastDenomLineEdit, 5, 3)
      
              # Row 6: Image2 contrast parameters.
              layout.addWidget(QLabel("Image2 Contrast Numerator:"), 6, 0)
              self.img2ContrastNumLineEdit = QLineEdit("1")
              layout.addWidget(self.img2ContrastNumLineEdit, 6, 1)
              layout.addWidget(QLabel("Denom:"), 6, 2)
              self.img2ContrastDenomLineEdit = QLineEdit("1")
              layout.addWidget(self.img2ContrastDenomLineEdit, 6, 3)
      
              # Row 7: Compute button.
              self.computeButton = QPushButton("Compute")
              self.computeButton.clicked.connect(self.computeOperation)
              layout.addWidget(self.computeButton, 7, 1, 1, 2)
      
              # Row 8: Status message label.
              self.statusLabel = QLabel("")
              layout.addWidget(self.statusLabel, 8, 0, 1, 4)
      
          def browseFirstImage(self):
              # Allow selection of .fits or .fit files.
              filename, _ = QFileDialog.getOpenFileName(self, "Open First Image", "", "FITS Files (*.fits *.fit)")
              if filename:
                  self.firstImageLineEdit.setText(filename)
      
          def browseSecondImage(self):
              filename, _ = QFileDialog.getOpenFileName(self, "Open Second Image", "", "FITS Files (*.fits *.fit)")
              if filename:
                  self.secondImageLineEdit.setText(filename)
      
          def browseOutputFile(self):
              filename, _ = QFileDialog.getSaveFileName(self, "Save Output Image", "", "FITS Files (*.fits *.fit)")
              if filename:
                  self.outputLineEdit.setText(filename)
      
          def validateInputs(self):
              """
              Validate the form entries and return a dictionary of converted values.
              In case of any validation error, update the statusLabel and return None.
              """
              first_file = self.firstImageLineEdit.text().strip()
              second_file = self.secondImageLineEdit.text().strip()
              output_file = self.outputLineEdit.text().strip()
              if not first_file or not second_file or not output_file:
                  self.statusLabel.setText("Error: Please provide file paths for first image, second image, and output.")
                  return None
      
              try:
                  brightness = float(self.brightnessLineEdit.text())
                  img1_num = float(self.img1ContrastNumLineEdit.text())
                  img1_denom = float(self.img1ContrastDenomLineEdit.text())
                  img2_num = float(self.img2ContrastNumLineEdit.text())
                  img2_denom = float(self.img2ContrastDenomLineEdit.text())
              except ValueError:
                  self.statusLabel.setText("Error: Brightness and contrast fields must be valid numbers.")
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
      
          def computeOperation(self):
              # Validate inputs first.
              inputs = self.validateInputs()
              if inputs is None:
                  return
      
              op = self.operationComboBox.currentText()
              try:
                  # Open FITS files (supports both .fits and .fit extensions).
                  hdul1 = fits.open(inputs["first_file"])
                  hdul2 = fits.open(inputs["second_file"])
                  image_data1 = hdul1[0].data.astype(np.float64)
                  image_data2 = hdul2[0].data.astype(np.float64)
      
                  # Ensure that the images/data have the same dimensions.
                  if image_data1.shape != image_data2.shape:
                      self.statusLabel.setText("Error: Input images do not have the same dimensions!")
                  if not (image_data1.ndim == image_data2.ndim ):
                      self.statusLabel.setText("Error: Image1 must be either Mono (2D) or Color RGB (3D with 3 channels).")
                      hdul1.close()
                      hdul2.close()
                      return
      
                  # Apply contrast scaling independently.
                  im1 = image_data1 * inputs["img1_scale"]
                  im2 = image_data2 * inputs["img2_scale"]
      
                  # Perform the selected arithmetic operation.
                  if op == "Add":
                      result_image = im1 + im2 + inputs["brightness"]
                  elif op == "Subtract":
                      result_image = im1 - im2 + inputs["brightness"]
                  elif op == "Multiply":
                      result_image = im1 * im2 + inputs["brightness"]
                  elif op == "Divide":
                      # Use numpy.divide with a condition to prevent division by zero.
                      result_image = np.divide(im1, im2, out=np.zeros_like(im1), where=im2 != 0) + inputs["brightness"]
                  elif op == "Max":
                      result_image = np.maximum(im1, im2) + inputs["brightness"]
                  elif op == "Min":
                      result_image = np.minimum(im1, im2) + inputs["brightness"]
                  else:
                      self.statusLabel.setText("Unknown operation selected!")
                      hdul1.close()
                      hdul2.close()
                      return
      
                  # Write the result to a new FITS file.
                  result_hdu = fits.PrimaryHDU(result_image)
                  result_hdulist = fits.HDUList([result_hdu])
                  result_hdulist.writeto(inputs["output_file"], overwrite=True)
                  self.statusLabel.setText("Operation completed; output saved successfully.")
      
                  hdul1.close()
                  hdul2.close()
      
              except Exception as e:
                  self.statusLabel.setText(f"An error occurred: {e}")
      
      if __name__ == '__main__':
          app = QApplication(sys.argv)
          window = PixelMathWindow()
          window.show()
          app.exec()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def Color():

  try:
        
      class ColorWindow(QMainWindow):
          def __init__(self):
              super(ColorWindow, self).__init__()
              self.setWindowTitle("Color")
              self.initUI()
      
          def initUI(self):
              # Main container and vertical layout
              centralWidget = QWidget()
              self.setCentralWidget(centralWidget)
              mainLayout = QVBoxLayout(centralWidget)
      
              # --- Operation selection ---
              opLayout = QHBoxLayout()
              opLabel = QLabel("Select Operation:")
              self.opCombo = QComboBox()
              self.opCombo.addItems(["Split Tricolor", "Combine Tricolor", "Create Luminance"])
              self.opCombo.currentIndexChanged.connect(self.operationChanged)
              opLayout.addWidget(opLabel)
              opLayout.addWidget(self.opCombo)
              opLayout.addStretch()
              mainLayout.addLayout(opLayout)
      
              # --- Stacked widget to show parameters for each operation ---
              self.stack = QStackedWidget()
              mainLayout.addWidget(self.stack)
      
              # --------- Page 0: Split Tricolor ----------
              self.splitWidget = QWidget()
              splitLayout = QGridLayout(self.splitWidget)
              # Input color image
              splitLayout.addWidget(QLabel("Input Color Image:"), 0, 0)
              self.splitInputLine = QLineEdit()
              splitLayout.addWidget(self.splitInputLine, 0, 1)
              self.splitInputBrowseBtn = QPushButton("Browse")
              self.splitInputBrowseBtn.clicked.connect(lambda: self.browseFile(self.splitInputLine))
              splitLayout.addWidget(self.splitInputBrowseBtn, 0, 2)
              # Mode: FITS or Other
              splitLayout.addWidget(QLabel("Mode:"), 1, 0)
              self.splitModeCombo = QComboBox()
              self.splitModeCombo.addItems(["FITS", "Other"])
              splitLayout.addWidget(self.splitModeCombo, 1, 1)
              # Optional output base name (if not provided then derived from input)
              splitLayout.addWidget(QLabel("Output Base Name:"), 2, 0)
              self.splitOutputBaseLine = QLineEdit()
              splitLayout.addWidget(self.splitOutputBaseLine, 2, 1)
              self.stack.addWidget(self.splitWidget)
      
              # --------- Page 1: Combine Tricolor ----------
              self.combineWidget = QWidget()
              combineLayout = QGridLayout(self.combineWidget)
              # Blue image
              combineLayout.addWidget(QLabel("Blue Image:"), 0, 0)
              self.combineBlueLine = QLineEdit()
              combineLayout.addWidget(self.combineBlueLine, 0, 1)
              self.combineBlueBrowseBtn = QPushButton("Browse")
              self.combineBlueBrowseBtn.clicked.connect(lambda: self.browseFile(self.combineBlueLine))
              combineLayout.addWidget(self.combineBlueBrowseBtn, 0, 2)
              # Green image
              combineLayout.addWidget(QLabel("Green Image:"), 1, 0)
              self.combineGreenLine = QLineEdit()
              combineLayout.addWidget(self.combineGreenLine, 1, 1)
              self.combineGreenBrowseBtn = QPushButton("Browse")
              self.combineGreenBrowseBtn.clicked.connect(lambda: self.browseFile(self.combineGreenLine))
              combineLayout.addWidget(self.combineGreenBrowseBtn, 1, 2)
              # Red image
              combineLayout.addWidget(QLabel("Red Image:"), 2, 0)
              self.combineRedLine = QLineEdit()
              combineLayout.addWidget(self.combineRedLine, 2, 1)
              self.combineRedBrowseBtn = QPushButton("Browse")
              self.combineRedBrowseBtn.clicked.connect(lambda: self.browseFile(self.combineRedLine))
              combineLayout.addWidget(self.combineRedBrowseBtn, 2, 2)
              # Output Combined Image
              combineLayout.addWidget(QLabel("Output Combined Image:"), 3, 0)
              self.combineOutputLine = QLineEdit()
              combineLayout.addWidget(self.combineOutputLine, 3, 1)
              self.combineOutputBrowseBtn = QPushButton("Browse")
              self.combineOutputBrowseBtn.clicked.connect(lambda: self.browseFile(self.combineOutputLine, save=True))
              combineLayout.addWidget(self.combineOutputBrowseBtn, 3, 2)
              # Mode
              combineLayout.addWidget(QLabel("Mode:"), 4, 0)
              self.combineModeCombo = QComboBox()
              self.combineModeCombo.addItems(["FITS", "Other"])
              combineLayout.addWidget(self.combineModeCombo, 4, 1)
              self.stack.addWidget(self.combineWidget)
      
              # --------- Page 2: Create Luminance ----------
              self.luminanceWidget = QWidget()
              lumLayout = QGridLayout(self.luminanceWidget)
              # Input color image
              lumLayout.addWidget(QLabel("Input Color Image:"), 0, 0)
              self.lumInputLine = QLineEdit()
              lumLayout.addWidget(self.lumInputLine, 0, 1)
              self.lumInputBrowseBtn = QPushButton("Browse")
              self.lumInputBrowseBtn.clicked.connect(lambda: self.browseFile(self.lumInputLine))
              lumLayout.addWidget(self.lumInputBrowseBtn, 0, 2)
              # Output Luminance Image
              lumLayout.addWidget(QLabel("Output Luminance Image:"), 1, 0)
              self.lumOutputLine = QLineEdit()
              lumLayout.addWidget(self.lumOutputLine, 1, 1)
              self.lumOutputBrowseBtn = QPushButton("Browse")
              self.lumOutputBrowseBtn.clicked.connect(lambda: self.browseFile(self.lumOutputLine, save=True))
              lumLayout.addWidget(self.lumOutputBrowseBtn, 1, 2)
              # Mode
              lumLayout.addWidget(QLabel("Mode:"), 2, 0)
              self.lumModeCombo = QComboBox()
              self.lumModeCombo.addItems(["FITS", "Other"])
              lumLayout.addWidget(self.lumModeCombo, 2, 1)
              self.stack.addWidget(self.luminanceWidget)
      
              # --- Run Button and Status Label ---
              self.runButton = QPushButton("Run")
              self.runButton.clicked.connect(self.runOperation)
              mainLayout.addWidget(self.runButton)
              self.statusLabel = QLabel("")
              mainLayout.addWidget(self.statusLabel)
      
          def operationChanged(self, index):
              # Change the shown parameter page
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
              # Get parameters for split tricolor
              inputFile = self.splitInputLine.text().strip()
              mode = self.splitModeCombo.currentText()
              outputBase = self.splitOutputBaseLine.text().strip()
              if not inputFile:
                  self.statusLabel.setText("Input file is required for Split Tricolor.")
                  return
              # Derive output base from input file if not provided
              if not outputBase:
                  outputBase = os.path.splitext(inputFile)[0]
              try:
                  if mode == "FITS":
                      hdul = fits.open(inputFile)
                      header = hdul[0].header
                      data = hdul[0].data.astype(np.float32)
                      hdul.close()
                      # Expect data shape to be (3, height, width)
                      if data.ndim != 3 or data.shape[0] != 3:
                          self.statusLabel.setText("FITS file does not appear to have 3 channels in first dimension.")
                          return
                      b = data[2, :, :]
                      g = data[1, :, :]
                      r = data[0, :, :]
                      fits.writeto(f"{outputBase}_channel_0_64bit.fits", b, header, overwrite=True)
                      fits.writeto(f"{outputBase}_channel_1_64bit.fits", g, header, overwrite=True)
                      fits.writeto(f"{outputBase}_channel_2_64bit.fits", r, header, overwrite=True)
                      self.statusLabel.setText("Split Tricolor (FITS) completed successfully.")
                  else:  # mode == "Other"
                      img = cv2.imread(inputFile, -1)
                      if img is None:
                          self.statusLabel.setText("Error reading image in Other mode.")
                          return
                      channels = cv2.split(img)
                      # OpenCV reads images in BGR order
                      cv2.imwrite(f"{outputBase}_Blue.png", channels[0])
                      cv2.imwrite(f"{outputBase}_Green.png", channels[1])
                      cv2.imwrite(f"{outputBase}_Red.png", channels[2])
                      self.statusLabel.setText("Split Tricolor (Other) completed successfully.")
              except Exception as e:
                  self.statusLabel.setText(f"Error in Split Tricolor: {e}")
      
          def runCombineTricolor(self):
              # Get parameters for combine tricolor
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
                      # Stack channels in order: red, green, blue (to define an RGB image)
                      RGB = np.stack((red, green, blue))
                      RGB = np.squeeze(RGB)
                      # Update header to indicate a 3D image
                      header['NAXIS'] = 3
                      header['NAXIS1'] = RGB.shape[1]
                      header['NAXIS2'] = RGB.shape[0]
                      header['NAXIS3'] = RGB.shape[2] if RGB.ndim == 3 else 1
                      hdu = fits.PrimaryHDU(data=RGB, header=header)
                      hdu.writeto(outputFile, overwrite=True)
                      self.statusLabel.setText("Combine Tricolor (FITS) completed successfully.")
                  else:  # mode == "Other"
                      blue = cv2.imread(blueFile, -1)
                      green = cv2.imread(greenFile, -1)
                      red = cv2.imread(redFile, -1)
                      if blue is None or green is None or red is None:
                          self.statusLabel.setText("Error reading one of the input images (Other).")
                          return
                      # Merge channels into an RGB image (note: OpenCV normally uses BGR order)
                      merged = cv2.merge((red, green, blue))
                      cv2.imwrite(outputFile, merged)
                      self.statusLabel.setText("Combine Tricolor (Other) completed successfully.")
              except Exception as e:
                  self.statusLabel.setText(f"Error in Combine Tricolor: {e}")
      
          def runCreateLuminance(self):
              # Get parameters for create luminance
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
                      # Assume data shape is (channels, height, width); convert to (height, width, channels)
                      img = np.transpose(data, (1, 2, 0))
                      R = img[:, :, 0].astype(np.float64)
                      G = img[:, :, 1].astype(np.float64)
                      B = img[:, :, 2].astype(np.float64)
                      luminance = 0.2989 * R + 0.5870 * G + 0.1140 * B
                      hdu = fits.PrimaryHDU(data=luminance)
                      hdu.writeto(outputFile, overwrite=True)
                      self.statusLabel.setText("Create Luminance (FITS) completed successfully.")
                  else:  # mode == "Other"
                      img = cv2.imread(inputFile, -1)
                      if img is None:
                          self.statusLabel.setText("Error reading input image (Other).")
                          return
                      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                      cv2.imwrite(outputFile, gray)
                      self.statusLabel.setText("Create Luminance (Other) completed successfully.")
              except Exception as e:
                  self.statusLabel.setText(f"Error in Create Luminance: {e}")
      
          def closeEvent(self, event):
              # Allow clean closing and return control to your main menu flow if needed.
              event.accept()
      
      # --- Run the application ---
      def Color():
          app = QApplication(sys.argv)
          window = ColorWindow()
          window.show()
          app.exec()
      
      if __name__ == '__main__':
          Color()
      
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def ImageFilters():
  try:

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
              self.stack.addWidget(self.createLrDeconvPage())    # Page 1
              self.stack.addWidget(self.createFFTPage())         # Page 2
              self.stack.addWidget(self.createErosionPage())     # Page 3
              self.stack.addWidget(self.createDilationPage())    # Page 4
              self.stack.addWidget(self.createGaussianPage())    # Page 5
              self.stack.addWidget(self.createHpMorePage())      # Page 6
              self.stack.addWidget(self.createLocAdaptPage())    # Page 7
      
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
      
          # ----------------------------
          # Helper: browse for FITS file
          def browseFile(self, lineEdit, save=False):
              if save:
                  fname, _ = QFileDialog.getSaveFileName(self, "Select File", "", "FITS Files (*.fits)")
              else:
                  fname, _ = QFileDialog.getOpenFileName(self, "Select File", "", "FITS Files (*.fits)")
              if fname:
                  lineEdit.setText(fname)
      
          # ----------------------------
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
      
          # ----------------------------
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
      
          # ----------------------------
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
      
          # ----------------------------
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
      
          # ----------------------------
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
      
          # ----------------------------
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
      
          # ----------------------------
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
      
          # ----------------------------
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
      
          # ----------------------------
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
      
          # ---------------------------------------------------------
          # Processing routines – each reads input FITS, processes, and writes output FITS.
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
              # Retrieve inputs from the GUI.
              inp = self.lrInput.text().strip()
              outp = self.lrOutput.text().strip()
              iters = int(self.lrIter.text().strip())
              psf_mode = self.lrPsfMode.currentText()
          
              # Open the FITS file and retrieve image & header.
              hdul = fits.open(inp)
              header = hdul[0].header
              image = hdul[0].data.astype(np.float64)
              hdul.close()
          
              # ------------------------------------------------------------------
              # Define the Richardson–Lucy deconvolution using convolve_fft.
              from astropy.convolution import convolve_fft
              def richardson_lucy(im, psf, iterations):
                  # Ensure image is float64.
                  im = im.astype(np.float64)
                  im_est = im.copy()  # initial estimate
                  psf_mirror = psf[::-1, ::-1]  # mirror the PSF
                  for i in range(iterations):
                      conv_est = convolve_fft(im_est, psf, normalize_kernel=True)
                      conv_est[conv_est == 0] = 1e-7  # Avoid divide-by-zero
                      relative_blur = im / conv_est
                      correction = convolve_fft(relative_blur, psf_mirror, normalize_kernel=True)
                      im_est *= correction
                  return im_est
          
              # ------------------------------------------------------------------
              # PSF definition and extraction.
              if psf_mode == "Analytical":
                  # Create an analytical Gaussian PSF.
                  sigma = 2.0
                  ks = 25  # kernel size
                  ax = np.linspace(-(ks-1)/2., (ks-1)/2., ks)
                  xx, yy = np.meshgrid(ax, ax)
                  psf = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
                  psf /= psf.sum()  # Normalize the PSF.
                  print("Using an analytical Gaussian PSF.")
              else:
                  # Extract the PSF using the user-provided (x, y) coordinates.
                  # Coordinates here are interpreted as the center of the PSF cutout.
                  from astropy.nddata import Cutout2D
                  x = float(self.lrPsfX.text().strip())
                  y = float(self.lrPsfY.text().strip())
                  size = int(self.lrPsfSize.text().strip())
                  # Create a cutout of size 'size' centered at (x, y)
                  cutout = Cutout2D(image, (x, y), size)
                  psf = cutout.data.copy()
                  # Normalize the PSF.
                  psf -= np.median(psf)
                  psf[psf < 0] = 0
                  if psf.sum() != 0:
                      psf /= psf.sum()
                  print("Using the PSF extracted from the image at (%.2f, %.2f) with size %d." % (x, y, size))
          
              # ------------------------------------------------------------------
              # Deconvolve the image or each channel.
              if image.ndim == 2:
                  deconv = richardson_lucy(image, psf, iters)
              elif image.ndim == 3:
                  deconv_channels = []
                  for c in range(image.shape[2]):
                      deconv_channels.append(richardson_lucy(image[:, :, c], psf, iters))
                  deconv = np.stack(deconv_channels, axis=2)
              else:
                  self.statusLabel.setText("Unsupported dimensions for LrDeconv.")
                  return
          
              # Save the deconvolved image to file.
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
              # Ensure 3-channel image: if not in (3, height, width), transpose accordingly.
              if data.ndim == 3:
                  if data.shape[0] != 3 and data.shape[-1] == 3:
                      data = np.transpose(data, (2, 0, 1))
              else:
                  self.statusLabel.setText("FFT requires a 3-channel image.")
                  return
              # Get channels (assumed order: blue, green, red)
              b, g, r = data[0], data[1], data[2]
              def high_pass_filter(im, cutoff, weight):
                  fft = np.fft.fft2(im)
                  fft_shift = np.fft.fftshift(fft)
                  rows, cols = im.shape
                  crow, ccol = rows // 2, cols // 2
                  rad = int(cutoff * min(rows, cols))
                  mask = np.ones((rows, cols), np.float32)
                  mask[crow-rad:crow+rad, ccol-rad:ccol+rad] = 0
                  fft_shift_filtered = fft_shift * mask
                  fft_inverse = np.fft.ifftshift(fft_shift_filtered)
                  im_filt = np.abs(np.fft.ifft2(fft_inverse))
                  im_weighted = cv2.addWeighted(im, 1 - weight, im_filt.astype(np.float32), weight, 0)
                  return im_weighted
              def feather_image(im, radius, distance):
                  im_blur = cv2.GaussianBlur(im, (radius, radius), 0)
                  mask = np.full(im.shape, 255, dtype=np.uint8)
                  mask_blur = cv2.GaussianBlur(mask, (distance*2+1, distance*2+1), 0)
                  return cv2.bitwise_and(im_blur, im_blur, mask=mask_blur)
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
              # Process each channel if 3D.
              if data.ndim == 3:
                  res_channels = []
                  for c in range(data.shape[2]):
                      channel = data[:, :, c]
                      # Normalize for processing
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
              # Read the input FITS file.
              hdul = fits.open(inp)
              data = hdul[0].data.astype(np.float64)
              hdul.close()
              # Check image dimensions and ensure 3-channel color.
              if data.ndim == 3:
                  if data.shape[-1] == 3:  # assume (ny, nx, 3)
                      data = np.transpose(data, (2, 0, 1))  # to (3, ny, nx)
                  elif data.shape[0] == 3:
                      pass
                  else:
                      self.statusLabel.setText("Unexpected color image shape in HpMore.")
                      return
              else:
                  self.statusLabel.setText("Input FITS is not a 3-channel image in HpMore.")
                  return
              # Define a 5x5 high-pass kernel (Laplacian-style)
              hp_kernel_5x5 = np.array([
                  [-1, -1, -1, -1, -1],
                  [-1,  1,  2,  1, -1],
                  [-1,  2,  4,  2, -1],
                  [-1,  1,  2,  1, -1],
                  [-1, -1, -1, -1, -1]
              ], dtype=float)
              # Process each channel with convolution and thresholding.
              filtered_channels = np.empty_like(data)
              for i in range(data.shape[0]):
                  channel = data[i]
                  highpass = convolve2d(channel, hp_kernel_5x5, mode='same', boundary='symm')
                  threshold = np.percentile(channel, 70)
                  mask = channel > threshold
                  filtered_channel = channel.copy()
                  filtered_channel[mask] = channel[mask] + highpass[mask]
                  filtered_channels[i] = filtered_channel
              # Save using original ordering (channels first)
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
              # For color images, split channels (assume data stored as (3, ny, nx))
              if data.ndim == 3 and data.shape[0] == 3:
                  # Process each channel separately; here for simplicity we apply processing only on the first channel.
                  # (LocAdapt code can be extended to process all channels.)
                  channel = data[0]
              elif data.ndim == 2:
                  channel = data
              else:
                  self.statusLabel.setText("Unsupported dimensions for LocAdapt.")
                  return
              # Define helper functions for local contrast calculation:
              def compute_optimum_contrast_percentage(image, target_std):
                  current_std = np.std(image)
                  if current_std == 0:
                      return 100.0
                  return (target_std / current_std) * 100.0
              def compute_optimum_feather_distance(image, neighborhood_size, factor=1.0):
                  adjusted_size = max(1, neighborhood_size - 1)
                  kernel = np.ones((adjusted_size, adjusted_size), dtype=np.float32) / (adjusted_size*adjusted_size)
                  local_mean = convolve(image, kernel, mode='reflect')
                  local_mean_sq = convolve(image**2, kernel, mode='reflect')
                  local_std = np.sqrt(np.abs(local_mean_sq - local_mean**2))
                  return factor * np.median(local_std)
              def contrast_filter(image, neighborhood_size, contrast_factor, feather_distance):
                  adjusted_size = max(1, neighborhood_size - 1)
                  kernel = np.ones((adjusted_size, adjusted_size), dtype=float)
                  kernel /= kernel.size
                  local_mean = convolve(image, kernel, mode='reflect')
                  enhanced = (image - local_mean) * contrast_factor + local_mean
                  squared = np.square(image)
                  local_mean_sq = convolve(squared, kernel, mode='reflect')
                  local_std = np.sqrt(np.abs(local_mean_sq - np.square(local_mean)))
                  weight = np.clip(local_std / feather_distance, 0, 1)
                  return weight * enhanced + (1 - weight) * image
              opt_contrast = compute_optimum_contrast_percentage(channel, target_std)
              contrast_factor = opt_contrast / 100.0
              opt_feather = compute_optimum_feather_distance(channel, neigh, feather_dist)
              fd = opt_feather / 100.0
              enhanced = contrast_filter(channel, neigh, contrast_factor, fd)
              # For color images, you might simply replace the first channel.
              if data.ndim == 3 and data.shape[0] == 3:
                  data[0] = enhanced
                  result = data
              else:
                  result = enhanced
              fits.writeto(outp, result.astype(np.float64), header, overwrite=True)
              self.statusLabel.setText("LocAdapt saved to " + outp)
      
      # ----------------------------
      # Main method
      def main():
          app = QApplication(sys.argv)
          window = ImageFiltersWindow()
          window.show()
          app.exec()
      
      if __name__ == "__main__":
          main()

      
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def AlignImgs():

  # tile dimensions – tune to your RAM/I/O balance
  CHUNK_Y = 1024
  CHUNK_X = 1024
  
  def process_chunk(chunk):
      """
      Example per‐chunk routine: simple min-max normalization.
      Swap this out for bias subtraction, filtering, stats, etc.
      """
      # Pre-check for an all-NaN slice and skip or fill it yourself
      #
      if np.all(np.isnan(chunk)):
          mn = 0.0   # e.g. 0 or a user-defined background
      else:
          mn = np.nanmin(chunk)
  
  
      mn = np.nanmin(chunk)
      mx = np.nanmax(chunk)
      if mx > mn:
          return (chunk - mn) / (mx - mn)
      return chunk
  
  def AlignImgs():
      class AlignImagesForm(QWidget):
          def __init__(self):
              super().__init__()
              self.setWindowTitle("Align images")
              self.resize(600, 400)
  
              # choose how many images (2–13)
              self.count_combo = QComboBox()
              self.count_combo.addItems([f"{i:02d}" for i in range(2,14)])
              self.count_combo.currentTextChanged.connect(self._update_fields)
  
              # input / output groups + layouts
              self.input_group  = QGroupBox("Reference FITS files")
              self.output_group = QGroupBox("Aligned FITS files")
              self.input_form   = QFormLayout()
              self.output_form  = QFormLayout()
              self.input_group.setLayout(self.input_form)
              self.output_group.setLayout(self.output_form)
  
              # keep parallel lists so we can show/hide rows
              self.in_labels      = []
              self.in_containers  = []
              self.out_labels     = []
              self.out_containers = []
  
              for i in range(13):
                  # build Input row
                  lab_in = QLabel(f"Image {i+1}:")
                  edit_in = QLineEdit()
                  btn_in  = QPushButton("Browse…")
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
                  btn_out  = QPushButton("Browse…")
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
              fn, _ = QFileDialog.getOpenFileName(
              self, f"Select reference #{idx+1}", "", "FITS Files (*.fits)"
              )
              if fn:
                  le = self.in_containers[idx].findChild(QLineEdit)
                  le.setText(fn)
  
          def _browse_output(self, idx):
              fn, _ = QFileDialog.getSaveFileName(
                  self, f"Save aligned #{idx+1}", "", "FITS Files (*.fits)"
              )
              if fn:
                  le = self.out_containers[idx].findChild(QLineEdit)
                  le.setText(fn)
  
          def _on_align(self):
              count = int(self.count_combo.currentText())
              inputs  = [self.in_containers[i].findChild(QLineEdit).text().strip()
                         for i in range(count)]
              outputs = [self.out_containers[i].findChild(QLineEdit).text().strip()
                         for i in range(count)]
  
              if any(not p for p in inputs+outputs):
                  QMessageBox.warning(self, "Missing", "Fill in all paths.")
                  return
  
              try:
                  # 1) Read everyone’s data & header (memmap!)
                  dw = []
                  for fn in inputs:
                      hdul = fits.open(fn, memmap=True)
                      dw.append((hdul[0].data.astype(np.float64), hdul[0].header))
                      hdul.close()
  
                  # 2) Compute common WCS + output shape
                  wcs_out, shape_out = find_optimal_celestial_wcs(dw)
  
                  # 3) For each input → reproject in tiles → write memmap
                  for (fn, (data, hdr)), outfn in zip(zip(inputs, dw), outputs):
                      # prepare an empty memmap’d FITS
                      primary = fits.PrimaryHDU(data=np.zeros(shape_out, dtype=np.float32),
                                                header=wcs_out.to_header())
                      primary.writeto(outfn, overwrite=True)
                      out_hdul = fits.open(outfn, mode='update', memmap=True)
                      out_mem  = out_hdul[0].data
  
                      # tile over Y/X
                      ny, nx = shape_out
                      for y0 in range(0, ny, CHUNK_Y):
                          y1 = min(y0 + CHUNK_Y, ny)
                          for x0 in range(0, nx, CHUNK_X):
                              x1 = min(x0 + CHUNK_X, nx)
  
                              # adjust WCS CRPIX so reproject_interp
                              # only fills [y0:y1, x0:x1]
                              wcs_tile = copy.deepcopy(wcs_out)
                              wcs_tile.wcs.crpix[0] -= x0
                              wcs_tile.wcs.crpix[1] -= y0
  
                              # do the reprojection for this tile
                              arr_tile, _ = reproject_interp(
                                  (data, hdr),
                                  wcs_tile,
                                  shape_out=(y1-y0, x1-x0)
                              )
  
                              # apply your per‐tile tweak
                              arr_tile = process_chunk(arr_tile)
  
                              # write it back
                              out_mem[y0:y1, x0:x1] = arr_tile
  
                      out_hdul.flush()
                      out_hdul.close()
  
                  QMessageBox.information(self, "Done", f"Aligned {count} images.")
              except Exception as e:
                  QMessageBox.critical(self, "Error", str(e))
  
      # bootstrap Qt
      app = QApplication(sys.argv)
      w = AlignImagesForm()
      w.show()
      app.exec()
  
  if __name__ == "__main__":
      AlignImgs()    

  return sysargv1
  menue()

def Stacker():
                        
  try:
                           
      def get_fits_files(input_dir):
          pattern = os.path.join(input_dir, "*.fits")
          files = sorted(glob.glob(pattern))
          if not files:
              raise FileNotFoundError(f"No FITS files found in {input_dir}")
          return files

      def reproject_all_to_ref(files):
          # Use first file as reference
          ref_hdr = fits.getheader(files[0], ext=0)
          ref_wcs = WCS(ref_hdr)
          ny, nx = ref_hdr["NAXIS2"], ref_hdr["NAXIS1"]

          cube = []
          for fn in files:
              with fits.open(fn) as hdul:
                  data = hdul[0].data.astype(np.float64)
                  wcs_in = WCS(hdul[0].header)
              reprojected, _ = reproject_interp((data, wcs_in),
                                                ref_wcs,
                                                shape_out=(ny, nx))
              cube.append(reprojected)
          return np.stack(cube, axis=0), ref_hdr

      def compute_average(cube):
          # Pixel-wise mean, ignore NaNs
          return np.nanmean(cube, axis=0)

      def save_as_float32(data64, header, output_file):
          # Convert to float32 and update header
          data32 = data64.astype(np.float32)
          hdr = header.copy()
          hdr["BITPIX"] = -32
          hdu = fits.PrimaryHDU(data=data32, header=hdr)
          hdu.writeto(output_file, overwrite=True)
          print(f"Saved average-stack as float32 to '{output_file}'")

      def display_image(data, header):
          plt.figure(figsize=(8, 6))
          ax = plt.subplot(projection=WCS(header))
          im = ax.imshow(data, origin="lower", cmap="gray")
          plt.colorbar(im, ax=ax, orientation="vertical", label="Flux")
          plt.title("Average-Stacked Image (WCS matched to first FITS)")
          plt.xlabel("RA")
          plt.ylabel("Dec")
          plt.tight_layout()
          plt.show()

      def main():
          input_dir = input("Enter the directory containing FITS files --> ").strip()
          output_file = input("Enter the output FITS filename     --> ").strip()

          files = get_fits_files(input_dir)
          cube, ref_hdr = reproject_all_to_ref(files)
          avg_image = compute_average(cube)
          save_as_float32(avg_image, ref_hdr, output_file)
          display_image(avg_image, ref_hdr)

      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def Normalize():
                        
  try:
                               
      def load_fits(path):
          """Read a FITS file and return (data, header)."""
          hdul = fits.open(path)
          data = hdul[0].data
          hdr = hdul[0].header
          hdul.close()
          return data, hdr

      def show_histogram(data, title="Histogram"):
          """Display histogram(s) of a 2D or 3D image array."""
          plt.figure()
          if data.ndim == 3:
              # detect channel-first vs. channel-last
              if data.shape[0] <= 4:
                  channels = [data[i] for i in range(data.shape[0])]
              else:
                  channels = [data[..., i] for i in range(data.shape[2])]
              for idx, ch in enumerate(channels):
                  plt.hist(ch.ravel(), bins=256, alpha=0.5, label=f"Ch{idx}")
              plt.legend()
          else:
              plt.hist(data.ravel(), bins=256, color="gray")
          plt.title(title)
          plt.xlabel("Pixel value")
          plt.ylabel("Count")
          plt.show()

      def normalize(data, old_min, old_max, new_min, new_max):
          """Linearly rescale data from [old_min,old_max] → [new_min,new_max]."""
          arr = data.astype(np.float64)
          scaled = (arr - old_min) / (old_max - old_min)
          scaled = scaled * (new_max - new_min) + new_min
          return np.clip(scaled, min(new_min, new_max), max(new_min, new_max))

      def print_image_stats(data, label="Image"):
          """
          Print basic statistics for a 2D or 3D NumPy array:
            • If 2D, prints one set of stats
            • If 3D, treats as channels and prints stats per channel
          """
          def stats(arr):
              return {
                  "min":    float(arr.min()),
                  "max":    float(arr.max()),
                  "median": float(np.median(arr)),
                  "mean":   float(arr.mean()),
                  "std":    float(arr.std())
              }

          print(f"{label} stats:")
          if data.ndim == 2:
              s = stats(data)
              print(f"  shape: {data.shape}")
              print(f"  min:    {s['min']:.6g}")
              print(f"  max:    {s['max']:.6g}")
              print(f"  median: {s['median']:.6g}")
              print(f"  mean:   {s['mean']:.6g}")
              print(f"  std:    {s['std']:.6g}")
          elif data.ndim == 3:
              # decide channel‐first vs. channel‐last
              if data.shape[0] <= 4:
                  channels = [data[i] for i in range(data.shape[0])]
                  idxs = range(data.shape[0])
              else:
                  channels = [data[..., i] for i in range(data.shape[2])]
                  idxs = range(data.shape[2])

              for i, ch in zip(idxs, channels):
                  s = stats(ch)
                  print(f"  Channel {i}: shape {ch.shape}")
                  print(f"    min:    {s['min']:.6g}")
                  print(f"    max:    {s['max']:.6g}")
                  print(f"    median: {s['median']:.6g}")
                  print(f"    mean:   {s['mean']:.6g}")
                  print(f"    std:    {s['std']:.6g}")
          else:
              flat = data.ravel()
              s = stats(flat)
              print(f"  (ndim={data.ndim}) flattened:")
              print(f"    min:    {s['min']:.6g}")
              print(f"    max:    {s['max']:.6g}")
              print(f"    median: {s['median']:.6g}")
              print(f"    mean:   {s['mean']:.6g}")
              print(f"    std:    {s['std']:.6g}")
          print()

      def open_in_siril(fits_path):
          # Ensure 'siril' is on the PATH
          if shutil.which('siril') is None:
              print("Error: 'siril' command not found. Install Siril or add it to your PATH.")
              sys.exit(1)

          try:
              # This will open Siril's GUI with your FITS file
              subprocess.run(['siril', fits_path], check=True)
          except subprocess.CalledProcessError as e:
              print(f"Failed to launch Siril: {e}")

      def main():
          fits_path = input("Enter path to FITS file: ").strip()
          data, hdr = load_fits(fits_path)

          # Print stats “as-is”
          print_image_stats(data, label="Original FITS")

          # Display histogram of the image
          show_histogram(data, title="Original Image Histogram")
          open_in_siril(fits_path)

          old_min = float(input("Enter old minimum: ").strip())
          old_max = float(input("Enter old maximum: ").strip())
          new_min = float(input("Enter new minimum: ").strip())
          new_max = float(input("Enter new maximum: ").strip())

          norm_data = normalize(data, old_min, old_max, new_min, new_max)

          out_path = input("Enter output FITS filename: ").strip()
          fits.writeto(out_path, norm_data.astype(np.float64), hdr, overwrite=True)
          print(f"Normalized FITS saved to {out_path}")

          # Print stats “as-is”
          print_image_stats(norm_data, label="Original FITS")

          # Display histogram of the Stretched image
          show_histogram(norm_data, title="New Image Histogram")
          open_in_siril(out_path)

      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

def RaDecTwoPtAng():

  try:
          
      def separation_and_arc_length(ra1, dec1, ra2, dec2):
            """
          Calculate the angular separation between two sky coordinates
          and the corresponding arc length on a unit sphere.

          Parameters
          ----------
           ra1, dec1 : str
           Right ascension and declination of point 1 (e.g. "12h30m00s", "+45d00m00s").
          ra2, dec2 : str
           Right ascension and declination of point 2.

          Returns
          -------
          sep_dms : tuple of int
              (degrees, arcminutes, arcseconds) of the separation angle.
          arc_length : float
              Arc length on a sphere of radius 1 (same units as input distance).
          """
            # Create SkyCoord objects
            c1 = SkyCoord(ra1, dec1, unit=(u.hourangle, u.deg))
            c2 = SkyCoord(ra2, dec2, unit=(u.hourangle, u.deg))

            # Compute separation angle
            sep = c1.separation(c2).to(u.deg)

            # Break down into degrees, minutes, seconds
            deg = int(sep.value)
            arcmin = int((sep.value - deg) * 60)
            arcsec = (sep.value - deg - arcmin/60) * 3600

            # Arc length on unit sphere = separation in radians * radius (1)
            arc_length = c1.separation(c2).to(u.rad).value

            return (deg, arcmin, arcsec), arc_length

      if __name__ == "__main__":
          # Example input
        sysargv1  = input("Enter Ra1  example   10h21m00s  -->")
        sysargv2  = input("Enter Dec1 example  +20d30m00s  -->")
        sysargv3  = input("Enter Ra2  example   11h15m30s  -->")
        sysargv4  = input("Enter Dec1 example  +22d05m15s  -->")


        ra1, dec1 = sysargv1, sysargv2
        ra2, dec2 = sysargv3, sysargv4

        sep_dms, length_unit = separation_and_arc_length(ra1, dec1, ra2, dec2)
        print(f"Separation: {sep_dms[0]}° {sep_dms[1]}' {sep_dms[2]:.3f}\"")
        print(f"Arc length on unit sphere: {length_unit:.6f} (same units as radius)")

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()



def menue(sysargv1):
#  sysargv1 = input("Enter \n>>1<< AffineTransform(3pts) >>2<< Mask an image >>3<< Mask Invert >>4<< Add2images(fit)  \n>>5<< Split tricolor >>6<< Combine Tricolor >>7<< Create Luminance(2ax) >>8<< Align2img \n>>9<< Plot_16-bit_img to 3d graph(2ax) >>10<< Centroid_Custom_filter(2ax) >>11<< UnsharpMask \n>>12<< FFT-(RGB) >>13<< Img-DeconvClr >>14<< Centroid_Custom_Array_loop(2ax) \n>>15<< Erosion(2ax) >>16<< Dilation(2ax) >>17<< DynamicRescale(2ax) >>18<< GausBlur  \n>>19<< DrCntByFileType >>20<< ImgResize >>21<< JpgCompress >>22<< subtract2images(fit)  \n>>23<< multiply2images >>24<< divide2images >>25<< max2images >>26<< min2images \n>>27<< imgcrop >>28<< imghiststretch >>29<< gif  >>30<< aling2img(2pts) >>31<< Video \n>>32<< gammaCor >>33<< ImgQtr >>34<< CpyOldHdr >>35<< DynReStr(RGB) \n>>36<< clahe >>37<< pm_vector_line >>38<< hist_match >>39<< distance >>40<< EdgeDetect \n>>41<< Mosaic(4) >>42<< BinImg >>43<< autostr >>44<< LocAdapt >>45<< WcsOvrlay \n>>46<< Stacking >>47<< CombineLRGB >>48<< MxdlAstap >>49<< CentRatio >>50<< ResRngHp \n>>51<< CombBgrAlIm >>52<< PixelMath >>53<< Color >>54<< ImageFilters \n>>1313<< Exit --> ")
  sysargv1 = input("Enter \n>>1<< AffineTransform(3pts) >>2<< Mask an image >>3<< Mask Invert  >>9<< Plot_img to 3d (2ax) \n>>10<< Centroid_Custom_filter(2ax) >>5<< DirSpltAllRgb \n>>14<< Centroid_Custom_Array_loop(2ax) >>17<< DynamicRescale(2ax) >>19<< DrCntByFileType \n>>>20<< ImgResize >>21<< JpgCompress >>27<< imgcrop >>28<< imghiststretch >>29<< gif \n>30<< aling2img(2pts) >>31<< Video >>32<< gammaCor >>33<< ImgQtr >>34<< CpyOldHdr \n>>35<< DynReStr(RGB) >>36<< clahe >>37<< pm_vector_line >>38<< hist_match >>39<< distance \n>>40<< EdgeDetect >>41<< Mosaic(4) >>42<< BinImg >>43<< autostr >>44<< Rank \n>>45<< WcsOvrlay >>46<< AlnImgsByDir >>47<< CombineLRGB >>48<< MxdlAstap >>49<< CentRatio \n>>51<< CombBgrAlIm >>52<< PixelMath >>53<< Color >>54<< ImageFilters >>55<< AlignImgs \n>>56<< Stacker >>57<< FitQc >>58<< Normalize  >>59<< RaDec2ptAng\n>>1313<< Exit --> ")

  return sysargv1

sysargv1 = ''
while not sysargv1 == '1313':  # Substitute for a while-True-break loop.
  sysargv1 = ''
  sysargv1 = menue(sysargv1)

  if sysargv1 == '1':
    AffineTransform()

  if sysargv1 == '2':
    mask()

  if sysargv1 == '3':
    maskinvert()

  if sysargv1 == '4':
    add2images()

  if sysargv1 == '5':
    splittricolor()

  if sysargv1 == '6':
    combinetricolor()

  if sysargv1 == '7':
    createLuminance()

  if sysargv1 == '8':
    align2img()

  if sysargv1 == '9':
    sysargv2  = input("Enter the file name -->")
    plotto3d16(sysargv2)
    menue(sysargv1)

  if sysargv1 == '10':
    sysargv2  = input("Enter the file name -->")
    sysargv3 = input("Enter the radius of the file-->")
    sysargv4 = input("Enter the x-coordinate of the centroid-->")
    sysargv5 = input("Enter the y-coordinate of the centroid-->")
    PNGcreateimage16(sysargv2, sysargv3, sysargv4, sysargv5)
    menue(sysargv1)

  if sysargv1 == '11':
    unsharpMask()

  if sysargv1 == '12':
    FFT()

  if sysargv1 == '13':
    LrDeconv()

  if sysargv1 == '14':
    sysargv2  = input("Enter the file name -->")
    sysargv3 = input("Enter the radius of the file-->")
    sysargv4 = input("Enter the x-coordinate of the centroid-->")
    sysargv5 = input("Enter the y-coordinate of the centroid-->")
    x = int(sysargv4)
    y = int(sysargv5)
    for i in range(3):
      i1=int(i + x - 1)
      file = str(i1)
      for j in range(3):
        j1=int(j + y - 1)
        file1 = str(j1)
        sysargv4 = file
        sysargv5 = file1
        PNGcreateimage16(sysargv2, sysargv3, sysargv4, sysargv5)
    menue(sysargv1)
 
  if sysargv1 == '15':
    erosion()
 
  if sysargv1 == '16':
    dilation()

  if sysargv1 == '17':
    DynamicRescale16()

  if sysargv1 == '18':
    gaussian()

  if sysargv1 == '19':
    filecount()

  if sysargv1 == '20':
    resize()

  if sysargv1 == '21':
    jpgcomp()

  if sysargv1 == '22':
    subtract2images()

  if sysargv1 == '23':
    multiply2images()

  if sysargv1 == '24':
    divide2images()

  if sysargv1 == '25':
    max2images()

  if sysargv1 == '26':
    min2images()

  if sysargv1 == '27':
    imgcrop1()

  if sysargv1 == '28':
    imghiststretch()

  if sysargv1 == '29':
    gif()

  if sysargv1 == '30':
    alingimg()

  if sysargv1 == '31':
    video()

  if sysargv1 == '32':
    gamma()

  if sysargv1 == '33':
    imgqtr()

  if sysargv1 == '34':
    CpyOldHdr()    

  if sysargv1 == '35':
    DynamicRescale16RGB()

  if sysargv1 == '36':
    clahe()

  if sysargv1 == '37':
    pm_vector_line()

  if sysargv1 == '38':
    hist_match()
  
  if sysargv1 == '39':
    distance()
  
  if sysargv1 == '40':
    edgedetect()

  if sysargv1 == '41':
    mosaic()

  if sysargv1 == '42':
    binimg()

  if sysargv1 == '43':
    autostr()

  if sysargv1 == '44':
      # read+exec with utf-8
      with open("wrank.py", "r", encoding="utf-8") as f:
          code = f.read()
      exec(code, globals(), locals())
      menue(sysargv1)

  if sysargv1 == '45':
    WcsOvrlay()

  if sysargv1 == '46':
     Stacking()

  if sysargv1 == '47':
    combinelrgb()

  if sysargv1 == '48':
    MxdlAstap()

  if sysargv1 == '49':
    CentRatio()

  if sysargv1 == '50':
    HpMore()

  if sysargv1 == '51':
    CombBgrAlIm()

  if sysargv1 == '52':
    pixelmath()

  if sysargv1 == '53':
    Color()

  if sysargv1 == '54':
    ImageFilters()

  if sysargv1 == '55':
    AlignImgs()

  if sysargv1 == '56':
    Stacker()

  if sysargv1 == '57':
      # read+exec with utf-8
      with open("analyze_fits_roundness_trails.py", "r", encoding="utf-8") as f:
          code = f.read()
      exec(code, globals(), locals())
      menue(sysargv1)

  if sysargv1 == '58':
    Normalize()

  if sysargv1 == '59':
    RaDecTwoPtAng()

  if sysargv1 == '1313':
    sys.exit()



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      