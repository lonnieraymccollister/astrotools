import sys
import os
import csv
import numpy as np
from PIL import Image
import matplotlib
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
import cv2
import scipy.misc

def csvWriter(fil_name, nparray):
  example = nparray.tolist()
  with open(fil_name+'.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(example)

def CSVcreateimage():
  my_data = genfromtxt('img_pixels2.csv', delimiter=',')
  matplotlib.image.imsave('img_pixels2.png', my_data, cmap='gray')
  image_1 = imread('img_pixels2.png')
  # plot raw pixel data
  pyplot.imshow(image_1)
  # show the figure
  pyplot.show()

def plotto3d(img):
  lena = cv2.imread(img, 0)
  # downscaling has a "smoothing" effect
  #lena = cv2.resize(lena, (14,28))  # create the x and y coordinate arrays (here we just use pixel indices)
  xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]
  # create the figure
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)
  # show it
  plt.show()

if sys.argv[1] == '1':
    img = np.array(Image.open(sys.argv[2]))
    csvWriter("img_pixel", img)

if sys.argv[1] == '2':
    CSVcreateimage()

if sys.argv[1] == '3':
    plotto3d(sys.argv[2])
