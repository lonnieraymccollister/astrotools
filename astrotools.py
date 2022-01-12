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

def CSVcreateimage16():
  my_data = genfromtxt('img_pixels2.csv', delimiter=',')
  #Rescale to 0-255 and convert to uint8
  #matplotlib.image.imsave('img_pixels2.png', my_data, cmap='gray')
  #Rescale to 0-65535 and convert to uint16
  rescaled = (65535.0 / my_data.max() * (my_data - my_data.min())).astype(np.uint16)
  im = Image.fromarray(rescaled)
  im.save('img_pixels2.png')
  image_1 = imread('img_pixels2.png')
  # plot raw pixel data
  pyplot.imshow(image_1)
  # show the figure
  pyplot.show()

def plotto3d16(img):
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

def PNGcreateimage16():
  my_data = np.array(Image.open(sys.argv[2]))
  img = np.array(Image.open(sys.argv[2]))
  for x in range(int(sys.argv[3])+1):
    for y in range(int(sys.argv[3])+1):
      my_data[x,y]=img[x,y]-min(img[x,y],img[x,int(sys.argv[3])-y],img[int(sys.argv[3])-x,y],img[int(sys.argv[3])-x,int(sys.argv[3])-y])
      print (x,y,img[x,y],my_data[x,y])
  #Rescale to 0-65535 and convert to uint16
  rescaled = (65535.0 / my_data.max() * (my_data - my_data.min())).astype(np.uint16)
  im = Image.fromarray(rescaled)
  im.save('img_pixels2.png')
  image_1 = imread('img_pixels2.png')
  # plot raw pixel data
  pyplot.imshow(image_1)
  # show the figure
  pyplot.show()

if sys.argv[1] == '1':
    img = np.array(Image.open(sys.argv[2]))
    csvWriter("img_pixel", img)

if sys.argv[1] == '2':
    CSVcreateimage()

if sys.argv[1] == '3':
    plotto3d(sys.argv[2])

if sys.argv[1] == '4':
    CSVcreateimage16()

if sys.argv[1] == '5':
    plotto3d16(sys.argv[2])

if sys.argv[1] == '6':
    PNGcreateimage16()


