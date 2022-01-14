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
  im.save('img_pixels2.png')
  image_1 = imread('img_pixels2.png')
  # plot raw pixel data
  pyplot.imshow(image_1)
  # show the figure
  pyplot.show()

def imgcoord(radius, radiusp1, diameter, diameterp1, one, two, three, four):
  radius = int(int(sys.argv[3]))
  radiusp1 = int((int(sys.argv[3]))+1)
  diameter = int(int(sys.argv[3])*2)
  diameterp1 = int((int(sys.argv[3])*2)+1)
  one = int((int(sys.argv[4]) - radius))
  two = int((int(sys.argv[5]) - radius))
  three = int((int(sys.argv[4]) + radiusp1))
  four = int((int(sys.argv[5]) + radiusp1))
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
  im = Image.open(sys.argv[2])
  im1 = im.crop((one, two, three, four))
  im1.save('crop.png')


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

if sys.argv[1] == '7':
  imgcrop()


