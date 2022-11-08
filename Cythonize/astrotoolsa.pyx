#!python
#cython: language_level=3

#python astrotools_tileloop.py B.tif 950 950 700 1 4
import subprocess
import sys
import shutil,os
import numpy as np
from PIL import Image
import cv2

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
  for x in range(diameterp1):
    for y in range(diameterp1):
        my_data[x,y]=img[x,y]-min(img[x,y],img[x,diameter-y],img[diameter-x,y],img[diameter-x,diameter-y])     
#Rescale to 0-65535 and convert to uint16
  my_data = my_data[my_data.any(axis=1)]
  my_data = np.transpose(my_data)
  my_data = my_data[my_data.any(axis=1)]
  my_data = np.transpose(my_data)
  rescaled = (65535.0 / my_data.max() * (my_data - my_data.min())).astype(np.uint16)
  im = Image.fromarray(rescaled)
  symfile = ("img_pixels2"+"_"+sys.argv[4]+"_"+sys.argv[5]+".png")
  im.save(symfile)

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

if sys.argv[1] == '6':   
    PNGcreateimage16()

