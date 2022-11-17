#python astrotools_tileloop.py B.tif 950 950 700 1 4
from numba import jit
import subprocess, shutil, os, sys
import numpy as np
from PIL import Image
import cv2
im = Image.open(sys.argv[1])
file = ""
file1 = ""
radius = 0
radiusp1 = 0
one = 0
two = 0
three = 0
four = 0
diameter = 0
diameterp1 = 0
im1 = im.crop((one, two, three, four))

def main():
  i = 0
  global file, file1
  xa = (int(sys.argv[5])) + 0
  for k in range(int(int((int(sys.argv[3])/100)*(int(sys.argv[6])))-1)):
    ya = 0
    xa = xa + (int(100/(int(sys.argv[6]))))
    for k1 in range(int(int((int(sys.argv[4])/100)*(int(sys.argv[6])))-1)):
      ya = ya + (int(100/(int(sys.argv[6]))))
  # centroid location rounded x,y
      x = int(xa)
      y = int(ya)
      print (xa)
      print (ya)
      if xa >= (int(sys.argv[3])):
        sys.exit()
      for i in range(3):
        i1=int(i + x - 1)
        file = str(i1)
        for j in range(3):
          j1=int(j + y - 1)
          file1 = str(j1)
          imgcrop()
          PNGcreateimage16()

@jit(nopython=False)
def PNGcreateimage16():
  global file, file1
  radius = 0  
  radiusp1 = 0
  one = 0
  two = 0
  three = 0
  four = 0
  diameter = 0
  diameterp1 = 0
  radius, radiusp1, diameter, diameterp1, one, two, three, four = imgcoord(radius, radiusp1, diameter, diameterp1, one, two, three, four)
  my_data = np.array(im1)
  img = my_data
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
  symfile = ("img_pixels2"+"_"+file+"_"+file1+".png")
  print(symfile)
  im.save(symfile)

def imgcoord(radius, radiusp1, diameter, diameterp1, one, two, three, four):
  radius = int(int(sys.argv[2]))
  radiusp1 = int((int(sys.argv[2]))+1)
  diameter = int(int(sys.argv[2])*2)
  diameterp1 = int((int(sys.argv[2])*2)+1)
  one = int((int(sys.argv[3]) - radius))
  two = int((int(sys.argv[4]) - radius))
  three = int((int(sys.argv[3]) + radiusp1))
  four = int((int(sys.argv[4]) + radiusp1))
  #print(radius, radiusp1, diameter, diameterp1, one, two, three, four)
  return radius, radiusp1, diameter, diameterp1, one, two, three, four

def imgcrop():
  global im1
  radius = 0
  radiusp1 = 0
  one = 0
  two = 0
  three = 0
  four = 0
  diameter = 0
  diameterp1 = 0
  radius, radiusp1, diameter, diameterp1, one, two, three, four = imgcoord(radius, radiusp1, diameter, diameterp1, one, two, three, four)
  im1 = im.crop((one, two, three, four)) 

if __name__ == "__main__":
  main()