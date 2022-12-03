#python astrotools_tileloop.py Lm33t2.tif 950 950 700 0 99 
from numba import jit
from numba import prange
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
  n = 0/4294836225
  first_image = None
  stacked_image = None
  global file, file1, im1
  xa = (int(sys.argv[5])) + 0
  for k in range(int(int((int(sys.argv[3])/100)*(int(sys.argv[6])))-1)):
    ya = 0
    xa = xa + (int(100/(int(sys.argv[6]))))
    for k1 in range(int(int((int(sys.argv[4])/100)*(int(sys.argv[6])))-1)):
      ya = ya + (int(100/(int(sys.argv[6]))))
  # centroid location rounded x,y
      x = int(xa)
      y = int(ya)
      print (xa, ya)
      if xa >= (int(sys.argv[3])):
        sys.exit()
      for i in range(3):
        i1=int(i + x - 1)
        file = str(i1)
        for j in range(1):
          j1=int(j + y - 1)
          file1 = str(j1)
          imgcrop()        
          my_data = PNGcreateimage16()
          n = n + 1
          image = my_data.astype(np.float64) / 4294836225
          if first_image is None:
              first_image = image
              stacked_image = image
          else:
              stacked_image += image   
  stacked_image /= n
  stacked_image = (stacked_image*4294836225).astype(np.uint16)
  cv2.imwrite(('resultnew'+sys.argv[1]),stacked_image)
  print (n)

@jit(nopython=True, parallel = True, nogil = True)
def PNGcreateimage16loop(img, my_data, diameterp1):
  for x in prange(diameterp1):
    for y in range(diameterp1):
      my_data[x,y]=img[x,y]-min(img[x,y],img[x,diameter-y],img[diameter-x,y],img[diameter-x,diameter-y])
  return my_data

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
  my_data = PNGcreateimage16loop(img, my_data, diameterp1)    
#Rescale to 0-65535 and convert to uint16
  my_data = my_data[my_data.any(axis=1)]
  my_data = np.transpose(my_data)
  my_data = my_data[my_data.any(axis=1)]
  my_data = np.transpose(my_data)
  #rescaled = (65535.0 / my_data.max() * (my_data - my_data.min())).astype(np.uint16)
  #im = Image.fromarray(rescaled)
  #symfile = ("img_pixels2"+"_"+file+"_"+file1+".png")
  #print(symfile)
  #im.save(symfile)
  return my_data

def imgcoord(radius, radiusp1, diameter, diameterp1, one, two, three, four):
  radius = int(int(sys.argv[2]))
  radiusp1 = int((int(sys.argv[2]))+1)
  diameter = int(int(sys.argv[2])*2)
  diameterp1 = int((int(sys.argv[2])*2)+1)
  one = int((int(file) - radius))
  two = int((int(file1) - radius))
  three = int((int(file) + radiusp1))
  four = int((int(file1) + radiusp1))
  #print(radius, radiusp1, diameter, diameterp1, one, two, three, four)
  return radius, radiusp1, diameter, diameterp1, one, two, three, four

def imgcrop():
  global im, im1 
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

def CLAHE():
  image = cv2.imread(('resultnew'+sys.argv[1]),-1) 
  image_bw = image 
  # The declaration of CLAHE
  # clipLimit -> Threshold for contrast limiting
  #clahe = cv2.createCLAHE(clipLimit = 5)
  #final_img = clahe.apply(image_bw) + 30
  clahe = cv2.createCLAHE(clipLimit = 2)
  final_img = clahe.apply(image_bw)
  cv2.imwrite(('final_img'+('resultnew'+sys.argv[1])),final_img)

def sn_incr():
  image = cv2.imread(('resultnew'+sys.argv[1]), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32) / 4294836225.0
  my_data = image * image
  rescaled = (655350.0 / my_data.max() * (my_data - my_data.min())).astype(np.uint16)
  #rescaled = (rescaled).astype(np.uint16)
  cv2.imwrite(('resultnew'+('sn_incr'+sys.argv[1])), rescaled)

def CLAHEsn_incr():
  image = cv2.imread(('resultnewsn_incr'+sys.argv[1]),-1) 
  image_bw = image 
  # The declaration of CLAHE
  # clipLimit -> Threshold for contrast limiting
  #clahe = cv2.createCLAHE(clipLimit = 5)
  #final_img = clahe.apply(image_bw) + 30
  clahe = cv2.createCLAHE(clipLimit = 2)
  final_img = clahe.apply(image_bw)
  cv2.imwrite(('final_img'+('resultnewsn_incr'+sys.argv[1])),final_img)

if __name__ == "__main__":
  main()
  sn_incr()
  CLAHE()
  CLAHEsn_incr()