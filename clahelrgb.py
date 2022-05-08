import sys
import os
import cv2
import numpy as np
 
image = cv2.imread('L.png',-1)
image_bw = image
clahe = cv2.createCLAHE(clipLimit = 2)
final_img = clahe.apply(image)
cv2.imwrite('final_imgL.tif',final_img)

image = cv2.imread('R.png',-1)
image_bw = image
clahe = cv2.createCLAHE(clipLimit = 2)
final_img = clahe.apply(image)
cv2.imwrite('final_imgR.tif',final_img)

image = cv2.imread('G.png',-1)
image_bw = image
clahe = cv2.createCLAHE(clipLimit = 2)
final_img = clahe.apply(image)
cv2.imwrite('final_imgG.tif',final_img)

image = cv2.imread('B.png',-1)
image_bw = image
clahe = cv2.createCLAHE(clipLimit = 2)
final_img = clahe.apply(image)
cv2.imwrite('final_imgB.tif',final_img)