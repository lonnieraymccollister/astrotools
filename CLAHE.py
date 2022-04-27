import sys
import os
import cv2
import numpy as np
 
# Reading the image from the present directory
image = cv2.imread(sys.argv[1])
# Resizing the image for compatibility
#image = cv2.resize(image, (485, 449))
 
# The initial processing of the image
# image = cv2.medianBlur(image, 3)
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image_bw = image
 
# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
#clahe = cv2.createCLAHE(clipLimit = 5)
#final_img = clahe.apply(image_bw) + 30
clahe = cv2.createCLAHE(clipLimit = 2)
final_img = clahe.apply(image_bw)
 
cv2.imwrite('final_img.png',final_img)