# import required libraries
from PIL import Image
import cv2, sys
import numpy as np

# function to display the coordinates of 
# of the points clicked on the image 
def click_event(event, x, y, flags, params): 

  # checking for left mouse clicks 
  if event == cv2.EVENT_LBUTTONDOWN: 
    # displaying the coordinates 
    # on the Shell 
    print(' x ', ' ', ' y ')
    print(x, ' ', y) 

    # displaying the coordinates 
    # on the image window 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(img, str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2) 
    cv2.imshow('image', img) 

  # checking for right mouse clicks	 
  if event==cv2.EVENT_RBUTTONDOWN: 
    # displaying the coordinates 
    # on the Shell
    print(' x ', ' ', ' y ')
    print(x, ' ', y) 

    # displaying the coordinates 
    # on the image window 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    b = img[y, x, 0] 
    g = img[y, x, 1] 
    r = img[y, x, 2] 
    cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x,y), font, 1, (255, 255, 0), 2) 
    cv2.imshow('image', img) 

def mask():
  sysargv1  = input("Enter the Image1  -->")
  sysargv3  = input("Enter the Mask for image  white- and black  -->")
  sysargv4  = input("Enter the filename of the masked image to save  -->")

  image = cv2.imread(sysargv1)
  mask = cv2.imread(sysargv3)

  # Apply the mask to the image
  masked_image = cv2.bitwise_and(image, mask)
  cv2.imwrite(sysargv4, masked_image)
  exit()

def maskinvert():
  sysargv1  = input("Enter the mask Image  -->")
  #sysargv2  = input("Enter the Image2  -->")
  sysargv3  = input("Enter the output invert Mask -->")
  image = cv2.imread(sysargv1)
    # Apply the inverted mask to the image
  masked_image = cv2.bitwise_not(image)
  cv2.imwrite(sysargv3, masked_image)
  exit()

def add2images():
  sysargv1  = input("Enter the enter fist masked Image  -->")
  sysargv3  = input("Enter the enter second masked Image  -->")
  sysargv4  = input("Enter the filename of the added images to save  -->")

  image = cv2.imread(sysargv1)
  mask = cv2.imread(sysargv3)

  # Apply the mask to the image
  masked_image = cv2.add(image, mask)
  cv2.imwrite(sysargv4, masked_image)
  exit()

sysargv1 = input("Enter >>1<< AffineTransform or >>2<< Mask an image  >>3<< Mask Invert >>4<< add2images  >>5<< Exit  -->")

if sysargv1 == '2':
  mask()

if sysargv1 == '3':
  maskinvert()

if sysargv1 == '4':
  add2images()

if sysargv1 == '5':
  exit()

sysargv1  = input("Enter the the original mask file name -->")
sysargv2  = input("Enter the comparison      file name -->")
sysargv2a  = input("Enter the new mask      file name -->")
sysargv3  = int(input("Enter the x-coordinate size new file -->"))
sysargv4  = int(input("Enter the y-coordinate size new file -->"))

# reading the image
win_name = "visualization"  #  1. use var to specify window name everywhere
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  #  2. use 'normal' flag
img = cv2.imread(sysargv1, 1) 

# displaying the image 
cv2.imshow('image', img) 

# setting mouse handler for the image 
# and calling the click_event() function 
cv2.setMouseCallback('image', click_event) 

# wait for a key to be pressed to exit 
cv2.waitKey(0) 

# close the window 
cv2.destroyAllWindows() 

sysargv5  = input("Enter the x-coordinate of p1 origin file name-->")
sysargv6  = input("Enter the y-coordinate of p1 origin file name-->")
sysargv7  = input("Enter the x-coordinate of p2 origin file name-->")
sysargv8  = input("Enter the y-coordinate of p2 origin file name-->")
sysargv9  = input("Enter the x-coordinate of p3 origin file name-->")
sysargv10 = input("Enter the y-coordinate of p3 origin file name-->")

# reading the image 
img = cv2.imread(sysargv2, 1) 

# displaying the image 
cv2.imshow('image', img) 

# setting mouse handler for the image 
# and calling the click_event() function 
cv2.setMouseCallback('image', click_event) 

# wait for a key to be pressed to exit 
cv2.waitKey(0) 

  # close the window 
cv2.destroyAllWindows() 

sysargv11 = input("Enter the X-coordinate of p1 new    file name-->")
sysargv12 = input("Enter the y-coordinate of p1 new    file name-->")
sysargv13 = input("Enter the x-coordinate of p2 new    file name-->")
sysargv14 = input("Enter the y-coordinate of p2 new    file name-->")
sysargv15 = input("Enter the x-coordinate of p3 new    file name-->")
sysargv16 = input("Enter the y-coordinate of p3 new    file name-->")

# read the input image
win_name = "visualization"  #  1. use var to specify window name everywhere
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  #  2. use 'normal' flag
img = cv2.imread(sysargv1)
# access the image height and width
rows,cols,_ = img.shape 
# define at three point on input image
pts1 = np.float32([[sysargv5,sysargv6],[sysargv7,sysargv8],[sysargv9,sysargv10]])

# define three points corresponding location to output image
pts2 = np.float32([[sysargv11,sysargv12],[sysargv13,sysargv14],[sysargv15,sysargv16]])

# get the affine transformation Matrix
M = cv2.getAffineTransform(pts1,pts2)

# apply affine transformation on the input image
#dst = cv2.warpAffine(img,M,(cols,rows))
dst = cv2.warpAffine(img,M,(sysargv3,sysargv4))
cv2.imshow("Affine Transform", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
im = Image.fromarray(dst, "RGB")
im.save(sysargv2a)

