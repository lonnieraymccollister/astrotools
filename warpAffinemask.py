# import required libraries
import os
from PIL import Image
import cv2, sys
import numpy as np
import matplotlib
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D


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
  return sysargv1
  menue()

def maskinvert():
  sysargv1  = input("Enter the mask Image  -->")
  #sysargv2  = input("Enter the Image2  -->")
  sysargv3  = input("Enter the output invert Mask -->")
  image = cv2.imread(sysargv1)
    # Apply the inverted mask to the image
  masked_image = cv2.bitwise_not(image)
  cv2.imwrite(sysargv3, masked_image)
  return sysargv1
  menue()

def add2images():
  sysargv1  = input("Enter the fist masked Image  -->")
  sysargv3  = input("Enter the second masked Image  -->")
  sysargv4  = input("Enter the filename of the added images to save  -->")

  image = cv2.imread(sysargv1)
  mask = cv2.imread(sysargv3)

  # Apply the mask to the image
  masked_image = cv2.add(image, mask)
  cv2.imwrite(sysargv4, masked_image)
  return sysargv1
  menue()

def splittricolor():
  sysargv1  = input("Enter the Color Image to be split  -->")
  img = cv2.imread(sysargv1)

  # split the Blue, Green and Red color channels
  blue,green,red = cv2.split(img)
  sysargv2 = "Blue" + sysargv1
  cv2.imwrite(sysargv2, blue)
  sysargv2 = "Green" + sysargv1
  cv2.imwrite(sysargv2, green)
  sysargv2 = "Red" + sysargv1
  cv2.imwrite(sysargv2, red)
  return sysargv1
  menue()

def combinetricolor():
  sysargv1  = input("Enter the Blue image to be combined  -->")
  blue = cv2.imread(sysargv1, cv2.IMREAD_GRAYSCALE)
  sysargv2  = input("Enter the Green image to be combined  -->")
  green = cv2.imread(sysargv2, cv2.IMREAD_GRAYSCALE)
  sysargv3  = input("Enter the Red image to be combined  -->")
  red = cv2.imread(sysargv3, cv2.IMREAD_GRAYSCALE)
  sysargv4  = input("Enter the RGB file to be created  -->")

  # Merge the Blue, Green and Red color channels
  newRGBImage = cv2.merge((red,green,blue))
  cv2.imwrite(sysargv4, newRGBImage)
  return sysargv1
  menue()

def createLuminance():
  sysargv1  = input("Enter the Color Image  -->")
  sysargv2  = input("Enter the Luminance image to be created  -->")
  img = cv2.imread(sysargv1)

  # createLuminance not percieved
  grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.imwrite(sysargv2, grayscale_img)
  return sysargv1
  menue()

def align2img():

  # Load the two images
  sysargv1  = input("Enter the reference image -->")
  sysargv2  = input("Enter the image to be aligned -->")
  sysargv3  = input("Enter the aligned image file name -->")
  img1 = cv2.imread( sysargv2 )
  img2 = cv2.imread( sysargv1 )

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
  return sysargv1
  menue()

def plotto3d16(sysargv2):
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

def PNGcreateimage16(sysargv2, sysargv3, sysargv4, sysargv5):
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
  blur = cv2.blur(image_1,(3,3)) 
  pyplot.imshow(blur,cmap='gray')
  # show the figure
  pyplot.show()


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

def menue(sysargv1):
  sysargv1 = input("Enter >>1<< AffineTransform or >>2<< Mask an image  >>3<< Mask Invert >>4<< Add2images  >>5<< Split tricolor  >>6<< Combine Tricolor  >>7<< Create Luminance  >>8<< Align2img  >>9<< Plot 16-bit image to 3d graph >>10<< Centroid custom filter >>1313<< Exit -->")
  return sysargv1

sysargv1 = ''
while not sysargv1 == '1313':  # Substitute for a while-True-break loop.
  sysargv1 = ''
  sysargv1 = menue(sysargv1)

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

  if sysargv1 == '1313':
    sys.exit()

  if sysargv1 == '1':
    sysargv1  = input("Enter the original mask file name -->")
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
