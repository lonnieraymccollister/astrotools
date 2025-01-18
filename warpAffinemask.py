# import required libraries
import fnmatch
from PIL import Image
import cv2, sys, os, shutil, ffmpeg, tifffile
import numpy as np
import matplotlib
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits
import glob
from skimage.exposure import match_histograms


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

  image = cv2.imread(sysargv1, -1)
  mask = cv2.imread(sysargv3, -1)

  # Apply the mask to the image
  masked_image = cv2.bitwise_and(image, mask)
  cv2.imwrite(sysargv4, masked_image)
  return sysargv1
  menue()

def maskinvert():
  sysargv1  = input("Enter the mask Image  -->")
  #sysargv2  = input("Enter the Image2  -->")
  sysargv3  = input("Enter the output invert Mask -->")
  image = cv2.imread(sysargv1, -1)
    # Apply the inverted mask to the image
  masked_image = cv2.bitwise_not(image)
  cv2.imwrite(sysargv3, masked_image)
  return sysargv1
  menue()

def filecount():
  sysargv1  = input("Enter the  directory path from explorer  -->")
  sysargv2  = input("Enter file type as (*.fit)  -->")
  count = len(fnmatch.filter(os.listdir(sysargv1), sysargv2))
  print('File Count:', count)
  return sysargv1
  menue()

def resize():
  sysargv1  = input("Enter the Image to be resized(bicubic)(16bit uses .tif)  -->")
  sysargv2  = input("Enter the scale(num)(1,2,3, or 4)   -->")
  sysargv2a = input("Enter the scale(denom)(1,2,3, or 4)   -->")
  sysargv3  = input("Enter the filename of the resized image to be saved(16bit uses .tif)  -->")    
  image = tifffile.imread(sysargv1)      
  img = cv2.resize(image,None,fx=int(sysargv2) / int(sysargv2a),fy=int(sysargv2) / int(sysargv2a),interpolation=cv2.INTER_LANCZOS4)
  tifffile.imwrite("INTER_LANCZOS4" + sysargv3, img)
  return sysargv1
  menue()

def add2imagesx():
  sysargv1  = input("Enter the first masked Image  -->")
  sysargv3  = input("Enter the second masked Image  -->")
  sysargv4  = input("Enter the filename of the added images to save  -->")
  sysargv5  = input("Adjusts the brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts the contrast by scaling the pixel values 1st img(numerator example 1)   -->")
  sysargv7  = input("Adjusts the contrast by scaling the pixel values 1st img(denominator example 1)  -->")

  image2 = cv2.imread(sysargv1, -1)
  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding 10 to each pixel value 
  brightness = int(sysargv5) 
  # Adjusts the contrast by scaling the pixel values by 2.3 
  contrast = int(sysargv6) / int(sysargv7)
  image = cv2.addWeighted(image2, contrast, np.zeros(image2.shape, image2.dtype), 0, brightness) 

  sysargv5  = input("Adjusts the brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts the contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
  sysargv7  = input("Adjusts the contrast by scaling the pixel values 2nd img(denominator example 1)  -->")

  mask2 = cv2.imread(sysargv3, -1)
  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding 10 to each pixel value 
  brightness = int(sysargv5) 
  # Adjusts the contrast by scaling the pixel values by 2.3 
  contrast = int(sysargv6) / int(sysargv7)  
  mask = cv2.addWeighted(mask2, contrast, np.zeros(mask2.shape, mask2.dtype), 0, brightness) 

  # Apply the mask to the image 
  masked_image = cv2.add(image, mask)
  cv2.imwrite(sysargv4, masked_image)
  return sysargv1
  menue()

def subtract2imagesx():
  sysargv1  = input("Enter the first masked Image  -->")
  sysargv3  = input("Enter the second masked Image  -->")
  sysargv4  = input("Enter the filename of the added images to save  -->")
  sysargv5  = input("Adjusts the brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts the contrast by scaling the pixel values 1st img(numerator example 1)   -->")
  sysargv7  = input("Adjusts the contrast by scaling the pixel values 1st img(denominator example 1)  -->")

  image2 = cv2.imread(sysargv1, -1)
  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding 10 to each pixel value 
  brightness = int(sysargv5) 
  # Adjusts the contrast by scaling the pixel values by 2.3 
  contrast = int(sysargv6) / int(sysargv7)
  image = cv2.addWeighted(image2, contrast, np.zeros(image2.shape, image2.dtype), 0, brightness) 

  sysargv5  = input("Adjusts the brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts the contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
  sysargv7  = input("Adjusts the contrast by scaling the pixel values 2nd img(denominator example 1)  -->")

  mask2 = cv2.imread(sysargv3, -1)
  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding 10 to each pixel value 
  brightness = int(sysargv5) 
  # Adjusts the contrast by scaling the pixel values by 2.3 
  contrast = int(sysargv6) / int(sysargv7)  
  mask = cv2.addWeighted(mask2, contrast, np.zeros(mask2.shape, mask2.dtype), 0, brightness) 

  # Apply the mask to the image
  masked_image = cv2.subtract(image, mask)
  cv2.imwrite(sysargv4, masked_image)
  return sysargv1
  menue()

def multiply2images():
  sysargv1  = input("Enter the first masked Image  -->")
  sysargv3  = input("Enter the second masked Image  -->")
  sysargv4  = input("Enter the filename of the added images to save  -->")
  sysargv5  = input("Adjusts the brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts the contrast by scaling the pixel values 1st img(numerator example 1)   -->")
  sysargv7  = input("Adjusts the contrast by scaling the pixel values 1st img(denominator example 1)  -->")

  image2 = cv2.imread(sysargv1, -1)
  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding 10 to each pixel value 
  brightness = int(sysargv5) 
  # Adjusts the contrast by scaling the pixel values by 2.3 
  contrast = int(sysargv6) / int(sysargv7)
  image = cv2.addWeighted(image2, contrast, np.zeros(image2.shape, image2.dtype), 0, brightness) 

  sysargv5  = input("Adjusts the brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts the contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
  sysargv7  = input("Adjusts the contrast by scaling the pixel values 2nd img(denominator example 1)  -->")

  mask2 = cv2.imread(sysargv3, -1)
  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding 10 to each pixel value 
  brightness = int(sysargv5) 
  # Adjusts the contrast by scaling the pixel values by 2.3 
  contrast = int(sysargv6) / int(sysargv7)  
  mask = cv2.addWeighted(mask2, contrast, np.zeros(mask2.shape, mask2.dtype), 0, brightness) 

  # Apply the mask to the image
  masked_image = cv2.multiply(image, mask)
  cv2.imwrite(sysargv4, masked_image)
  return sysargv1
  menue()

def divide2images():
  sysargv1  = input("Enter the first masked Image  -->")
  sysargv3  = input("Enter the second masked Image  -->")
  sysargv4  = input("Enter the filename of the added images to save  -->")
  sysargv5  = input("Adjusts the brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts the contrast by scaling the pixel values 1st img(numerator example 1)   -->")
  sysargv7  = input("Adjusts the contrast by scaling the pixel values 1st img(denominator example 1)  -->")

  image2 = cv2.imread(sysargv1, -1)
  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding 10 to each pixel value 
  brightness = int(sysargv5) 
  # Adjusts the contrast by scaling the pixel values by 2.3 
  contrast = int(sysargv6) / int(sysargv7)
  image = cv2.addWeighted(image2, contrast, np.zeros(image2.shape, image2.dtype), 0, brightness) 

  sysargv5  = input("Adjusts the brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts the contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
  sysargv7  = input("Adjusts the contrast by scaling the pixel values 2nd img(denominator example 1)  -->")

  mask2 = cv2.imread(sysargv3, -1)
  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding 10 to each pixel value 
  brightness = int(sysargv5) 
  # Adjusts the contrast by scaling the pixel values by 2.3 
  contrast = int(sysargv6) / int(sysargv7)  
  mask = cv2.addWeighted(mask2, contrast, np.zeros(mask2.shape, mask2.dtype), 0, brightness) 

  # Apply the mask to the image
  masked_image = cv2.divide(image, mask)
  cv2.imwrite(sysargv4, masked_image)
  return sysargv1
  menue()

def max2images():
  sysargv1  = input("Enter the first masked Image  -->")
  sysargv3  = input("Enter the second masked Image  -->")
  sysargv4  = input("Enter the filename of the added images to save  -->")
  sysargv5  = input("Adjusts the brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts the contrast by scaling the pixel values 1st img(numerator example 1)   -->")
  sysargv7  = input("Adjusts the contrast by scaling the pixel values 1st img(denominator example 1)  -->")

  image2 = cv2.imread(sysargv1, -1)
  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding 10 to each pixel value 
  brightness = int(sysargv5) 
  # Adjusts the contrast by scaling the pixel values by 2.3 
  contrast = int(sysargv6) / int(sysargv7)
  image = cv2.addWeighted(image2, contrast, np.zeros(image2.shape, image2.dtype), 0, brightness) 

  sysargv5  = input("Adjusts the brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts the contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
  sysargv7  = input("Adjusts the contrast by scaling the pixel values 2nd img(denominator example 1)  -->")

  mask2 = cv2.imread(sysargv3, -1)
  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding 10 to each pixel value 
  brightness = int(sysargv5) 
  # Adjusts the contrast by scaling the pixel values by 2.3 
  contrast = int(sysargv6) / int(sysargv7)  
  mask = cv2.addWeighted(mask2, contrast, np.zeros(mask2.shape, mask2.dtype), 0, brightness) 

  # Apply the mask to the image
  masked_image = cv2.max(image, mask)
  cv2.imwrite(sysargv4, masked_image)
  return sysargv1
  menue()

def min2images():
  sysargv1  = input("Enter the first masked Image  -->")
  sysargv3  = input("Enter the second masked Image  -->")
  sysargv4  = input("Enter the filename of the added images to save  -->")
  sysargv5  = input("Adjusts the brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts the contrast by scaling the pixel values 1st img(numerator example 1)   -->")
  sysargv7  = input("Adjusts the contrast by scaling the pixel values 1st img(denominator example 1)  -->")

  image2 = cv2.imread(sysargv1, -1)
  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding 10 to each pixel value 
  brightness = int(sysargv5) 
  # Adjusts the contrast by scaling the pixel values by 2.3 
  contrast = int(sysargv6) / int(sysargv7)
  image = cv2.addWeighted(image2, contrast, np.zeros(image2.shape, image2.dtype), 0, brightness) 

  sysargv5  = input("Adjusts the brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts the contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
  sysargv7  = input("Adjusts the contrast by scaling the pixel values 2nd img(denominator example 1)  -->")

  mask2 = cv2.imread(sysargv3, -1)
  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding 10 to each pixel value 
  brightness = int(sysargv5) 
  # Adjusts the contrast by scaling the pixel values by 2.3 
  contrast = int(sysargv6) / int(sysargv7)  
  mask = cv2.addWeighted(mask2, contrast, np.zeros(mask2.shape, mask2.dtype), 0, brightness) 

  # Apply the mask to the image
  masked_image = cv2.min(image, mask)
  cv2.imwrite(sysargv4, masked_image)
  return sysargv1
  menue()

def splittricolor():
  sysargv2  = input("Enter the Color Image to be split  -->")
  sysargv7  = input("Enter 0 for fits or 1 for other file -->")

  if sysargv7 == '0':

    # Function to read FITS file and return data
    def read_fits(file):
        hdul = fits.open(file)
        header = hdul[0].header
        data = hdul[0].data
        hdul.close()
        return data, header

    # Read the FITS files
    file1 = sysargv2

    # Read the image data from the FITS file
    image_data, header = read_fits(file1)

    # Split the color image into its individual channels
    #b, g, r = cv2.split(image_data)
    b, g, r = np.split(image_data, image_data.shape[0], axis=0)

    # Save each channel as a separate file
    fits.writeto(f'channel_0_64bit.fits', b, header, overwrite=True)
    fits.writeto(f'channel_1_64bit.fits', g, header, overwrite=True)
    fits.writeto(f'channel_2_64bit.fits', r, header, overwrite=True)

  if sysargv7 == '1':

    img = cv2.imread(sysargv2, -1)

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
  sysargv2  = input("Enter the Green image to be combined  -->")
  sysargv3  = input("Enter the Red image to be combined  -->")
  sysargv4  = input("Enter the RGB file to be created  -->")
  sysargv7  = input("Enter 0 for fits or 1 for other file -->")

  if sysargv7 == '0':
 

    with fits.open(sysargv1) as old_hdul:
        # Access the header of the primary HDU
      old_header = old_hdul[0].header
      old_data = old_hdul[0].data
    
    # Function to read FITS file and return data
    def read_fits(file):
      with fits.open(file, mode='update') as hdul:#
        data = hdul[0].data
        # hdul.close()
      return data

    # Read the FITS files
    file1 = sysargv1
    file2 = sysargv2
    file3 = sysargv3

    # Read the image data from the FITS file
    blue = read_fits(file1)
    green = read_fits(file2)
    red = read_fits(file3)

    # Check dimensions
    print("Data1 shape:", blue.shape)
    print("Data2 shape:", green.shape)
    print("Data3 shape:", red.shape)

    #newRGBImage = cv2.merge((red,green,blue))
    RGB_Image1 = np.stack((red,green,blue))

    # Remove the extra dimension
    RGB_Image = np.squeeze(RGB_Image1)

    # Create a FITS header with NAXIS = 3
    header = old_header
    header['NAXIS'] = 3
    header['NAXIS1'] = RGB_Image.shape[2]
    header['NAXIS2'] = RGB_Image.shape[1]
    header['NAXIS3'] = RGB_Image.shape[0]

    # Ensure the data type is correct 
    newRGB_Image = RGB_Image.astype(np.float64)

    print("newRGB_Image shape:", newRGB_Image.shape)

    fits.writeto( sysargv4, newRGB_Image, overwrite=True)
    # Save the RGB image as a new FITS file with the correct header
    hdu = fits.PrimaryHDU(data=newRGB_Image, header=header)
    hdu.writeto(sysargv4, overwrite=True)

    # Function to read and verify the saved FITS file
    def verify_fits(sysargv4):
      with fits.open(sysargv4) as hdul:
        data = hdul[0].data
    return data

    # Verify the saved RGB image
    verified_image = verify_fits(sysargv4)
    print("Verified image shape:", verified_image.shape)

  if sysargv7 == '1':

    sysargv1  = input("Enter the Blue image to be combined  -->")
    blue = cv2.imread(sysargv1, -1)
    sysargv2  = input("Enter the Green image to be combined  -->")
    green = cv2.imread(sysargv2, -1)
    sysargv3  = input("Enter the Red image to be combined  -->")
    red = cv2.imread(sysargv3, -1)
    sysargv4  = input("Enter the RGB file to be created  -->")

    # Merge the Blue, Green and Red color channels
    newRGBImage = cv2.merge((red,green,blue))
    cv2.imwrite(sysargv4, newRGBImage)
    return sysargv1

  return sysargv1
  menue()

def createLuminance():
  sysargv1  = input("Enter the Color Image  -->")
  sysargv2  = input("Enter the Luminance image to be created  -->")
  img = cv2.imread(sysargv1, -1)

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
  img1 = cv2.imread( sysargv2, -1 )
  img2 = cv2.imread( sysargv1, -1 )

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
  rescaled = (65535.0 / my_data.max() * (my_data - my_data.min())).astype(np.float64)
  im = Image.fromarray(rescaled)
  symfile = (sysargv2+"_"+sysargv4+"_"+sysargv5+".png")
  im.save(symfile)
  image_1 = imread(symfile)
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

def unsharpMask():
  sysargv1  = input("Enter the Color Image  -->")
  sysargv2  = input("Enter the unsharpMask image to be created  -->")
  image = cv2.imread(sysargv1, -1)
  gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
  unsharp_image = cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)
  cv2.imwrite( sysargv2, unsharp_image)
  return sysargv1
  menue()

def DynamicRescale16():
  sysargv1  = input("Enter the grayscale image(fits Siril)  -->")
  sysargv2  = input("Enter the the width of square(5)  -->")
  sysargv4  = input("Enter the image width in pixels(1000)  -->")
  sysargv3  = input("Enter the image height in pixels(1000)  -->")
  sysargv5  = input("Enter the final image name progrm will output a .fit file   -->") 
  sysargv6  = input("Enter the bin value   -->") 
  gamma     = float(input("Enter gamma(.3981) for 1 magnitude  -->"))
  # Replace 'your_fits_file.fits' with the actual path to your FITS file
  fits_image_filename = sysargv1
  # Open the FITS file
  with fits.open(fits_image_filename) as hdul:
      # Access the primary HDU (extension 0)
      image_data = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  print(image_data.shape)
  print(image_data.dtype.name)

  my_data = (image_data * 65535)
  img = (image_data * 65535)
  #make the Dynamic square loops
  for xw in range(0, int(sysargv3), int(sysargv2)):
    for yh in range(0, int(sysargv4), int(sysargv2)): 
      my_data1 = np.zeros((int(sysargv2), int(sysargv2)))
      for (x) in range(int(sysargv2)):
        for (y) in range(int(sysargv2)):
          my_data1[x,y]=img[(x+xw),(y+yh)]
      #Rescale to 0-65535 and convert to uint16
      rescaled1 = ((my_data1.max()+1) * ((my_data1+1) - my_data1.min()) / 65535.0).astype(np.float64)
      rescaled = (np.round(rescaled1))
      my_data[xw:(xw+int(sysargv2)), yh:(yh+int(sysargv2))] = rescaled
  
  for gamma in [float(gamma)]: 
    # Apply gamma correction. 
    gamma_corrected1 = np.array(65535.0 *(my_data / 65535) ** gamma, dtype = 'float64') 
    gamma_corrected = (np.round(gamma_corrected1))
  #cv2.imwrite(str(sysargv5)+'.tif', gamma_corrected)  

  ##hdu = fits.PrimaryHDU(gamma_corrected)
  # Create an HDU list and add the primary HDU
  ##hdulist = fits.HDUList([hdu])
  # Specify the output FITS file path
  ##output_fits_filename = sysargv5
  # Write the HDU list to the FITS file
  ##hdulist.writeto(str(sysargv5)+'gamma_corrected_drs.fit', overwrite=True)
  
  
  img_array = np.asarray(gamma_corrected / 6553500, dtype = 'float64')
  bin_factor = int(sysargv6) 
  # Get image dimensions
  height, width = img_array.shape

  # Calculate new dimensions
  new_height = height // bin_factor
  new_width = width // bin_factor

  # Bin the image using summation
  binned_image = np.zeros((new_height, new_width), dtype=img_array.dtype)
  for y in range(new_height):
    for x in range(new_width):
      # Sum pixel values within the bin
      binned_image[y, x] = np.sum(img_array[y*bin_factor:(y+1)*bin_factor, x*bin_factor:(x+1)*bin_factor])

  hdu = fits.PrimaryHDU(binned_image)
  # Create an HDU list and add the primary HDU
  hdulist = fits.HDUList([hdu])
  # Specify the output FITS file path
  output_fits_filename = sysargv5
  # Write the HDU list to the FITS file
  hdulist.writeto(str(sysargv5)+'_binned_gamma_corrected_drs.fit', overwrite=True)

  return sysargv1
  menue()

def DynamicRescale16RGB():
  sysargv1  = input("Enter the B image(fits Siril)  -->")
  sysargv1a  = input("Enter the G image(fits Siril)  -->")
  sysargv1b  = input("Enter the R image(fits Siril)  -->")
  sysargv2  = input("Enter the the width of square(5)  -->")
  sysargv4  = input("Enter the image width in pixels(1000)  -->")
  sysargv3  = input("Enter the image height in pixels(1000)  -->")
  sysargv5  = input("Enter the final image name progrm will output a .fit file   -->") 
  sysargv6  = input("Enter the bin value   -->") 
  gamma     = float(input("Enter gamma(.3981) for 1 magnitude  -->"))
#################################################################################
  # Replace 'your_fits_file.fits' with the actual path to your FITS file
  fits_image_filename = sysargv1
  # Open the FITS file
  with fits.open(fits_image_filename) as hdul:
      # Access the primary HDU (extension 0)
      image_data = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  print(image_data.shape)
  print(image_data.dtype.name)

  my_data = (image_data * 65535)
  img = (image_data * 65535)
  #make the Dynamic square loops
  for xw in range(0, int(sysargv3), int(sysargv2)):
    for yh in range(0, int(sysargv4), int(sysargv2)): 
      my_data1 = np.zeros((int(sysargv2), int(sysargv2)))
      for (x) in range(int(sysargv2)):
        for (y) in range(int(sysargv2)):
          my_data1[x,y]=img[(x+xw),(y+yh)]
      #Rescale to 0-65535 and convert to uint16
      rescaled1 = ((my_data1.max()+1) * ((my_data1+1) - my_data1.min()) / 65535.0).astype(np.float64)
      rescaled = (np.round(rescaled1))
      my_data[xw:(xw+int(sysargv2)), yh:(yh+int(sysargv2))] = rescaled
  
  for gamma in [float(gamma)]: 
    # Apply gamma correction. 
    gamma_corrected1 = np.array(65535.0 *(my_data / 65535) ** gamma, dtype = 'float64') 
    gamma_corrected = (np.round(gamma_corrected1))
  #cv2.imwrite(str(sysargv5)+'.tif', gamma_corrected)  

  ##hdu = fits.PrimaryHDU(gamma_corrected)
  # Create an HDU list and add the primary HDU
  ##hdulist = fits.HDUList([hdu])
  # Specify the output FITS file path
  ##output_fits_filename = sysargv5
  # Write the HDU list to the FITS file
  ##hdulist.writeto(str(sysargv5)+'gamma_corrected_drs.fit', overwrite=True)
  
  
  img_array = np.asarray(gamma_corrected / 6553500, dtype = 'float64')
  bin_factor = int(sysargv6) 
  # Get image dimensions
  height, width = img_array.shape

  # Calculate new dimensions
  new_height = height // bin_factor
  new_width = width // bin_factor

  # Bin the image using summation
  binned_image = np.zeros((new_height, new_width), dtype=img_array.dtype)
  for y in range(new_height):
    for x in range(new_width):
      # Sum pixel values within the bin
      binned_image[y, x] = np.sum(img_array[y*bin_factor:(y+1)*bin_factor, x*bin_factor:(x+1)*bin_factor])

  hdu = fits.PrimaryHDU(binned_image)
  # Create an HDU list and add the primary HDU
  hdulist = fits.HDUList([hdu])
  # Specify the output FITS file path
  output_fits_filename = sysargv5
  # Write the HDU list to the FITS file
  hdulist.writeto(str(sysargv5)+'_binned_gamma_corrected_drs_B.fit', overwrite=True)

###########################################################################################
#################################################################################
  # Replace 'your_fits_file.fits' with the actual path to your FITS file
  fits_image_filename = sysargv1a
  # Open the FITS file
  with fits.open(fits_image_filename) as hdul:
      # Access the primary HDU (extension 0)
      image_data = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  print(image_data.shape)
  print(image_data.dtype.name)

  my_data = (image_data * 65535)
  img = (image_data * 65535)
  #make the Dynamic square loops
  for xw in range(0, int(sysargv3), int(sysargv2)):
    for yh in range(0, int(sysargv4), int(sysargv2)): 
      my_data1 = np.zeros((int(sysargv2), int(sysargv2)))
      for (x) in range(int(sysargv2)):
        for (y) in range(int(sysargv2)):
          my_data1[x,y]=img[(x+xw),(y+yh)]
      #Rescale to 0-65535 and convert to uint16
      rescaled1 = ((my_data1.max()+1) * ((my_data1+1) - my_data1.min()) / 65535.0).astype(np.float64)
      rescaled = (np.round(rescaled1))
      my_data[xw:(xw+int(sysargv2)), yh:(yh+int(sysargv2))] = rescaled
  
  for gamma in [float(gamma)]: 
    # Apply gamma correction. 
    gamma_corrected1 = np.array(65535.0 *(my_data / 65535) ** gamma, dtype = 'float64') 
    gamma_corrected = (np.round(gamma_corrected1))
  #cv2.imwrite(str(sysargv5)+'.tif', gamma_corrected)  

  ##hdu = fits.PrimaryHDU(gamma_corrected)
  # Create an HDU list and add the primary HDU
  ##hdulist = fits.HDUList([hdu])
  # Specify the output FITS file path
  ##output_fits_filename = sysargv5
  # Write the HDU list to the FITS file
  ##hdulist.writeto(str(sysargv5)+'gamma_corrected_drs.fit', overwrite=True)
  
  
  img_array = np.asarray(gamma_corrected / 6553500, dtype = 'float64')
  bin_factor = int(sysargv6) 
  # Get image dimensions
  height, width = img_array.shape

  # Calculate new dimensions
  new_height = height // bin_factor
  new_width = width // bin_factor

  # Bin the image using summation
  binned_image = np.zeros((new_height, new_width), dtype=img_array.dtype)
  for y in range(new_height):
    for x in range(new_width):
      # Sum pixel values within the bin
      binned_image[y, x] = np.sum(img_array[y*bin_factor:(y+1)*bin_factor, x*bin_factor:(x+1)*bin_factor])

  hdu = fits.PrimaryHDU(binned_image)
  # Create an HDU list and add the primary HDU
  hdulist = fits.HDUList([hdu])
  # Specify the output FITS file path
  output_fits_filename = sysargv5
  # Write the HDU list to the FITS file
  hdulist.writeto(str(sysargv5)+'_binned_gamma_corrected_drs_G.fit', overwrite=True)

###########################################################################################
#################################################################################
  # Replace 'your_fits_file.fits' with the actual path to your FITS file
  fits_image_filename = sysargv1b
  # Open the FITS file
  with fits.open(fits_image_filename) as hdul:
      # Access the primary HDU (extension 0)
      image_data = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  print(image_data.shape)
  print(image_data.dtype.name)

  my_data = (image_data * 65535)
  img = (image_data * 65535)
  #make the Dynamic square loops
  for xw in range(0, int(sysargv3), int(sysargv2)):
    for yh in range(0, int(sysargv4), int(sysargv2)): 
      my_data1 = np.zeros((int(sysargv2), int(sysargv2)))
      for (x) in range(int(sysargv2)):
        for (y) in range(int(sysargv2)):
          my_data1[x,y]=img[(x+xw),(y+yh)]
      #Rescale to 0-65535 and convert to uint16
      rescaled1 = ((my_data1.max()+1) * ((my_data1+1) - my_data1.min()) / 65535.0).astype(np.float64)
      rescaled = (np.round(rescaled1))
      my_data[xw:(xw+int(sysargv2)), yh:(yh+int(sysargv2))] = rescaled
  
  for gamma in [float(gamma)]: 
    # Apply gamma correction. 
    gamma_corrected1 = np.array(65535.0 *(my_data / 65535) ** gamma, dtype = 'float64') 
    gamma_corrected = (np.round(gamma_corrected1))
  #cv2.imwrite(str(sysargv5)+'.tif', gamma_corrected)  

  ##hdu = fits.PrimaryHDU(gamma_corrected)
  # Create an HDU list and add the primary HDU
  ##hdulist = fits.HDUList([hdu])
  # Specify the output FITS file path
  ##output_fits_filename = sysargv5
  # Write the HDU list to the FITS file
  ##hdulist.writeto(str(sysargv5)+'gamma_corrected_drs.fit', overwrite=True)
  
  
  img_array = np.asarray(gamma_corrected / 6553500, dtype = 'float64')
  bin_factor = int(sysargv6) 
  # Get image dimensions
  height, width = img_array.shape

  # Calculate new dimensions
  new_height = height // bin_factor
  new_width = width // bin_factor

  # Bin the image using summation
  binned_image = np.zeros((new_height, new_width), dtype=img_array.dtype)
  for y in range(new_height):
    for x in range(new_width):
      # Sum pixel values within the bin
      binned_image[y, x] = np.sum(img_array[y*bin_factor:(y+1)*bin_factor, x*bin_factor:(x+1)*bin_factor])

  hdu = fits.PrimaryHDU(binned_image)
  # Create an HDU list and add the primary HDU
  hdulist = fits.HDUList([hdu])
  # Specify the output FITS file path
  output_fits_filename = sysargv5
  # Write the HDU list to the FITS file
  hdulist.writeto(str(sysargv5)+'_binned_gamma_corrected_drs_R.fit', overwrite=True)

###########################################################################################


  return sysargv1
  menue()

def gaussian():
  sysargv1  = input("Enter the Color Image(.png or .tif)  -->")
  sysargv2  = input("Enter the Difference image to be created(.png or .tif)  -->")
  sysargv4a  = input("Enter the gausian blur 3, 5, or 7  -->")
  sysargv4 = float(sysargv4a)
  image1 = cv2.imread(sysargv1,cv2.IMREAD_ANYDEPTH)
  image = image1.astype(np.float32)
  selfavg = np.array((cv2.GaussianBlur(image, (0, 0), sysargv4)), dtype='float32')
  unsharp_image1 = selfavg.astype(np.uint16)
  cv2.imwrite( sysargv2, unsharp_image1)
  return sysargv1
  menue()

def FFT():
  #From Python for Microscopists-Bhattiprolu, S. (2023). python_for_microscopists. GitHub. https://github.com/bnsreenu/python_for_microscopists/blob/master/330_Detectron2_Instance_3D_EM_Platelet.ipynb
  sysargv1  = input("Enter the Greyscale Image  -->")
  sysargv2  = input("Enter the FFT HP image to be created  -->")
  img = cv2.imread(sysargv1, 0) # load an image

  #Output is a 2D complex array. 1st channel real and 2nd imaginary
  #For fft in opencv input image needs to be converted to float32
  dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

  #Rearranges a Fourier transform X by shifting the zero-frequency 
  #component to the center of the array.
  #Otherwise it starts at the tope left corenr of the image (array)
  dft_shift = np.fft.fftshift(dft)

  ##Magnitude of the function is 20.log(abs(f))
  #For values that are 0 we may end up with indeterminate values for log. 
  #So we can add 1 to the array to avoid seeing a warning. 
  magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))


  # Circular HPF mask, center circle is 0, remaining all ones
  #Can be used for edge detection because low frequencies at center are blocked
  #and only high frequencies are allowed. Edges are high frequency components.
  #Amplifies noise.

  rows, cols = img.shape
  crow, ccol = int(rows / 2), int(cols / 2)

  mask = np.ones((rows, cols, 2), np.uint8)
  r = 80
  center = [crow, ccol]
  x, y = np.ogrid[:rows, :cols]
  mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
  mask[mask_area] = 0


  # Circular LPF mask, center circle is 1, remaining all zeros
  # Only allows low frequency components - smooth regions
  #Can smooth out noise but blurs edges.
  #
  
  rows, cols = img.shape
  crow, ccol = int(rows / 2), int(cols / 2)

  mask = np.zeros((rows, cols, 2), np.uint8)
  r = 100
  center = [crow, ccol]
  x, y = np.ogrid[:rows, :cols]
  mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
  mask[mask_area] = 1

  # Band Pass Filter - Concentric circle mask, only the points living in concentric circle are ones
  rows, cols = img.shape
  crow, ccol = int(rows / 2), int(cols / 2)

  mask = np.zeros((rows, cols, 2), np.uint8)
  r_out = 80
  r_in = 10
  center = [crow, ccol]
  x, y = np.ogrid[:rows, :cols]
  mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                             ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
  mask[mask_area] = 1
  


  # apply mask and inverse DFT
  fshift = dft_shift * mask

  fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

  f_ishift = np.fft.ifftshift(fshift)
  img_back = cv2.idft(f_ishift)
  img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

  fig = plt.figure(figsize=(12, 12))
  ax1 = fig.add_subplot(2,2,1)
  ax1.imshow(img, cmap='gray')
  ax1.title.set_text('Input Image')
  ax2 = fig.add_subplot(2,2,2)
  ax2.imshow(magnitude_spectrum, cmap='gray')
  ax2.title.set_text('FFT of image')
  ax3 = fig.add_subplot(2,2,3)
  ax3.imshow(fshift_mask_mag, cmap='gray')
  ax3.title.set_text('FFT + Mask')
  ax4 = fig.add_subplot(2,2,4)
  ax4.imshow(img_back, cmap='gray')
  ax4.title.set_text('After inverse FFT')
  plt.show()
  img = cv2.convertScaleAbs(img_back, alpha=(255.0))
  #cv2.imwrite( sysargv2, img_back)
  matplotlib.pyplot.imsave(sysargv2, img_back, cmap='gray')
   
  return sysargv1
  menue()

from scipy.signal import convolve2d as conv2
import skimage as skimage
from skimage import color, data, restoration
from skimage import io, img_as_ubyte

def  LrDeconv():
  sysargv1  = input("Enter the Image name  -->")
  sysargv2  = input("Enter the psf image name  -->")
  sysargv3  = input("Enter the name of deconvoluted image name  -->")
  sysargv4  = input("Enter the number of iterations for deconvolions image name  -->")
  astro = color.rgb2gray(io.imread(sysargv1))
  psf = color.rgb2gray(io.imread(sysargv2))
  
  astro = conv2(astro, psf, 'same')
  # Restore Image using Richardson-Lucy algorithm
  deconvolved_RL = restoration.richardson_lucy(astro, psf, num_iter=(int(sysargv4)))
  deconvolved_RL1 = skimage.img_as_uint(deconvolved_RL)
  io.imsave(sysargv3, deconvolved_RL1)
  plt.imshow(deconvolved_RL, cmap='gray', vmin=deconvolved_RL.min(), vmax=deconvolved_RL.max())
  plt.show()
  
  return sysargv1
  menue()

def erosion():
  sysargv2  = input("Enter the input file name --> ")
  sysargv3  = input("Enter number of iterations example 3,5,7 --> ")
  sysargv4  = input("Enter (Kernel)structuring element of radius example 3,5,7 --> ")

  # Read the image as grayscale
  img = (io.imread(sysargv2))
  # Define a kernel (a matrix of odd size) for the erosion operation
  # You can choose different shapes and sizes for the kernel
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ((int(sysargv4)), (int(sysargv4))))

  # Apply the erosion operation using cv2.erode()
  # You can adjust the number of iterations for more or less erosion
  img_erosion = cv2.erode(img, kernel, iterations=(int(sysargv3)))
  cv2.imshow('output', img)
  cv2.imshow('Erosion', img_erosion)

  # Wait for a key press to exit
  cv2.waitKey(0)
  # close the window 
  cv2.destroyAllWindows() 

  return sysargv1
  menue()

def jpgcomp():
  sysargv2  = input("Enter the input file name --> ")
  sysargv3  = input("Enter number percent to compress to (10) ")
  sysargv4  = input("Enter the output file name --> ")

  # Read the image as grayscale
  image = cv2.imread(sysargv2)
  cv2.imwrite(sysargv4, image, [cv2.IMWRITE_JPEG_QUALITY, int(sysargv3)])

  return sysargv1
  menue()


def dilation():
  sysargv2  = input("Enter the input file name --> ")
  sysargv3  = input("Enter number of iterations example 3,5,7 --> ")
  sysargv4  = input("Enter (Kernel)structuring element of radius example 3,5,7 --> ")

  # Create a disk-shaped structuring element of radius 5
  # Read the image as grayscale
  img = (io.imread(sysargv2))
  # Define a kernel (a matrix of odd size) for the erosion operation
  # You can choose different shapes and sizes for the kernel
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ((int(sysargv4)), (int(sysargv4))))

  # Apply the erosion operation using cv2.erode()
  # You can adjust the number of iterations for more or less erosion
  img_dilated = cv2.dilate(img, kernel, iterations=(int(sysargv3)))

  cv2.imshow('output', img)
  cv2.imshow('Dilation', img_dilated)

  # Wait for a key press to exit
  cv2.waitKey(0)
  # close the window 
  cv2.destroyAllWindows() 

  return sysargv1
  menue()

def imgcrop1():
  sysargv1  = input("Enter the Image to crop  -->")
  sysargv3  = input("Enter the filename of the image to save  -->")

  # Opens a image in RGB mode
  img = cv2.imread(sysargv1)

  # Size of the image in pixels (size of original image)
  # (This is not mandatory)
  height, width = img.shape[:2]
  print("width", width)
  print("height", height)

  # Setting the points for cropped image
  sysargv4  = input("Enter x point of new image -->")
  sysargv5  = input("Enter y point of new image   -->")
  sysargv6  = input("Enter width of new image   -->")
  sysargv7  = input("Enter height of new image  -->")
  x = int(sysargv4)
  y = int(sysargv5)

  cropped_image = img[y:y+int(height), x:x+int(width)]

  # Cropped image of above dimension
  # (It will not change original image)
  cv2.imshow('image',cropped_image)
  cv2.waitKey(0)  
  cv2.imwrite(sysargv3, cropped_image)

  return sysargv1
  menue()

def imghiststretch():
  sysargv2  = input("Enter the greyscale Image for hist  -->")
  sysargv3  = input("Enter the Image depth(256/16536)  -->")
  image = cv2.imread(sysargv2)
  image1 = cv2.imread(sysargv2)
  image_Height = image.shape[0]
  image_Width = image.shape[1]	
  histogram = np.zeros([int(sysargv3)], np.int32)
  for x in range(1, image_Height):
    for y in range(1, image_Width):
      histogram[image[x,y]] +=1

  plt.figure()
  plt.title("GrayScale Histogram")
  plt.xlabel("Intensity Level")
  plt.ylabel("Intensity Frequency")
  plt.xlim([0, int(sysargv3)])
  plt.plot(histogram)
  plt.show()

  sysargv4  = input("Enter min int value of stretch  -->")
  sysargv5  = input("Enter max int value of stretch  -->")
  sysargv6  = input("Enter the greyscale image save  -->")
  
  sysargv4a=int(sysargv4)
  sysargv5a=int(sysargv5)
  for x in range(1, image_Height):
    for y in range(1, image_Width):
      image1[x,y] = np.where(image[x,y] < sysargv4a, 0, image[x,y])
      image1[x,y] = np.where(image[x,y] > sysargv5a, 0, image[x,y])
      image1[x,y] = ((int(sysargv3)-1) / ((int(sysargv5)-1) - (int(sysargv4)-1)))*(image1[x,y]-(int(sysargv4)-1))

  img_normalized = cv2.normalize(image1, None, 0, (int(sysargv3)-1), cv2.NORM_MINMAX)
  cv2.imshow('Image', img_normalized)
  cv2.waitKey(0)  
  cv2.imwrite(sysargv6, img_normalized)

  return sysargv1
  menue()

def gif():
  sysargv1  = input("jpg only Enter Image width  -->")
  sysargv2  = input("jpg only Enter Image height  -->")
  sysargv3  = input("Enter Gif to save -->")
  sysargv4  = input("Enter (*.jpg)etc to use for Gif -->")
  sysargv5  = input("Enter  image duration in millisconds -->")
  fGIF = sysargv3
  W = int(sysargv1)
  H = int(sysargv2)
  # Create the frames
  frames = []
  images = glob.glob(sysargv4)

  for i in images:
      newImg = Image.open(i)
      frames.append(newImg)
 

  # Save into a GIF file that loops forever: duration is in milli-second
  frames[0].save(fGIF, format='GIF', optimize=False, append_images=frames[1:],
      save_all=True, duration=int(sysargv5), loop=0)

  return sysargv1
  menue()

def video():
  sysargv3  = input("Enter file name to save mp4 Video -->")
  sysargv3a = input("Enter file name to save gif Video -->")
  sysargv3b = input("Enter scale (800) pixels size to save gif Video -->")
  sysargv4  = input("Enter (*.jpg)etc to use for Video -->")
  sysargv5  = input("Enter frames per second -->")

  img_array = []
  for filename in glob.glob(sysargv4):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

  # Save into a video file duration is in fps
  out = cv2.VideoWriter(sysargv3,cv2.VideoWriter_fourcc(*'mp4v'), int(sysargv5), size)
  
  for i in range(len(img_array)):
    out.write(img_array[i])
  out.release()

  ffmpeg.input(sysargv3).filter('scale', sysargv3b, -1).output(sysargv3a).run()
  
  return sysargv1
  menue()

def alingimg():
  sysargv2  = input("jpg enter reference Image  -->")
  sysargv3  = input("jpg enter alignment Image  -->")
  sysargv4  = input("Enter Image to save -->")

  # Open the image files.
  img1_color = cv2.imread(sysargv3)  # Image to be aligned.
  img2_color = cv2.imread(sysargv2)    # Reference image.

  # Convert to grayscale.
  img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
  img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
  height, width = img2.shape

  # Create ORB detector with 5000 features.
  orb_detector = cv2.ORB_create(5000)

  # Find keypoints and descriptors.
  # The first arg is the image, second arg is the mask
  #  (which is not required in this case).
  kp1, d1 = orb_detector.detectAndCompute(img1, None)
  kp2, d2 = orb_detector.detectAndCompute(img2, None)

  # Match features between the two images.
  # We create a Brute Force matcher with 
  # Hamming distance as measurement mode.
  matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

  # Match the two sets of descriptors.
  matches = matcher.match(d1, d2)

  # Sort matches on the basis of their Hamming distance.
  matches.sort(key = lambda x: x.distance)

  # Take the top 90 % matches forward.
  matches = matches[:int(len(matches)*0.9)]
  no_of_matches = len(matches)

  # Define empty matrices of shape no_of_matches * 2.
  p1 = np.zeros((no_of_matches, 2))
  p2 = np.zeros((no_of_matches, 2))

  for i in range(len(matches)):
    p1[i, :] = kp1[matches[i].queryIdx].pt
    p2[i, :] = kp2[matches[i].trainIdx].pt

  # Find the homography matrix.
  homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

  # Use this matrix to transform the
  # colored image wrt the reference image.
  transformed_img = cv2.warpPerspective(img1_color,
                      homography, (width, height))

  # Save the output.
  cv2.imwrite(sysargv4, transformed_img)

  return sysargv1
  menue()

def gamma():
  sysargv1  = input("Enter the image(fits Siril)  -->")
  sysargv5  = input("Enter the final image name progrm will output a .fit   -->") 
  gamma     = float(input("Enter gamma(.3981) for 1 magnitude  -->"))
  # Replace 'your_fits_file.fits' with the actual path to your FITS file
  fits_image_filename = sysargv1
  # Open the FITS file
  with fits.open(fits_image_filename) as hdul:
      # Access the primary HDU (extension 0)
      image_data = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  print(image_data.shape)
  print(image_data.dtype.name)

  my_data = (image_data * 65535)
  
  for gamma in [float(gamma)]: 
    # Apply gamma correction. 
    gamma_corrected1 = gamma_corrected = np.array(((65535.0 *(my_data / 65535) ** gamma/65535)/100), dtype = 'float64') 
    #gamma_corrected = (np.round(gamma_corrected1))
  #cv2.imwrite(str(sysargv5)+'gamma_corrected'+'.tif', gamma_corrected)  

  hdu = fits.PrimaryHDU(gamma_corrected)
  # Create an HDU list and add the primary HDU
  hdulist = fits.HDUList([hdu])
  # Specify the output FITS file path
  output_fits_filename = sysargv5
  # Write the HDU list to the FITS file
  hdulist.writeto(str(sysargv5)+'gamma_corrected'+'.fit', overwrite=True)
  return sysargv1
  menue()

def add2images():
  sysargv1  = input("Enter the first (default fits Siril) Image  -->")
  sysargv3  = input("Enter the second (default fits Siril) Image  -->")
  sysargv4  = input("Enter the filename of the added images to save  -->")
  sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
  sysargv5a  = input("Adjusts Image2 brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
  sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
  sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
  sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")

  fits_image_filename = sysargv1
  # Open the FITS file
  with fits.open(fits_image_filename) as hdul:
      # Access the primary HDU (extension 0)
      image_data = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  fits_image_filename = sysargv3
  # Open the FITS file
  with fits.open(fits_image_filename) as hdul:
      # Access the primary HDU (extension 0)
      image_data1 = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding sysargv5 to each pixel value 
  brightness = int(sysargv5) 
  brightness1 = int(sysargv5a)
  # Adjusts the contrast by scaling the pixel values by contrast
  contrast = int(sysargv6) / int(sysargv7)
  contrast1 = int(sysargv8) / int(sysargv9)

  print(image_data.shape)
  print(image_data.dtype.name)

  my_data = np.array((((((image_data * 65535)*contrast)+brightness) + (((image_data1 * 65535)*contrast1)+brightness1) / 65535)/100), dtype = 'float64') 

  hdu = fits.PrimaryHDU(my_data)
  # Create an HDU list and add the primary HDU
  hdulist = fits.HDUList([hdu])
  # Specify the output FITS file path
  output_fits_filename = sysargv4
  # Write the HDU list to the FITS file
  hdulist.writeto(str(sysargv4)+'pm_add'+'.fit', overwrite=True)
  return sysargv1
  menue()

def subtract2images():
  sysargv1  = input("Enter the first (default fits Siril) Image  -->")
  sysargv3  = input("Enter the second (default fits Siril) Image  -->")
  sysargv4  = input("Enter the filename of the added images to save  -->")
  sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
  sysargv5a  = input("Adjusts Image2 brightness by adding x to each pixel value example 0   -->")
  sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
  sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
  sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
  sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")

  fits_image_filename = sysargv1
  # Open the FITS file
  with fits.open(fits_image_filename) as hdul:
      # Access the primary HDU (extension 0)
      image_data = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  fits_image_filename = sysargv3
  # Open the FITS file
  with fits.open(fits_image_filename) as hdul:
      # Access the primary HDU (extension 0)
      image_data1 = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  # Apply the mask to the image
  # Adjust the brightness and contrast 
  # Adjusts the brightness by adding sysargv5 to each pixel value 
  brightness = int(sysargv5) 
  brightness1 = int(sysargv5a)
  # Adjusts the contrast by scaling the pixel values by contrast
  contrast = int(sysargv6) / int(sysargv7)
  contrast1 = int(sysargv8) / int(sysargv9)

  print(image_data.shape)
  print(image_data.dtype.name)

  my_data = np.array(((((image_data * 65535)*contrast)+brightness) - (((image_data1 * 65535)*contrast1)+brightness1) / 65535), dtype = 'float64') 

  hdu = fits.PrimaryHDU(my_data)
  # Create an HDU list and add the primary HDU
  hdulist = fits.HDUList([hdu])
  # Specify the output FITS file path
  output_fits_filename = sysargv4
  # Write the HDU list to the FITS file
  hdulist.writeto(str(sysargv4)+'pm_sub'+'.fit', overwrite=True)
  return sysargv1
  menue()

def clahe():
  sysargv2  = input("Enter file name of color image to enter(16bit tif/png) -->")
  sysargv3  = input("Enter clip limit (3) -->")
  sysargv4  = input("Enter tile Grid Size (8) -->")
  sysargv5  = input("Enter output filename -->")

  colorimage = cv2.imread(sysargv2, -1) 
  clahe_model = cv2.createCLAHE(clipLimit=int(sysargv3), tileGridSize=(int(sysargv4),int(sysargv4)))
  colorimage_b = clahe_model.apply(colorimage[:,:,0])
  colorimage_g = clahe_model.apply(colorimage[:,:,1])
  colorimage_r = clahe_model.apply(colorimage[:,:,2])
  colorimage_clahe = np.stack((colorimage_b,colorimage_g,colorimage_r), axis=2)
  cv2.imwrite(sysargv5, colorimage_clahe)
  
  return sysargv1
  menue()

def pm_vector_line():
  sysargv2  = input("Enter file name of color image to enter -->")
  sysargv3  = input("Enter starting point(x) -->")
  sysargv4  = input("Enter starting point(y) -->")
  sysargv5  = input("Enter ending point(mas_x) -->")
  sysargv6  = input("Enter ending point(mas_y) -->")
  sysargv7  = input("Enter color b val(255) -->")
  sysargv8  = input("Enter color g val(255) -->")
  sysargv9  = input("Enter color r val(255) -->")
  sysargv10  = input("Enter thickness(1) -->")
  sysargv11  = input("Enter file name of color image to save -->")

  # Start coordinate, here (0, 0)
  # represents the top left corner of image
  start_point = (int(sysargv3), int(sysargv4))

  # End coordinate, here (int(sysargv5), int(sysargv6))
  # represents the bottom right corner of image
  end_point = (int(sysargv3) + (int(sysargv5) * -1), int(sysargv4) + (int(sysargv6) * -1))

  # Green color in BGR
  color = (int(sysargv7), int(sysargv8), int(sysargv9))

  # Line thickness of sysargv10 px
  thickness = int(sysargv10)

  
  colorimage = cv2.imread(sysargv2, -1) 
  # Using cv2.line() method
  # Draw a diagonal green line with thickness of 9 px
  image = cv2.line(colorimage, start_point, end_point, color, thickness)
  cv2.imwrite(sysargv11, image)
  
  return sysargv1
  menue()

def hist_match():
  sysargv1  = input("Enter the reference Image  -->")
  sysargv3  = input("Enter the Image  -->")
  sysargv4  = input("Enter the filename of the added images to save  -->")

  # Load example images
  reference = cv2.imread(sysargv1, -1)
  image = cv2.imread(sysargv3, -1)

  # Perform histogram matching
  matched = match_histograms(image, reference, channel_axis=-1)

  cv2.imwrite(sysargv4, matched)

  return sysargv1
  menue()

def distance():
  sysargv1  = float(input("parallax angle in milliarcseconds  -->"))

  distancepar = 1 / (sysargv1 / 1000)
  distanceltyr = 3.26 * distancepar
  print('distance parsecs', distancepar)
  print('distance light year', distanceltyr)

  return sysargv1
  menue()

def edgedetect():
  sysargv1  = input("Enter the filename of the 3x(jpg) Image  -->")
  sysargv2  = input("Enter the filename of Edge sbl/cny to save  -->")
  sysargv3  = input("Enter the lower threshold(100)  -->")
  sysargv4  = input("Enter the upper threshold(200)  -->")

  # Read the original image
  img = cv2.imread(sysargv1, -1) 

  # Convert to graycsale
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Blur the image for better edge detection
  img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
 
  # Sobel Edge Detection
  sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
  sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
  sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection 
  img_sobel_xy = cv2.merge([sobelxy, sobelxy, sobelxy])
  cv2.imwrite(sysargv2 + "sbl.jpg", img_sobel_xy)
  imgcny = cv2.Canny(img_blur, int(sysargv3), int(sysargv4))
  img_cny = cv2.merge([imgcny, imgcny, imgcny])
  cv2.imwrite(sysargv2 + "cny.jpg", img_cny)

  return sysargv1
  menue()

def mosaic():
  sysargv2  = input("Enter file name of image 1 -->")
  sysargv3  = input("Enter file name of image 2 -->")
  sysargv4  = input("Enter file name of image 3 -->")
  sysargv5  = input("Enter file name of image 4 -->")
  sysargv6  = input("Enter file name of color image to save -->")
  sysargv7  = input("Enter 0 for fits or 1 for other file -->")

  if sysargv7 == '0':

    # Function to read FITS file and return data
    def read_fits(file):
        hdul = fits.open(file)
        data = hdul[0].data
        hdul.close()
        return data

    # Read the FITS files
    file1 = sysargv2
    file2 = sysargv3
    file3 = sysargv4
    file4 = sysargv5

    data1 = read_fits(file1)
    data2 = read_fits(file2)
    data3 = read_fits(file3)
    data4 = read_fits(file4)

    # Check dimensions
    print("Data1 shape:", data1.shape)
    print("Data2 shape:", data2.shape)
    print("Data3 shape:", data3.shape)
    print("Data4 shape:", data4.shape)

    # Combine the images into a mosaic (2x2 grid)
    mosaic = np.block([[data1, data2], [data3, data4]])

    # Save the mosaic to a new FITS file
    hdu = fits.PrimaryHDU(mosaic)
    hdul = fits.HDUList([hdu])
    hdul.writeto(sysargv6, overwrite=True)

    print("Mosaic FITS file saved successfully!")

  if sysargv7 == '1':

    # Read the four images
    img1 = cv2.imread(sysargv2)
    img2 = cv2.imread(sysargv3)
    img3 = cv2.imread(sysargv4)
    img4 = cv2.imread(sysargv5)

    # Get the size of the images
    height, width, _ = img1.shape

    # Create a new image with double the width and height
    mosaic = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

    # Place the images in the mosaic
    mosaic[0:height, 0:width] = img1
    mosaic[0:height, width:width*2] = img2
    mosaic[height:height*2, 0:width] = img3
    mosaic[height:height*2, width:width*2] = img4

    # Save the mosaic
    cv2.imwrite( sysargv6, mosaic)

  return sysargv1
  menue()

def imgqtr():

  sysargv2  = input("Enter file name of image(tif) -->")
  sysargv7  = input("Enter 0 for fits or 1 for other file -->")

  if sysargv7 == '0':

    # Function to read FITS file and return data
    def read_fits(file1):
      with fits.open(file1) as hdul:
        data = hdul[0].data
        hdul.close()
        return data

    # Read the FITS files
    file1 = sysargv2

    data1 = read_fits(file1)

    # Get the dimensions of the image
    channels, height, width = data1.shape

    # Calculate the dimensions of each quarter
    quarter_width = width // 2
    quarter_height = height // 2

    # Crop the image into four quarters for each channel
    top_left = data1[:, :quarter_height, :quarter_width]
    fits.writeto('mosaic_top_left_1.fits', top_left, overwrite=True)
    top_right = data1[:, :quarter_height, quarter_width:]
    fits.writeto('mosaic_top_right_2.fits', top_right, overwrite=True)

    # Get the dimensions of the image
    channels, height, width = data1.shape

    # Calculate the dimensions of each quarter
    quarter_width = width // 2
    quarter_height = height // 2

    # Crop the image into four quarters for each channel
    bottom_left = data1[:, quarter_height:, :quarter_width]
    fits.writeto('mosaic_bottom_left_3.fits', bottom_left, overwrite=True)
    bottom_right = data1[:, quarter_height:, quarter_width:]
    fits.writeto('mosaic_bottom_right_4.fits', bottom_right, overwrite=True)

  if sysargv7 == '1':

    image = cv2.imread(sysargv2)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the dimensions of each quarter
    quarter_width = width // 2
    quarter_height = height // 2

    # Crop the image into four quarters
    top_left = image[0:quarter_height, 0:quarter_width]
    top_right = image[0:quarter_height, quarter_width:width]
    bottom_left = image[quarter_height:height, 0:quarter_width]
    bottom_right = image[quarter_height:height, quarter_width:width]

    # Save the quarters as separate images
    cv2.imwrite(sysargv2 + 'mosaictop_left.tif', top_left)
    cv2.imwrite(sysargv2 + 'mosaictop_right.tif', top_right)
    cv2.imwrite(sysargv2 + 'mosaicbottom_left.tif', bottom_left)
    cv2.imwrite(sysargv2 + 'mosaicbottom_right.tif', bottom_right)

  return sysargv1
  menue()


def menue(sysargv1):
  sysargv1 = input("Enter \n>>1<< AffineTransform(3pts) >>2<< Mask an image >>3<< Mask Invert >>4<< Add2images(fit)  \n>>5<< Split tricolor >>6<< Combine Tricolor >>7<< Create Luminance(2ax) >>8<< Align2img \n>>9<< Plot_16-bit_img to 3d graph(2ax) >>10<< Centroid_Custom_filter(2ax) >>11<< UnsharpMask \n>>12<< FFT-Bandpass(2ax) >>13<< Img-DeconvClr >>14<< Centroid_Custom_Array_loop(2ax) \n>>15<< Erosion(2ax) >>16<< Dilation(2ax) >>17<< DynamicRescale(2ax) >>18<< Gaussian  \n>>19<< DrCntByFileType >>20<< ImgResize >>21<< JpgCompress >>22<< subtract2images(fit)  \n>>23<< multiply2images >>24<< divide2images >>25<< max2images >>26<< min2images \n>>27<< imgcrop >>28<< imghiststretch >>29<< gif  >>30<< aling2img(2pts) >>31<< Video \n>>32<< gammaCor >>33<< Add2images(tif) >>34<< subtract2images(tif) >>35<< DynReStr(RGB) \n>>36<< clahe >>37<< pm_vector_line >>38<< hist_match >>39<< distance >>40<< EdgeDetect \n>>41<< Mosaic(4) >>42<< ImgQtr \n>>1313<< Exit --> ")
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

  if sysargv1 == '11':
    unsharpMask()

  if sysargv1 == '12':
    FFT()

  if sysargv1 == '13':
    LrDeconv()

  if sysargv1 == '14':
    sysargv2  = input("Enter the file name -->")
    sysargv3 = input("Enter the radius of the file-->")
    sysargv4 = input("Enter the x-coordinate of the centroid-->")
    sysargv5 = input("Enter the y-coordinate of the centroid-->")
    x = int(sysargv4)
    y = int(sysargv5)
    for i in range(3):
      i1=int(i + x - 1)
      file = str(i1)
      for j in range(3):
        j1=int(j + y - 1)
        file1 = str(j1)
        sysargv4 = file
        sysargv5 = file1
        PNGcreateimage16(sysargv2, sysargv3, sysargv4, sysargv5)
    menue(sysargv1)
 
  if sysargv1 == '15':
    erosion()
 
  if sysargv1 == '16':
    dilation()

  if sysargv1 == '17':
    DynamicRescale16()

  if sysargv1 == '18':
    gaussian()

  if sysargv1 == '19':
    filecount()

  if sysargv1 == '20':
    resize()

  if sysargv1 == '21':
    jpgcomp()

  if sysargv1 == '22':
    subtract2images()

  if sysargv1 == '23':
    multiply2images()

  if sysargv1 == '24':
    divide2images()

  if sysargv1 == '25':
    max2images()

  if sysargv1 == '26':
    min2images()

  if sysargv1 == '27':
    imgcrop1()

  if sysargv1 == '28':
    imghiststretch()

  if sysargv1 == '29':
    gif()

  if sysargv1 == '30':
    alingimg()

  if sysargv1 == '31':
    video()

  if sysargv1 == '32':
    gamma()

  if sysargv1 == '33':
    add2imagesx()

  if sysargv1 == '34':
    subtract2imagesx()

  if sysargv1 == '35':
    DynamicRescale16RGB()
  
  if sysargv1 == '36':
    clahe()

  if sysargv1 == '37':
    pm_vector_line()

  if sysargv1 == '38':
    hist_match()
  
  if sysargv1 == '39':
    distance()
  
  if sysargv1 == '40':
    edgedetect()

  if sysargv1 == '41':
    mosaic()

  if sysargv1 == '42':
    imgqtr()

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
