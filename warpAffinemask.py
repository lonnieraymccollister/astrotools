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
from astropy.wcs import WCS
import glob
from skimage.exposure import match_histograms
from scipy.ndimage import zoom
from reproject import reproject_interp, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs

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
  sysargv3  = input("Enter the output invert Mask -->")

  sysargv7  = input("Enter 0 for fits or 1 for other file -->")

  if sysargv7 == '0':

    # Function to read FITS file and return data
    with fits.open(sysargv1) as hdul:
        data = hdul[0].data.astype(np.float64)
        data_range = np.max(data) - np.min(data)
        if data_range == 0:
          normalized_data = np.zeros_like(data)  # or handle differently
        else:
          normalized_data = (data - np.min(data)) / data_range

        #Invert the data
        inverted_data = 1 - normalized_data

        #Create a new HDU with the inverted data
        hdu = fits.PrimaryHDU(inverted_data)

        #Copy the header from the original HDU
        hdu.header = hdul[0].header

        #Create a new HDU list and save it to a new FITS file
        new_hdul = fits.HDUList([hdu])
        new_hdul.writeto(sysargv3, overwrite=True)

    print("Inverted FITS file saved successfully!")

  if sysargv7 == '1':
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
  sysargv1  = input("Enter the Image to be resized(bicubic)(16bit uses .tif/.fit)  -->")
  sysargv2  = input("Enter the scale(num)(1,2,3, or 4)(***Cubic only picels, gry and (x=y)***)  -->")
  sysargv2a = input("Enter the scale(denom)(1,2,3, or 4)   -->")
  sysargv3  = input("Enter the filename of the resized image to be saved(16bit uses .tif/.fit)  -->")    
  sysargv7  = input("Enter 0 for fits cubic gry float64, 1 for fits LANCZOS4(32bit) or 2 for other file -->")

  if sysargv7 == '0':
    # Open the FITS file
    fits_file = sysargv1
    hdul = fits.open(fits_file)
    data = hdul[0].data

    # Ensure the data is in float64 format
    data = data.astype(np.float64)

    # Original dimensions of the image
    original_height, original_width = data.shape

    # Set the desired width while preserving the aspect ratio
    target_width = int(sysargv2) / int(sysargv2a)  # Example: New width in pixels
    scaling_factor = target_width / original_width

    # Apply the same scaling factor to both dimensions
    resized_data = zoom(data, (scaling_factor, scaling_factor), order=3)  # Cubic interpolation

    # Ensure the resized data is still float64
    resized_data = resized_data.astype(np.float64)

    # Save the resized image to a new FITS file
    hdu = fits.PrimaryHDU(resized_data)
    hdu.writeto( sysargv3, overwrite=True)

    print("Image resized using Cubic Interpolation (float64) and saved successfully!")


  if sysargv7 == '1':

    # Function to read FITS file and return data
    def read_fits(file):
        hdul = fits.open(file)
        header = hdul[0].header
        data = hdul[0].data
        hdul.close()
        return data, header

    # Read the FITS files
    file1 = sysargv1

    # Read the image data from the FITS file
    image_data, header = read_fits(file1)

    #image_data = np.swapaxes(image_data, 0, 2)
    #image_data = np.swapaxes(image_data, 0, 1)
    image_data = np.transpose(image_data, (1, 2, 0))

    # Normalize the image data to the range [0, 65535]
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 65535
    image_data = image_data.astype(np.uint16)

    # Convert the image to BGR format (OpenCV uses BGR by default)
    image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

    img = cv2.resize(image_bgr,None,fx=int(sysargv2) / int(sysargv2a),fy=int(sysargv2) / int(sysargv2a),interpolation=cv2.INTER_LANCZOS4)

    # Save or display the result
    image_rgb = np.transpose(img, (2, 0, 1))

    image_data = image_rgb.astype(np.float64)
    data_range = np.max(image_data) - np.min(image_data)
    if data_range == 0:
      normalized_data = np.zeros_like(image_data)  # or handle differently
    else:
      normalized_data = (image_data - np.min(image_data)) / data_range
    image_rgb = normalized_data    

    # Create a FITS HDU
    hdu = fits.PrimaryHDU(image_rgb, header)

    # Write to FITS file
    hdu.writeto(sysargv3,  overwrite=True)
 
  if sysargv7 == '2':

    image = tifffile.imread(sysargv1)      
    img = cv2.resize(image,None,fx=int(sysargv2) / int(sysargv2a),fy=int(sysargv2) / int(sysargv2a),interpolation=cv2.INTER_LANCZOS4)
    tifffile.imwrite("INTER_LANCZOS4" + sysargv3, img)
  return sysargv1
  menue()

def multiply2images():
  sysargv2  = input("Enter the first (default fits Siril) Image  -->")
  sysargv3  = input("Enter the second (default fits Siril) Image  -->")

  # Load the first FITS file
  hdul1 = fits.open(sysargv2)
  image_data1 = hdul1[0].data.astype(np.float64)
    
  # Load the second FITS file
  hdul2 = fits.open(sysargv3)
  image_data2 = hdul2[0].data.astype(np.float64)
    
  # Check if the image data from both FITS files have the same shape
  if image_data1.shape != image_data2.shape:
    print("Error: The input images do not have the same dimensions!")
    hdul1.close()
    hdul2.close()
    return
  print(image_data1.shape)
  print(image_data2.shape)
   #Add the RGB channels from both images
   #Assuming both images are in the format (height, width, 3) (RGB)
  if image_data1.ndim == 3 and image_data2.ndim == 3 :
   # Add the corresponding channels (R, G, B) of both images
    sysargv4  = input("Enter the filename of the added images to save  -->")
    sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
    sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
    sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
    sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")
    sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
    image_data1_contrastadd = int(sysargv5)
    image_data1_contrastscale = (int(sysargv6)/int(sysargv7))
    image_data2_contrastscale = (int(sysargv8)/int(sysargv9))

    result_image = ((image_data1 * image_data1_contrastscale ) * (image_data2 * image_data1_contrastscale)) + image_data1_contrastadd

    #result_image = image_data1 + image_data2

    # Create a new FITS HDU for the result
    result_hdu = fits.PrimaryHDU(result_image)
        
    # Create an HDU list (this only contains the result HDU)
    hdulist = fits.HDUList([result_hdu])
        
    # Save the result as a new FITS file
    hdulist.writeto(sysargv4, overwrite=True)
        
  else:
    print("Error: The FITS files do not appear to be in the expected RGB format.")
    
  # Close the FITS files
    hdul1.close()
    hdul2.close()

    return sysargv1
    menue()

def divide2images():
  sysargv2  = input("Enter the first (default fits Siril) Image  -->")
  sysargv3  = input("Enter the second (default fits Siril) Image  -->")

  # Load the first FITS file
  hdul1 = fits.open(sysargv2)
  image_data1 = hdul1[0].data.astype(np.float64)
    
  # Load the second FITS file
  hdul2 = fits.open(sysargv3)
  image_data2 = hdul2[0].data.astype(np.float64)
    
  # Check if the image data from both FITS files have the same shape
  if image_data1.shape != image_data2.shape:
    print("Error: The input images do not have the same dimensions!")
    hdul1.close()
    hdul2.close()
    return
  print(image_data1.shape)
  print(image_data2.shape)
   #Add the RGB channels from both images
   #Assuming both images are in the format (height, width, 3) (RGB)
  if image_data1.ndim == 3 and image_data2.ndim == 3 :
   # Add the corresponding channels (R, G, B) of both images
    sysargv4  = input("Enter the filename of the added images to save  -->")
    sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
    sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
    sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
    sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")
    sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
    image_data1_contrastadd = int(sysargv5)
    image_data1_contrastscale = (int(sysargv6)/int(sysargv7))
    image_data2_contrastscale = (int(sysargv8)/int(sysargv9))

    result_image = ((image_data1 * image_data1_contrastscale ) / (image_data2 * image_data1_contrastscale)) + image_data1_contrastadd

    #result_image = image_data1 + image_data2

    # Create a new FITS HDU for the result
    result_hdu = fits.PrimaryHDU(result_image)
        
    # Create an HDU list (this only contains the result HDU)
    hdulist = fits.HDUList([result_hdu])
        
    # Save the result as a new FITS file
    hdulist.writeto(sysargv4, overwrite=True)
        
  else:
    print("Error: The FITS files do not appear to be in the expected RGB format.")
    
  # Close the FITS files
    hdul1.close()
    hdul2.close()

    return sysargv1
    menue()

def max2images():
  sysargv2  = input("Enter the first (default fits Siril) Image  -->")
  sysargv3  = input("Enter the second (default fits Siril) Image  -->")

  # Load the first FITS file
  hdul1 = fits.open(sysargv2)
  image_data1 = hdul1[0].data.astype(np.float64)
    
  # Load the second FITS file
  hdul2 = fits.open(sysargv3)
  image_data2 = hdul2[0].data.astype(np.float64)
    
  # Check if the image data from both FITS files have the same shape
  if image_data1.shape != image_data2.shape:
    print("Error: The input images do not have the same dimensions!")
    hdul1.close()
    hdul2.close()
    return
  print(image_data1.shape)
  print(image_data2.shape)
   #Add the RGB channels from both images
   #Assuming both images are in the format (height, width, 3) (RGB)
  if image_data1.ndim == 3 and image_data2.ndim == 3 :
   # Add the corresponding channels (R, G, B) of both images
    sysargv4  = input("Enter the filename of the added images to save  -->")
    sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
    sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
    sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
    sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")
    sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
    image_data1_contrastadd = int(sysargv5)
    image_data1_contrastscale = (int(sysargv6)/int(sysargv7))
    image_data2_contrastscale = (int(sysargv8)/int(sysargv9))

    result_image = max((image_data1 * image_data1_contrastscale ), (image_data2 * image_data1_contrastscale)) + image_data1_contrastadd

    #result_image = image_data1 + image_data2

    # Create a new FITS HDU for the result
    result_hdu = fits.PrimaryHDU(result_image)
        
    # Create an HDU list (this only contains the result HDU)
    hdulist = fits.HDUList([result_hdu])
        
    # Save the result as a new FITS file
    hdulist.writeto(sysargv4, overwrite=True)
        
  else:
    print("Error: The FITS files do not appear to be in the expected RGB format.")
    
  # Close the FITS files
    hdul1.close()
    hdul2.close()

    return sysargv1
    menue()

def min2images():
  sysargv2  = input("Enter the first (default fits Siril) Image  -->")
  sysargv3  = input("Enter the second (default fits Siril) Image  -->")

  # Load the first FITS file
  hdul1 = fits.open(sysargv2)
  image_data1 = hdul1[0].data.astype(np.float64)
    
  # Load the second FITS file
  hdul2 = fits.open(sysargv3)
  image_data2 = hdul2[0].data.astype(np.float64)
    
  # Check if the image data from both FITS files have the same shape
  if image_data1.shape != image_data2.shape:
    print("Error: The input images do not have the same dimensions!")
    hdul1.close()
    hdul2.close()
    return
  print(image_data1.shape)
  print(image_data2.shape)
   #Add the RGB channels from both images
   #Assuming both images are in the format (height, width, 3) (RGB)
  if image_data1.ndim == 3 and image_data2.ndim == 3 :
   # Add the corresponding channels (R, G, B) of both images
    sysargv4  = input("Enter the filename of the added images to save  -->")
    sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
    sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
    sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
    sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")
    sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
    image_data1_contrastadd = int(sysargv5)
    image_data1_contrastscale = (int(sysargv6)/int(sysargv7))
    image_data2_contrastscale = (int(sysargv8)/int(sysargv9))

    result_image = min((image_data1 * image_data1_contrastscale ), (image_data2 * image_data1_contrastscale)) + image_data1_contrastadd

    #result_image = image_data1 + image_data2

    # Create a new FITS HDU for the result
    result_hdu = fits.PrimaryHDU(result_image)
        
    # Create an HDU list (this only contains the result HDU)
    hdulist = fits.HDUList([result_hdu])
        
    # Save the result as a new FITS file
    hdulist.writeto(sysargv4, overwrite=True)
        
  else:
    print("Error: The FITS files do not appear to be in the expected RGB format.")
    
  # Close the FITS files
    hdul1.close()
    hdul2.close()

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
    image_data = image_data.astype(np.float64)

    # Split the color image into its individual channels
    #b, g, r = cv2.split(image_data)
    b, g, r = image_data[0, :, :], image_data[1, :, :], image_data[2, :, :] 


    # Save each channel as a separate file
    fits.writeto(f'channel_0_64bit.fits', b.astype(np.float64), header, overwrite=True)
    fits.writeto(f'channel_1_64bit.fits', g.astype(np.float64), header, overwrite=True)
    fits.writeto(f'channel_2_64bit.fits', r.astype(np.float64), header, overwrite=True)

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

    blue = blue.astype(np.float64)
    green = green.astype(np.float64)
    red = red.astype(np.float64)

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
  sysargv7  = input("Enter 0 for fits or 1 for other file -->")

  if sysargv7 == '0':

    # Function to read FITS file and return data
    hdul = fits.open(sysargv1)
    
    # Extract the image data from the first HDU (Primary HDU or Image Data HDU)
    image_data = hdul[0].data
    image_data = np.transpose(image_data, (1, 2, 0))

    # Extract RGB channels
    R = image_data[:, :, 0]
    G = image_data[:, :, 1]
    B = image_data[:, :, 2]
    R = R.astype(np.float64)
    G = G.astype(np.float64)
    B = B.astype(np.float64)
        
    # Calculate the luminance (grayscale) using the standard formula
    luminance = 0.2989 * R + 0.5870 * G + 0.1140 * B
    # Create a new FITS HDU with the luminance data
    luminance_hdu = fits.PrimaryHDU(luminance)
        
    # Create an HDU list (this is just the luminance HDU in this case)
    hdulist = fits.HDUList([luminance_hdu])
        
    # Save the luminance image as a new FITS file
    hdulist.writeto(sysargv2, overwrite=True)

  if sysargv7 == '1':

    img = cv2.imread(sysargv1, -1)
    # createLuminance not percieved
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(sysargv2, grayscale_img)
    return sysargv1
    menue()

def align2img():

  # Load the two images
  sysargv1  = input("Enter the 1st reference image name(WCS/std) -->")
  sysargv7  = input("Enter 02, 03, 04, 05 for fits or 1 for other file -->")

    
  if sysargv7 == '02':
    sysargv3  = input("Enter the 2nd reference image file name(WCS/std) -->")  
  
    sysargv2  = input("Enter 1st Aligned image name(WCS/std)  -->")

    sysargv4  = input("Enter 2nd Aligned image name(WCS/std)  -->")
    # Open the FITS files

    def read_fits(file):
        hdul = fits.open(file)
        header = hdul[0].header
        data = hdul[0].data.astype(np.float64)
        hdul.close()
        return data, header

    # Read the FITS files
    file1 = sysargv1    
    data1, wcs1 = read_fits(file1)
    file1 = sysargv3    
    data2, wcs2 = read_fits(file1)

    # Find the optimal WCS for the reprojected images
    wcs_out, shape_out = find_optimal_celestial_wcs([(data1, wcs1), (data2, wcs2)])

    # Reproject the images to the new WCS
    array1, footprint1 = reproject_interp((data1, wcs1), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv2, overwrite=True)
    array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv4, overwrite=True)
    
  if sysargv7 == '03':
    sysargv3  = input("Enter the 2nd reference image file name(WCS/std) -->")  
    sysargv3a  = input("Enter the 3rd reference image file name(WCS/std) -->")  
    
    sysargv2  = input("Enter 1st Aligned image name(WCS/std)  -->")

    sysargv4  = input("Enter 2nd Aligned image name(WCS/std)  -->")
    # Open the FITS files

    sysargv4a  = input("Enter 3rd Aligned image name(WCS/std)  -->")
    # Open the FITS files

    def read_fits(file):
        hdul = fits.open(file)
        header = hdul[0].header
        data = hdul[0].data.astype(np.float64)
        hdul.close()
        return data, header

    # Read the FITS files
    file1 = sysargv1    
    data1, wcs1 = read_fits(file1)
    file1 = sysargv3    
    data2, wcs2 = read_fits(file1)
    file1 = sysargv3a    
    data3, wcs3 = read_fits(file1)

    # Find the optimal WCS for the reprojected images
    wcs_out, shape_out = find_optimal_celestial_wcs([(data1, wcs1), (data2, wcs2), (data3, wcs3)])

    # Reproject the images to the new WCS
    array1, footprint1 = reproject_interp((data1, wcs1), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv2, overwrite=True)
    array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv4, overwrite=True)
    array3, footprint3 = reproject_interp((data3, wcs3), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array3, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv4a, overwrite=True)

  if sysargv7 == '04':
    sysargv3  = input("Enter the 2nd reference image file name(WCS/std) -->")  
    sysargv3a  = input("Enter the 3rd reference image file name(WCS/std) -->")  
    sysargv3b  = input("Enter the 4th reference image file name(WCS/std) -->")  
    
    sysargv2  = input("Enter 1st Aligned image name(WCS/std)  -->")

    sysargv4  = input("Enter 2nd Aligned image name(WCS/std)  -->")
    # Open the FITS files

    sysargv4a  = input("Enter 3rd Aligned image name(WCS/std)  -->")
    # Open the FITS files

    sysargv4b  = input("Enter 4th Aligned image name(WCS/std)  -->")
    # Open the FITS files

    def read_fits(file):
        hdul = fits.open(file)
        header = hdul[0].header
        data = hdul[0].data.astype(np.float64)
        hdul.close()
        return data, header

    # Read the FITS files
    file1 = sysargv1    
    data1, wcs1 = read_fits(file1)
    file1 = sysargv3    
    data2, wcs2 = read_fits(file1)
    file1 = sysargv3a    
    data3, wcs3 = read_fits(file1)
    file1 = sysargv3b    
    data4, wcs4 = read_fits(file1)

    # Find the optimal WCS for the reprojected images
    wcs_out, shape_out = find_optimal_celestial_wcs([(data1, wcs1), (data2, wcs2), (data3, wcs3), (data4, wcs4)])

    # Reproject the images to the new WCS
    array1, footprint1 = reproject_interp((data1, wcs1), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv2, overwrite=True)
    array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv4, overwrite=True)
    array3, footprint3 = reproject_interp((data3, wcs3), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array3, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv4a, overwrite=True)
    array4, footprint4 = reproject_interp((data4, wcs4), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array4, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv4b, overwrite=True)

    
  if sysargv7 == '05':
    sysargv3  = input("Enter the 2nd reference image file name(WCS/std) -->")  
    sysargv3a  = input("Enter the 3rd reference image file name(WCS/std) -->")  
    sysargv3b  = input("Enter the 4th reference image file name(WCS/std) -->")  
    sysargv3c  = input("Enter the 5threference image file name(WCS/std) -->") 
        
    sysargv2  = input("Enter 1st Aligned image name(WCS/std)  -->")

    sysargv4  = input("Enter 2nd Aligned image name(WCS/std)  -->")
    # Open the FITS files

    sysargv4a  = input("Enter 3rd Aligned image name(WCS/std)  -->")
    # Open the FITS files

    sysargv4b  = input("Enter 4th Aligned image name(WCS/std)  -->")
    # Open the FITS files

    sysargv4c  = input("Enter 5th Aligned image name(WCS/std)  -->")
    # Open the FITS files

    def read_fits(file):
        hdul = fits.open(file)
        header = hdul[0].header
        data = hdul[0].data.astype(np.float64)
        hdul.close()
        return data, header

    # Read the FITS files
    file1 = sysargv1    
    data1, wcs1 = read_fits(file1)
    file1 = sysargv3    
    data2, wcs2 = read_fits(file1)
    file1 = sysargv3a    
    data3, wcs3 = read_fits(file1)
    file1 = sysargv3b    
    data4, wcs4 = read_fits(file1)
    file1 = sysargv3c    
    data5, wcs5 = read_fits(file1)

    # Find the optimal WCS for the reprojected images
    wcs_out, shape_out = find_optimal_celestial_wcs([(data1, wcs1), (data2, wcs2), (data3, wcs3), (data4, wcs4)])

    # Reproject the images to the new WCS
    array1, footprint1 = reproject_interp((data1, wcs1), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv2, overwrite=True)
    array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv4, overwrite=True)
    array3, footprint3 = reproject_interp((data3, wcs3), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array3, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv4a, overwrite=True)
    array4, footprint4 = reproject_interp((data4, wcs4), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array4, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv4b, overwrite=True)
    array5, footprint5 = reproject_interp((data5, wcs5), wcs_out, shape_out=shape_out)
    # Assuming array1 is the reprojected data and wcs_out is the WCS header
    hdu = fits.PrimaryHDU(data=array5, header=wcs_out.to_header())
    # Write to the FITS file
    hdu.writeto(sysargv4c, overwrite=True)

  if sysargv7 == '1':

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
  #sysargv4  = input("Enter the image width in pixels(1000)  -->")
  #sysargv3  = input("Enter the image height in pixels(1000)  -->")
  sysargv5  = input("Enter the final image name progrm will output a .fit file   -->") 
  sysargv6  = input("Enter the bin value   -->") 
  gamma     = float(input("Enter gamma(.3981) for 1 magnitude  -->"))

  # Replace 'your_fits_file.fits' with the actual path to your FITS file
  fits_image_filename = sysargv1
  # Open the FITS file
  with fits.open(fits_image_filename) as hdul:
      # Access the primary HDU (extension 0)
      header = hdul[0].header
      image_data = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  print(image_data.shape)
  print(image_data.dtype.name)
  height, width = image_data.shape
  sysargv4 = str(width)
  sysargv3 = str(height)


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

  hdu = fits.PrimaryHDU(binned_image, header)
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
  #sysargv4  = input("Enter the image width in pixels(1000)  -->")
  #sysargv3  = input("Enter the image height in pixels(1000)  -->")
  sysargv5  = input("Enter the final image name progrm will output a .fit file   -->") 
  sysargv6  = input("Enter the bin value   -->") 
  gamma     = float(input("Enter gamma(.3981) for 1 magnitude  -->"))


#################################################################################
  # Replace 'your_fits_file.fits' with the actual path to your FITS file
  fits_image_filename = sysargv1
  # Open the FITS file
  with fits.open(fits_image_filename) as hdul:
      # Access the primary HDU (extension 0)
      header = hdul[0].header
      image_data = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  print(image_data.shape)
  print(image_data.dtype.name)
  height, width = image_data.shape
  sysargv4 = str(width)
  sysargv3 = str(height)

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
      header = hdul[0].header
      image_data = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  print(image_data.shape)
  print(image_data.dtype.name)
  height, width = image_data.shape
  sysargv4 = str(width)
  sysargv3 = str(height)

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
      header = hdul[0].header
      image_data = hdul[0].data
  # Now 'image_data' contains the data from the FITS file as a 2D numpy array
  hdul.close()

  print(image_data.shape)
  print(image_data.dtype.name)
  height, width = image_data.shape
  sysargv4 = str(width)
  sysargv3 = str(height)

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
  sysargv2  = input("Enter the Color Image(fits(gry))  -->")
  sysargv3  = input("Enter the fits image to be created(w/o-GauBlr.fit)  -->")
  sysargv4a  = input("Enter the gausian blur 1.313, 2 etc.  -->")
  sysargv4 = float(sysargv4a)

  def apply_gaussian_blur(image, sigma):
    # Apply Gaussian blur with specified sigma
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return blurred_image

  # Load the FITS file
  fits_path = sysargv2
  hdul = fits.open(fits_path)
  image_data = hdul[0].data

  # Apply the Gaussian blur
  sigma = sysargv4  # Standard deviation in X and Y directions
  blurred_image = apply_gaussian_blur(image_data, sigma)

  # Save the final image as a FITS file
  hdu = fits.PrimaryHDU(blurred_image)
  hdu.writeto(sysargv3 + 'GauBlr' + '.fit', overwrite=True)

  return sysargv1
  menue()

def FFT():
  sysargv2  = input("Enter the Greyscale Image  -->")
  sysargv3  = input("Enter the FFT HP image to be created  -->")
  sysargv4  = input("Enter the cutoff(25)(NUM)  -->")
  sysargv5  = input("Enter the weight(50)(NUM)   -->")
  sysargv6  = input("Enter the Denominator(100)  -->")
  sysargv7  = input("Enter the radius(1))  -->")
  sysargv8  = input("Enter the cutoff(10))  -->")

  #copilot output
  def high_pass_filter(image, cutoff=(int(sysargv4)/int(sysargv6)), weight=(int(sysargv5)/int(sysargv6))):
      # Perform FFT
      fft = np.fft.fft2(image)
      fft_shift = np.fft.fftshift(fft)

      # Create a high-pass filter mask
      rows, cols = image.shape
      crow, ccol = rows // 2, cols // 2
      mask = np.ones((rows, cols), np.float32)
      r = int(cutoff * min(rows, cols))
      center = (crow, ccol)
      mask[center[0]-r:center[0]+r, center[1]-r:center[1]+r] = 0

      # Apply the mask to the FFT shift
      fft_shift_filtered = fft_shift * mask
  
      # Perform inverse FFT
      fft_inverse_shift = np.fft.ifftshift(fft_shift_filtered)
      image_filtered = np.fft.ifft2(fft_inverse_shift)
      image_filtered = np.abs(image_filtered)

      # Weight the filtered image
      image_weighted = cv2.addWeighted(image, 1 - weight, image_filtered.astype(np.float32), weight, 0)

      return image_weighted

  def feather_image(image, radius=int(sysargv7), distance=int(sysargv8)):
      # Reduce radius
      image_blurred = cv2.GaussianBlur(image, (radius, radius), 0)
    
      # Create a mask with feathered edges covering the entire image
      mask = np.zeros(image.shape, dtype=np.uint8)
      mask[:,:] = 255  # Set all mask values to 255 (white)
    
      # Apply distance feathering
      mask_blurred = cv2.GaussianBlur(mask, (distance*2+1, distance*2+1), 0)
    
      # Apply the mask to the image
      result_image = cv2.bitwise_and(image_blurred, image_blurred, mask=mask_blurred)

      return result_image

    # Load the FITS file
  fits_path = sysargv2
  hdul = fits.open(fits_path)
  image_data = hdul[0].data

  # Normalize the image data
  image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)

  # Apply high-pass filter
  filtered_image = high_pass_filter(image_data, cutoff=(int(sysargv4)/int(sysargv6)), weight=(int(sysargv5)/int(sysargv6)))

  # Apply feathering
  final_image = feather_image(filtered_image, radius=int(sysargv7), distance=int(sysargv8))

  # Save the final image as a FITS file
  hdu = fits.PrimaryHDU(final_image)
  hdu.writeto( sysargv3, overwrite=True)

   
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
  sysargv2  = input("Enter the first (default fits Siril) Image  -->")
  sysargv3  = input("Enter the second (default fits Siril) Image  -->")

  # Load the first FITS file
  hdul1 = fits.open(sysargv2)
  image_data1 = hdul1[0].data.astype(np.float64)
    
  # Load the second FITS file
  hdul2 = fits.open(sysargv3)
  image_data2 = hdul2[0].data.astype(np.float64)
    
  # Check if the image data from both FITS files have the same shape
  if image_data1.shape != image_data2.shape:
    print("Error: The input images do not have the same dimensions!")
    hdul1.close()
    hdul2.close()
    return
  print(image_data1.shape)
  print(image_data2.shape)
   #Add the RGB channels from both images
   #Assuming both images are in the format (height, width, 3) (RGB)
  if image_data1.ndim == 3 and image_data2.ndim == 3 :
   # Add the corresponding channels (R, G, B) of both images
    sysargv4  = input("Enter the filename of the added images to save  -->")
    sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
    sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
    sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
    sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")
    sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
    image_data1_contrastadd = int(sysargv5)
    image_data1_contrastscale = (int(sysargv6)/int(sysargv7))
    image_data2_contrastscale = (int(sysargv8)/int(sysargv9))

    result_image = ((image_data1 * image_data1_contrastscale ) + (image_data2 * image_data1_contrastscale)) + image_data1_contrastadd

    #result_image = image_data1 + image_data2

    # Create a new FITS HDU for the result
    result_hdu = fits.PrimaryHDU(result_image)
        
    # Create an HDU list (this only contains the result HDU)
    hdulist = fits.HDUList([result_hdu])
        
    # Save the result as a new FITS file
    hdulist.writeto(sysargv4, overwrite=True)
        
  else:
    print("Error: The FITS files do not appear to be in the expected RGB format.")
    
  # Close the FITS files
    hdul1.close()
    hdul2.close()

  return sysargv1
  menue()

def subtract2images():
  sysargv2  = input("Enter the first (default fits Siril) Image  -->")
  sysargv3  = input("Enter the second (default fits Siril) Image  -->")

  # Load the first FITS file
  hdul1 = fits.open(sysargv2)
  image_data1 = hdul1[0].data.astype(np.float64)
    
  # Load the second FITS file
  hdul2 = fits.open(sysargv3)
  image_data2 = hdul2[0].data.astype(np.float64)
    
  # Check if the image data from both FITS files have the same shape
  if image_data1.shape != image_data2.shape:
    print("Error: The input images do not have the same dimensions!")
    hdul1.close()
    hdul2.close()
    return
  print(image_data1.shape)
  print(image_data2.shape)
   #Add the RGB channels from both images
   #Assuming both images are in the format (height, width, 3) (RGB)
  if image_data1.ndim == 3 and image_data2.ndim == 3 :
   # Add the corresponding channels (R, G, B) of both images
    sysargv4  = input("Enter the filename of the added images to save  -->")
    sysargv6  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(numerator example 1)   -->")
    sysargv7  = input("Adjusts Image1 contrast by scaling the pixel values 1st img(denominator example 1)  -->")
    sysargv8  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(numerator example 1)   -->")
    sysargv9  = input("Adjusts Image2 contrast by scaling the pixel values 2nd img(denominator example 1)  -->")
    sysargv5  = input("Adjusts Image1 brightness by adding x to each pixel value example 0   -->")
    image_data1_contrastadd = int(sysargv5)
    image_data1_contrastscale = (int(sysargv6)/int(sysargv7))
    image_data2_contrastscale = (int(sysargv8)/int(sysargv9))

    result_image = ((image_data1 * image_data1_contrastscale ) - (image_data2 * image_data1_contrastscale)) + image_data1_contrastadd

    #result_image = image_data1 + image_data2

    # Create a new FITS HDU for the result
    result_hdu = fits.PrimaryHDU(result_image)
        
    # Create an HDU list (this only contains the result HDU)
    hdulist = fits.HDUList([result_hdu])
        
    # Save the result as a new FITS file
    hdulist.writeto(sysargv4, overwrite=True)
        
  else:
    print("Error: The FITS files do not appear to be in the expected RGB format.")
    
  # Close the FITS files
    hdul1.close()
    hdul2.close()

  return sysargv1
  menue()

def clahe():
  sysargv2  = input("Enter file name of color image to enter(16bit tif/png/fit) -->")
  sysargv3  = input("Enter clip limit (3) -->")
  sysargv4  = input("Enter tile Grid Size (8) -->")
  sysargv5  = input("Enter output filename -->")
  sysargv7  = input("Enter 0 for fits or 1 for other file -->")

  if sysargv7 == '0':

    # Function to read FITS file and return data
    # Read the FITS file
    hdulist = fits.open(sysargv2)
    header = hdulist[0].header
    image_data = hdulist[0].data
    hdulist.close()

    #image_data = np.swapaxes(image_data, 0, 2)
    #image_data = np.swapaxes(image_data, 0, 1)
    image_data = np.transpose(image_data, (1, 2, 0))

    # Normalize the image data to the range [0, 65535]
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 65535
    image_data = image_data.astype(np.uint16)

    # Convert the image to BGR format (OpenCV uses BGR by default)
    image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

    # Apply CLAHE to each channel
    clahe = cv2.createCLAHE(clipLimit=int(sysargv3), tileGridSize=(int(sysargv4), int(sysargv4)))
    channels = cv2.split(image_bgr)
    clahe_channels = [clahe.apply(channel) for channel in channels]
    clahe_image = cv2.merge(clahe_channels)

    # Save or display the result
    image_rgb = np.transpose(clahe_image, (2, 0, 1))

    # Create a FITS HDU
    hdu = fits.PrimaryHDU(image_rgb, header)

    # Write to FITS file
    hdu.writeto(sysargv5,  overwrite=True)
    
    # Save or display the result
    #cv2.imshow('CLAHE Image', clahe_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


  if sysargv7 == '1':

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

  sysargv2  = input("Enter file name of image(.tif/.fit) -->")
  sysargv7  = input("Enter 0 for fits or 1 for other file -->")

  if sysargv7 == '0':

    # Function to read FITS file and return data
    def read_fits(file1):
      with fits.open(file1) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        hdul.close()
        return data

    # Read the FITS files
    file1 = sysargv2

    data1 = read_fits(file1)

    image_data = data1.astype(np.float64)
    data_range = np.max(image_data) - np.min(image_data)
    if data_range == 0:
      normalized_data = np.zeros_like(image_data)  # or handle differently
    else:
      normalized_data = (image_data - np.min(image_data)) / data_range
    data1 = normalized_data

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

def CpyOldHdr():

  sysargv2  = input("Enter file name of image with correct old fitsheader  -->")
  sysargv3  = input("Enter file name of image with correct image  -->")
  sysargv4  = input("Enter new file name to be updated with old header new image -->")

  with fits.open(sysargv2) as old_hdul:
      # Access the header of the primary HDU
    old_header = old_hdul[0].header
    old_data = old_hdul[0].data
 
  file = sysargv3
   
  # Function to read FITS file and return data
  def read_fits(file):
    with fits.open(file, mode='update') as hdul:#
      data = hdul[0].data
      # hdul.close()
    return data

  new_image_data = read_fits(file)

  fits.writeto( sysargv4, new_image_data, old_header, overwrite=True)
  # Save the RGB image as a new FITS file with the correct header
  #hdu = fits.PrimaryHDU(data=new_image_data, header=old_header)
  #hdu.writeto(sysargv3, overwrite=True)

  return sysargv1
  menue()

def binimg():
  
  sysargv2  = input("Enter file name of input color image fits  -->")
  sysargv3  = input("Enter file name of binned color image  -->")
  sysargv4  = input("Enter binning_factor(25) -->")
  bin_size  = int(sysargv4)
  with fits.open(sysargv2) as hdul:
      data = hdul[0].data
        
      # Check if the image is color (3D array)
      if data.ndim == 3:
        binned_data = np.zeros((data.shape[0], data.shape[1] // bin_size, data.shape[2] // bin_size), dtype=data.dtype)
        for i in range(data.shape[0]):  # Iterate through color channels
          for y in range(0, data.shape[1], bin_size):
            for x in range(0, data.shape[2], bin_size):
              binned_data[i, y // bin_size, x // bin_size] = np.mean(data[i, y:y + bin_size, x:x + bin_size])
      else:
      # Handle grayscale or other formats if needed
        raise ValueError("Only color images are supported. The image must be a 3D array.")

      # Create a new HDU with the binned data
      hdu = fits.PrimaryHDU(binned_data)
      # Copy header from original
      hdu.header = hdul[0].header.copy()
      # Save the binned image to a new FITS file
      hdu.writeto(sysargv3, overwrite=True) 

  return sysargv1
  menue()


def autostr():

  sysargv3  = input("Enter file name of image to auto_str  -->")
  sysargv4  = input("Enter file name of output image -->")

  # Function for histogram transformation--Copilot--
  def histogram_transformation(image, shadows=0.1, midtones=1.0, highlights=0.9):
    """
      Perform histogram transformation with midtone, shadow, and highlight adjustments.
      Args:
          image (numpy.ndarray): Input grayscale or RGB image.
          shadows (float): Shadow cutoff (0 to 1).
          midtones (float): Gamma-like adjustment for midtones (default 1.0).
          highlights (float): Highlight cutoff (0 to 1).

      Returns:
          numpy.ndarray: Transformed image.
    """
      # Process each channel for RGB or single grayscale image
    if len(image.shape) == 3:
      channels = [image[0], image[1], image[2]]  # Split RGB (assuming FITS stores as separate layers)
    else:
        channels = [image]

    result_channels = []
    for channel in channels:
      # Calculate percentiles
      low_bound = np.percentile(channel, shadows * 100)
      high_bound = np.percentile(channel, highlights * 100)

      # Clip and normalize
      clipped = np.clip(channel, low_bound, high_bound)
      normalized = (clipped - low_bound) / (high_bound - low_bound)
      normalized = np.clip(normalized, 0, 1)

      # Adjust midtones
      adjusted = np.power(normalized, 1.0 / midtones)

      # Scale to 16-bit FITS data range
      transformed = (adjusted * 65535).astype(np.uint16)
      result_channels.append(transformed)

    # Merge RGB back or return single grayscale channel
    if len(result_channels) > 1:
        return np.stack(result_channels, axis=0)  # Stack for RGB
    return result_channels[0]

  # Read FITS file (replace 'input.fits' with your FITS file path)
  fits_file = sysargv3
  hdul = fits.open(fits_file)
  image_data = hdul[0].data.astype(np.float32)  # Assuming the FITS file contains an image in 16-bit or floating-point
  hdul.close()

  # Ensure input is 2D or 3D
  if len(image_data.shape) != 2 and len(image_data.shape) != 3:
    raise ValueError("The FITS file must contain a 2D (grayscale) or 3D (RGB) image.")

  # Apply histogram transformation
  transformed_image = histogram_transformation(image_data, shadows=0.02, midtones=1.2, highlights=0.98)

  # Write to a new FITS file
  output_fits_file = sysargv4
  hdu = fits.PrimaryHDU(transformed_image)
  hdu.writeto(output_fits_file, overwrite=True)
  print(f"Transformed FITS file saved as '{output_fits_file}'")


  return sysargv1
  menue()

def LocAdapt():

  sysargv2  = input("Enter file name of to enhance contrast  -->")
  sysargv3  = input("Enter file name of output(w/o-LoAd.fit) -->")
  sysargv4  = input("Enter numerator of contrast(10)  -->")
  sysargv5  = input("Enter denominator contrast(100) -->")
  sysargv6  = input("Enter neighborhood_size(7) -->")

  def contrast_filter(image, neighborhood_size=int(sysargv6)):
      # Define the kernel for calculating local mean and standard deviation
      kernel = np.ones((neighborhood_size, neighborhood_size), np.float32) / (neighborhood_size * neighborhood_size)
    
      # Calculate local mean
      local_mean = cv2.filter2D(image, -1, kernel)
    
      # Calculate local mean of squared values
      squared_image = np.square(image)
      local_mean_squared = cv2.filter2D(squared_image, -1, kernel)
    
      # Calculate local standard deviation
      local_std = np.sqrt(local_mean_squared - np.square(local_mean))
    
      # Avoid division by zero
      local_std[local_std == 0] = 1
    
      # Calculate contrast factor
      #contrast_factor = (local_mean / local_std)
      contrast_factor = (int(sysargv4) / int(sysargv5))
    
      # Calculate the contrast-enhanced image
      enhanced_image = (image - local_mean) * contrast_factor + local_mean
      enhanced_image = np.clip(enhanced_image, 0, np.max(image)).astype(image.dtype)
    
      return enhanced_image

  # Load the FITS file
  fits_path = sysargv2
  hdul = fits.open(fits_path)
  image_data = hdul[0].data

  # Apply the contrast filter
  filtered_image = contrast_filter(image_data, neighborhood_size=int(sysargv6))

  # Save the final image as a FITS file
  hdu = fits.PrimaryHDU(filtered_image)
  hdu.writeto( sysargv3 + 'LoAd' + '.fit', overwrite=True)

  return sysargv1
  menue()

def WcsOvrlay():

  sysargv2  = input("Enter fits wcs file name  -->")
  sysargv3  = input("Enter Title of plot -->")

  # Load the color FITS file
  fits_path = sysargv2
  hdul = fits.open(fits_path)

  # Assuming the color FITS file has three axes for R, G, and B channels in a single HDU
  image_data = hdul[0].data
  wcs = WCS(hdul[0].header, naxis=2)

  # Extract the R, G, and B channels
  red_data = image_data[0]
  green_data = image_data[1]
  blue_data = image_data[2]

  # Normalize the data to the range [0, 1]
  red_data = red_data / np.max(red_data)
  green_data = green_data / np.max(green_data)
  blue_data = blue_data / np.max(blue_data)

  # Create an RGB image
  rgb_image = np.stack((red_data, green_data, blue_data), axis=-1)

  # Create a plot with world coordinates
  plt.figure(figsize=(10, 10))
  ax = plt.subplot(projection=wcs)
  ax.imshow(rgb_image, origin='lower')
  ax.set_xlabel('Right Ascension')
  ax.set_ylabel('Declination')
  ax.coords.grid(True, color='white', ls='dotted')
  plt.title( sysargv3 )  # Add the title here
  # Display the plot
  plt.show()

  return sysargv1
  menue()

def WcsStack():

  #================================================================================
  sysargv1  = input("Enter the  directory path(E:'//work//'w)  -->")
  sysargv3  = input("Enter the output file(stacked_image.fits)  -->")
  sysargv2  = input("Enter file type as (.fit)  -->")
  #================================================================================
  directory_path = sysargv1

  # Get all file names
  all_files = os.listdir(directory_path)

  # Filter only .fit or .fits files
  fits_files = [file for file in all_files if file.endswith('.fit') or file.endswith('.fits')]

  # Print the names of all FITS files
  print("FITS files in the directory:")
  if not fits_files:
      raise FileNotFoundError("No FITS files found in the specified directory!")
  for file in fits_files:
      print(file)

  # Load the reference image (use the first image as the reference)
  ref_hdul = fits.open(os.path.join(directory_path, fits_files[0]))
  ref_header = ref_hdul[0].header
  ref_data = ref_hdul[0].data.astype(np.float64)  # Ensure float64 precision
  print(f"Successfully opened reference FITS file: {fits_files[0]}")
  ref_wcs = WCS(ref_header)

  # Reproject and align all images to the reference WCS
  aligned_images = []

  fits_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if 

  file.endswith( sysargv2 ) ]
  if not ref_wcs.has_celestial:
      raise ValueError(f"Reference image {fits_files[0]} does not contain valid celestial WCS information.")

  for fits_file in fits_files:
      with fits.open(fits_file) as hdul:
          target_header = hdul[0].header
          target_data = hdul[0].data.astype(np.float64)

          try:
              target_wcs = WCS(target_header)
              if not target_wcs.has_celestial:
                  raise ValueError("No celestial WCS information found.")
          except Exception as e:
              print(f"Skipping {fits_file} due to WCS error: {e}")
              continue

          # Reproject the target image onto the reference WCS
          reprojected_data, _ = reproject_exact((target_data, target_wcs), ref_wcs, ref_data.shape)
          aligned_images.append(reprojected_data)

  # Close the reference image
  ref_hdul.close()

  # Stack the aligned images (e.g., average stacking)
  stacked_image = np.mean(aligned_images, axis=0).astype(np.float64)

  # Save the stacked image to a new FITS file
  stacked_hdu = fits.PrimaryHDU(stacked_image, header=ref_header)
  stacked_hdu.writeto(sysargv3, overwrite=True)

  print("Stacked image saved as " + sysargv3 )


  return sysargv1
  menue()

def combinelrgb():

  sysargv0  = input("Enter the Lum image to be combined  -->")
  sysargv1  = input("Enter the Blue image to be combined  -->")
  sysargv2  = input("Enter the Green image to be combined  -->")
  sysargv3  = input("Enter the Red image to be combined  -->")
  sysargv4  = input("Enter the RGB file to be created  -->")
 
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
  file0 = sysargv0
  file1 = sysargv1
  file2 = sysargv2
  file3 = sysargv3

  # Read the image data from the FITS file
  lum = read_fits(file0)
  blue = read_fits(file1)
  green = read_fits(file2)
  red = read_fits(file3)

  lum = blue.astype(np.float64)
  blue = blue.astype(np.float64)
  green = green.astype(np.float64)
  red = red.astype(np.float64)

  # Check dimensions
  print("Data0 shape:", lum.shape)
  print("Data1 shape:", blue.shape)
  print("Data2 shape:", green.shape)
  print("Data3 shape:", red.shape)

  #newRGBImage = cv2.merge((red,green,blue))
  RGB_Image1 = np.stack((red,green,blue))

  # Remove the extra dimension
  RGB_Image = np.squeeze(RGB_Image1)

  # Normalize the data for RGB scaling (0 to 1)
  def normalize(data):
      return (data - np.min(data)) / (np.max(data) - np.min(data))

  luminance = normalize(lum)
  red = normalize(red)
  green = normalize(green)
  blue = normalize(blue)

  # Combine the channels into an RGB array while maintaining float64 precision
  rgb_array = np.zeros((luminance.shape[0], luminance.shape[1], 3), dtype=np.float64)
  rgb_array[ :, :, 2] = red * luminance  # R channel with luminance
  rgb_array[ :, :, 1] = green * luminance  # G channel with luminance
  rgb_array[ :, :, 0] = blue * luminance  # B channel with luminance

  # Create a FITS header with NAXIS = 3
  header = old_header
  header['NAXIS'] = 3
  header['NAXIS1'] = RGB_Image.shape[2]
  header['NAXIS2'] = RGB_Image.shape[1]
  header['NAXIS3'] = RGB_Image.shape[0]
  header['FILTER'] = 'Luminance+Red+Green+Blue' 

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

  return sysargv1
  menue()



def menue(sysargv1):
  sysargv1 = input("Enter \n>>1<< AffineTransform(3pts) >>2<< Mask an image >>3<< Mask Invert >>4<< Add2images(fit)  \n>>5<< Split tricolor >>6<< Combine Tricolor >>7<< Create Luminance(2ax) >>8<< Align2img \n>>9<< Plot_16-bit_img to 3d graph(2ax) >>10<< Centroid_Custom_filter(2ax) >>11<< UnsharpMask \n>>12<< FFT-Bandpass(2ax) >>13<< Img-DeconvClr >>14<< Centroid_Custom_Array_loop(2ax) \n>>15<< Erosion(2ax) >>16<< Dilation(2ax) >>17<< DynamicRescale(2ax) >>18<< GausBlur  \n>>19<< DrCntByFileType >>20<< ImgResize >>21<< JpgCompress >>22<< subtract2images(fit)  \n>>23<< multiply2images >>24<< divide2images >>25<< max2images >>26<< min2images \n>>27<< imgcrop >>28<< imghiststretch >>29<< gif  >>30<< aling2img(2pts) >>31<< Video \n>>32<< gammaCor >>33<< ImgQtr >>34<< CpyOldHdr >>35<< DynReStr(RGB) \n>>36<< clahe >>37<< pm_vector_line >>38<< hist_match >>39<< distance >>40<< EdgeDetect \n>>41<< Mosaic(4) >>42<< BinImg >>43<< autostr >>44<< LocAdapt >>45<< WcsOvrlay \n>>46<< WcsStack >>47<< CombineLRGB \n>>1313<< Exit --> ")
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
    imgqtr()

  if sysargv1 == '34':
    CpyOldHdr()    

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
    binimg()

  if sysargv1 == '43':
    autostr()

  if sysargv1 == '44':
    LocAdapt()

  if sysargv1 == '45':
    WcsOvrlay()

  if sysargv1 == '46':
     WcsStack()

  if sysargv1 == '47':
    combinelrgb()

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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      