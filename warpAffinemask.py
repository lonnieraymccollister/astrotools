# some code from copilot
# import required libraries
import fnmatch
from PIL import Image
import cv2, sys, os, shutil, ffmpeg, tifffile
import numpy as np
import matplotlib
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits
from astropy.convolution import convolve_fft
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian2DKernel
from astropy.wcs import WCS
from astropy.coordinates import Angle
import mpmath as mp
import glob
from skimage.exposure import match_histograms
from scipy.special import erfinv
from scipy.ndimage import zoom 
from scipy.ndimage import convolve
from scipy.signal import convolve2d
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
  try:
      
      def main():
          
                 sysargv1  = input("Enter the Image1  -->")
                 sysargv3  = input("Enter the Mask for image  white- and black  -->")
                 sysargv4  = input("Enter the filename of the masked image to save  -->")
           
                 image = cv2.imread(sysargv1, -1)
                 mask = cv2.imread(sysargv3, -1)
           
                 # Apply the mask to the image
                 masked_image = cv2.bitwise_and(image, mask)
                 cv2.imwrite(sysargv4, masked_image)
                 
      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def maskinvert():
  try:
      
      def main():
      
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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def filecount():
  try:
      
      def main():
      
                 sysargv1  = input("Enter the  directory path from explorer  -->")
                 sysargv2  = input("Enter file type as (*.fit)  -->")
                 count = len(fnmatch.filter(os.listdir(sysargv1), sysargv2))
                 print('File Count:', count)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def resize():
  try:

      def main():

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
           
      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def multiply2images():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def divide2images():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def max2images():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def min2images():

  try:

      def main():

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

      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def splittricolor():
    
  try:

      def main():

                 sysargv2a  = input("Enter the Color Image to be split  -->")
                 sysargv7  = input("Enter 0 for fits or 1 for other file -->")
                 sysargv2 = sysargv2a.split('.')[0]
           
                 if sysargv7 == '0':
           
                   # Function to read FITS file and return data
                   def read_fits(file):
                       hdul = fits.open(file)
                       header = hdul[0].header
                       data = hdul[0].data
                       hdul.close()
                       return data, header
           
                   # Read the FITS files
                   file1 = sysargv2a
           
                   # Read the image data from the FITS file
                   image_data, header = read_fits(file1)
                   image_data = image_data.astype(np.float32)
            
                   # Split the color image into its individual channels
                   #b, g, r = cv2.split(image_data)
                   b, g, r = image_data[2, :, :], image_data[1, :, :], image_data[0, :, :] 
           
           
                   # Save each channel as a separate file
                   fits.writeto(f'{sysargv2}channel_0_64bit.fits', b.astype(np.float32), header, overwrite=True)
                   fits.writeto(f'{sysargv2}channel_1_64bit.fits', g.astype(np.float32), header, overwrite=True)
                   fits.writeto(f'{sysargv2}channel_2_64bit.fits', r.astype(np.float32), header, overwrite=True)
           
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

      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def combinetricolor():

  try:

      def main():

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
           
                   blue = blue.astype(np.float32)
                   green = green.astype(np.float32)
                   red = red.astype(np.float32)
           
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
                   header['NAXIS1'] = RGB_Image.shape[0]
                   header['NAXIS2'] = RGB_Image.shape[1]
                   header['NAXIS3'] = RGB_Image.shape[2]
           
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

      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def createLuminance():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def align2img():

  try:

      def main():

                 # Load the two images
                 sysargv1  = input("Enter the 1st reference image name(WCS/std) -->")
                 sysargv7  = input("Enter 02, 03, 04, 05, 06 for fits or 1 for other file -->")
               
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
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array1 = np.flipud(array1)
                   if fliplr == '1':  array1 = np.fliplr(array1)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv2, overwrite=True)
                   array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1': array2 = np.flipud(array2)
                   if fliplr == '1': array2 = np.fliplr(array2)
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
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array1 = np.flipud(array1)
                   if fliplr == '1':  array1 = np.fliplr(array1)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv2, overwrite=True)
                   array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array2 = np.flipud(array2)
                   if fliplr == '1':  array2 = np.fliplr(array2)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4, overwrite=True)
                   array3, footprint3 = reproject_interp((data3, wcs3), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array3 = np.flipud(array3)
                   if fliplr == '1':  array3 = np.fliplr(array3)
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
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array1 = np.flipud(array1)
                   if fliplr == '1':  array1 = np.fliplr(array1)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv2, overwrite=True)
                   array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array2 = np.flipud(array2)
                   if fliplr == '1':  array2 = np.fliplr(array2)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4, overwrite=True)
                   array3, footprint3 = reproject_interp((data3, wcs3), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array3 = np.flipud(array3)
                   if fliplr == '1':  array3 = np.fliplr(array3)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array3, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4a, overwrite=True)
                   array4, footprint4 = reproject_interp((data4, wcs4), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array4 = np.flipud(array4)
                   if fliplr == '1':  array4 = np.fliplr(array4)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array4, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4b, overwrite=True)
           
               
                 if sysargv7 == '05':
                   sysargv3  = input("Enter the 2nd reference image file name(WCS/std) -->")  
                   sysargv3a  = input("Enter the 3rd reference image file name(WCS/std) -->")  
                   sysargv3b  = input("Enter the 4th reference image file name(WCS/std) -->")  
                   sysargv3c  = input("Enter the 5th reference image file name(WCS/std) -->") 
                   
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
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array1 = np.flipud(array1)
                   if fliplr == '1':  array1 = np.fliplr(array1)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv2, overwrite=True)
                   array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array2 = np.flipud(array2)
                   if fliplr == '1':  array2 = np.fliplr(array2)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4, overwrite=True)
                   array3, footprint3 = reproject_interp((data3, wcs3), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array3 = np.flipud(array3)
                   if fliplr == '1':  array3 = np.fliplr(array3)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array3, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4a, overwrite=True)
                   array4, footprint4 = reproject_interp((data4, wcs4), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array4 = np.flipud(array4)
                   if fliplr == '1':  array4 = np.fliplr(array4)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array4, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4b, overwrite=True)
                   array5, footprint5 = reproject_interp((data5, wcs5), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array5 = np.flipud(array5)
                   if fliplr == '1':  array5 = np.fliplr(array5)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array5, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4c, overwrite=True)
           
                 if sysargv7 == '06':
                   sysargv3  = input("Enter the 2nd reference image file name(WCS/std) -->")  
                   sysargv3a  = input("Enter the 3rd reference image file name(WCS/std) -->")  
                   sysargv3b  = input("Enter the 4th reference image file name(WCS/std) -->")  
                   sysargv3c  = input("Enter the 5th reference image file name(WCS/std) -->") 
                   sysargv3d  = input("Enter the 6th reference image file name(WCS/std) -->")
                   
                   sysargv2  = input("Enter 1st Aligned image name(WCS/std)  -->")
           
                   sysargv4  = input("Enter 2nd Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   sysargv4a  = input("Enter 3rd Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   sysargv4b  = input("Enter 4th Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   sysargv4c  = input("Enter 5th Aligned image name(WCS/std)  -->")
                   # Open the FITS files
           
                   sysargv4d  = input("Enter 6th Aligned image name(WCS/std)  -->")
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
                   file1 = sysargv3d    
                   data6, wcs6 = read_fits(file1)
           
           
                   # Find the optimal WCS for the reprojected images
                   wcs_out, shape_out = find_optimal_celestial_wcs([(data1, wcs1), (data2, wcs2), (data3, wcs3), (data4, wcs4)])
           
                   # Reproject the images to the new WCS
                   array1, footprint1 = reproject_interp((data1, wcs1), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array1 = np.flipud(array1)
                   if fliplr == '1':  array1 = np.fliplr(array1)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array1, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv2, overwrite=True)
                   array2, footprint2 = reproject_interp((data2, wcs2), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array2 = np.flipud(array2)
                   if fliplr == '1':  array2 = np.fliplr(array2)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array2, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4, overwrite=True)
                   array3, footprint3 = reproject_interp((data3, wcs3), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array3 = np.flipud(array3)
                   if fliplr == '1':  array3 = np.fliplr(array3)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array3, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4a, overwrite=True)
                   array4, footprint4 = reproject_interp((data4, wcs4), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array4 = np.flipud(array4)
                   if fliplr == '1':  array4 = np.fliplr(array4)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array4, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4b, overwrite=True)
                   array5, footprint5 = reproject_interp((data5, wcs5), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array5 = np.flipud(array5)
                   if fliplr == '1':  array5 = np.fliplr(array5)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array5, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4c, overwrite=True)
                   array6, footprint6 = reproject_interp((data6, wcs6), wcs_out, shape_out=shape_out)
                   # flip and mirror image
                   flipud  = input("Enter 1 to flipud image-(WCS/std) else 0 to remain same  -->")
                   fliplr  = input("Enter 1 to fliplr image-(WCS/std) else 0 to remain same  -->")
                   if flipud == '1':  array6 = np.flipud(array6)
                   if fliplr == '1':  array6 = np.fliplr(array6)
                   # Assuming array1 is the reprojected data and wcs_out is the WCS header
                   hdu = fits.PrimaryHDU(data=array6, header=wcs_out.to_header())
                   # Write to the FITS file
                   hdu.writeto(sysargv4d, overwrite=True)
           
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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def plotto3d16(sysargv2):

  try:
      
      def main():

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

      if __name__ == "__main__":
          main()                 

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def PNGcreateimage16(sysargv2, sysargv3, sysargv4, sysargv5):

  try:

      def main():

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
                 symfile = (sysargv2+"_"+sysargv4+"_"+sysargv5+".png")
                 im.save(symfile)
                 image_1 = imread(symfile)
                 # plot raw pixel data
                 blur = cv2.blur(image_1,(3,3)) 
                 pyplot.imshow(blur,cmap='gray')
                 # show the figure
                 pyplot.show()

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

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

  try:

      def main():

                 sysargv1  = input("Enter the Color Image  -->")
                 sysargv2  = input("Enter the unsharpMask image to be created  -->")
           
                 image = cv2.imread(sysargv1, -1)
                 gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
                 unsharp_image = cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)
                 cv2.imwrite( sysargv2, unsharp_image)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def DynamicRescale16():

  try:

      def main():

           # -------------------------------------------------
           # UTILITY FUNCTIONS FOR FITS HANDLING
           # -------------------------------------------------
                 def load_fits(file_path):
                     with fits.open(file_path) as hdul:
                         return hdul[0].data, hdul[0].header
           
                 def split_image(image, tile_size=(600, 600), output_dir="tiles"):
                     os.makedirs(output_dir, exist_ok=True)
                     h, w = image.shape
                     print("Image shape:", image.shape)
                     tile_h, tile_w = tile_size
                     tiles = []
           
                     for i in range(0, h, tile_h):
                         for j in range(0, w, tile_w):
                             # Get the subimage; may be smaller near the right/bottom edges.
                             sub_image = image[i:i+tile_h, j:j+tile_w]
                             sub_h, sub_w = sub_image.shape
           
                             # Create a full-size tile and place the subimage in the top-left corner.
                             padded_tile = np.zeros(tile_size)
                             padded_tile[:sub_h, :sub_w] = sub_image
           
                             # Save the tile with metadata in the filename.
                             tile_file = f"{output_dir}/tile_{i}_{j}_{sub_h}_{sub_w}.fits"
                             fits.writeto(tile_file, padded_tile, overwrite=True)
                             tiles.append(tile_file)
               
                     return tiles
           
                 import re
           
                 def reassemble_image(tiles, original_shape):
                     """
                     Reassemble the full image from a list of processed tile files.
                     The tile filename is expected to include the pattern:
                        tile_i_j_subH_subW
                     (even though additional suffix text is appended).
                     """
                     final_image = np.zeros(original_shape)
               
                     # Regular expression pattern to extract the metadata numbers.
                     pattern = r"tile_(\d+)_(\d+)_(\d+)_(\d+)"
               
                     for tile_file in tiles:
                         base = os.path.basename(tile_file)
                         match = re.search(pattern, base)
                         if not match:
                             print(f"Filename {base} does not match expected pattern.")
                             continue
                         i_str, j_str, sub_h_str, sub_w_str = match.groups()
                         i, j, sub_h, sub_w = map(int, [i_str, j_str, sub_h_str, sub_w_str])
               
                         with fits.open(tile_file) as hdul:
                             data = hdul[0].data
                             # Only take the valid (unpadded) portion from the tile.
                             final_image[i:i+sub_h, j:j+sub_w] = data[:sub_h, :sub_w]
           
                     return final_image
           
                 # -------------------------------------------------
                 # UPDATED TILE PROCESSING FUNCTION
                 # -------------------------------------------------
                 def process_tile(tile_file, width_of_square, bin_value, gamma_value, resize_factor, resize_div):
                     """
                     Processes a given tile file applying dynamic block rescaling,
                     gamma correction, and binning. The parameters are passed in from main.
                     """
                     print(f"\nProcessing tile: {tile_file}")
               
                     # Open the tile file.
                     with fits.open(tile_file) as hdul:
                         header = hdul[0].header
                         image_data = hdul[0].data
           
                     print("Original tile shape:", image_data.shape)
                     print("Data type:", image_data.dtype.name)
               
                     # Normalize the image data to the range [0, 65535] and cast to uint16.
                     norm_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 65535
                     norm_image = norm_image.astype(np.uint16)
               
                     # Resize the image using cv2.resize.
                     resized_image = cv2.resize(norm_image, None, fx=(resize_factor / resize_div), 
                                                fy=(resize_factor / resize_div), interpolation=cv2.INTER_LANCZOS4)
               
                     # Multiply the resized image to scale it up.
                     my_data = resized_image * 65535
                     img = resized_image * 65535
           
                     # Use the provided square block width for dynamic square processing.
                     block_size = int(width_of_square)
                     new_h, new_w = resized_image.shape
           
                     # Process image in blocks.
                     for xw in range(0, new_h, block_size):
                         for yh in range(0, new_w, block_size):
                             block_h = min(block_size, new_h - xw)
                             block_w = min(block_size, new_w - yh)
                             my_data1 = np.zeros((block_h, block_w))
                             for x in range(block_h):
                                 for y in range(block_w):
                                     my_data1[x, y] = img[x + xw, y + yh]
                             # Rescale block to the 0–65535 range.
                             rescaled1 = ((my_data1.max() + 1) * ((my_data1 + 1) - my_data1.min()) / 65535.0).astype(np.float64)
                             rescaled = np.round(rescaled1)
                             my_data[xw:xw+block_h, yh:yh+block_w] = rescaled[:block_h, :block_w]
           
                     # Apply gamma correction.
                     gamma_corrected1 = np.array(65535.0 * (my_data / 65535) ** gamma_value, dtype='float64')
                     gamma_corrected = np.round(gamma_corrected1)
               
                     # Normalize before binning (division factor from original code is 6553500).
                     img_array = np.asarray(gamma_corrected / 6553500, dtype='float64')
               
                     # Calculate new dimensions based on the bin factor.
                     bin_factor = int(bin_value)
                     h_img, w_img = img_array.shape
                     new_height = h_img // bin_factor
                     new_width = w_img // bin_factor
           
                     # Bin the image using summation in non-overlapping blocks.
                     binned_image = np.zeros((new_height, new_width), dtype=img_array.dtype)
                     for y in range(new_height):
                         for x in range(new_width):
                             binned_image[y, x] = np.sum(
                                 img_array[y * bin_factor:(y + 1) * bin_factor,
                                           x * bin_factor:(x + 1) * bin_factor]
                             )
           
                     # Write the processed tile to a new FITS file. The new file name gets the extra suffix.
                     hdu = fits.PrimaryHDU(binned_image, header=header)
                     hdulist = fits.HDUList([hdu])
                     out_filename = tile_file + '_binned_gamma_corrected_drs.fits'
                     hdulist.writeto(out_filename, overwrite=True)
                     print(f"Tile processed and saved to {out_filename}\n")
           
                 # -------------------------------------------------
                 # MAIN EXECUTION
                 # -------------------------------------------------
                 if __name__ == "__main__":
                     sysargv7 = input("Enter 01 to split tile, 02 to process tile files, or 03 to combine tiles --> ")
               
                     if sysargv7 == '01':
                         # Splitting mode.
                         input_file_name = input("Enter the input FITS file name--> ")
                         fits_file = input_file_name
                         image, header = load_fits(fits_file)
                         tiles = split_image(image)
                         print("Image split into tiles:")
                         for t in tiles:
                             print("  ", t)
               
                     elif sysargv7 == '02':
                         # Processing mode: Collect processing parameters once.
                         width_of_square = input("Enter the width of square (e.g., 5): ")
                         bin_value = input("Enter the bin value (e.g., 25): ")
                         gamma_value = float(input("Enter gamma (e.g., 0.3981) for 1 magnitude: "))
                         # Fixed values used for resizing.
                         resize_factor = int(25)   # corresponds to sysargv2 = 25 in the original code.
                         resize_div = int(1)       # corresponds to sysargv2a = 1 in the original code.
                   
                         # Get list of tiles from the "tiles" directory.
                         tiles = sorted([os.path.join("tiles", f) for f in os.listdir("tiles") if f.endswith(".fits")])
                         for tile_file in tiles:
                             process_tile(tile_file, width_of_square, bin_value,
                                          gamma_value, resize_factor, resize_div)
               
                     elif sysargv7 == '03':
                         # Reassemble mode.
                         input_file_name = input("Enter the input FITS file name--> ")
                         fits_file = input_file_name
                         image, header = load_fits(fits_file)
                         # Select only those processed tile files.
                         tiles = sorted([os.path.join("tiles", f) for f in os.listdir("tiles") if f.endswith("_binned_gamma_corrected_drs.fits")])
                         final_image = reassemble_image(tiles, image.shape)
                         fits.writeto("output.fits", final_image, header, overwrite=True)
                         # Optional: Cleanup processed tile files.
                         for tile_file in tiles:
                             os.remove(tile_file)
                         print("Processing complete. Final image saved as 'output.fits'.")
               
                     else:
                         print("Invalid option entered.")
           
      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def DynamicRescale16RGB():
  try:
      
      def main():      

                 sysargv1  = input("Enter the Image to be resized(LANCZOS4)  -->")
                 width_of_square  = input("Enter the the width of square(5)  -->")
                 sysargv2  = int(25)
                 sysargv2a = int(1)
                 sysargv3  = "img_enlarged_25x.fit"
           
                 #################################################################################
                 #################################################################################
           
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
           
                 #################################################################################
                 #################################################################################
           
                 sysargv2  = "img_enlarged_25x.fit"
           
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
                 b, g, r = image_data[2, :, :], image_data[1, :, :], image_data[0, :, :] 
           
           
                 # Save each channel as a separate file
                 fits.writeto(f'channel_0_64bit.fits', b.astype(np.float64), header, overwrite=True)
                 fits.writeto(f'channel_1_64bit.fits', g.astype(np.float64), header, overwrite=True)
                 fits.writeto(f'channel_2_64bit.fits', r.astype(np.float64), header, overwrite=True)
           
                 sysargv1  = "channel_0_64bit.fits"
                 sysargv1a  = "channel_1_64bit.fits"
                 sysargv1b  = "channel_2_64bit.fits"
                 #sysargv2  = input("Enter the the width of square(5)  -->")
                 sysargv2  = width_of_square
                 #sysargv4  = input("Enter the image width in pixels(1000)  -->")
                 #sysargv3  = input("Enter the image height in pixels(1000)  -->")
                 sysargv5  = "channel_RGB_64bit"
                 sysargv6  = "25" #image bin value
                 gamma     = float(".3981")
           
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
           
                 sysargv1  = "channel_RGB_64bit_binned_gamma_corrected_drs_B.fit"
                 sysargv2  = "channel_RGB_64bit_binned_gamma_corrected_drs_G.fit"
                 sysargv3  = "channel_RGB_64bit_binned_gamma_corrected_drs_R.fit"
                 sysargv4  = "channel_RGB_64bit_binned_gamma_corrected_drs_RGB.fit"
           
           
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
                 header['NAXIS1'] = RGB_Image.shape[0]
                 header['NAXIS2'] = RGB_Image.shape[1]
                 header['NAXIS3'] = RGB_Image.shape[2]
           
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
           
                 #################################################################################
                 #################################################################################

      if __name__ == "__main__":
          main()           

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def gaussian():

  try:

      def main():

                 sysargv2  = input("Enter the Color Image for GB -->")
           
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
                 b, g, r = image_data[2, :, :], image_data[1, :, :], image_data[0, :, :] 
           
           
                 # Save each channel as a separate file
                 fits.writeto(f'channel_0_64bit.fits', b.astype(np.float64), header, overwrite=True)
                 fits.writeto(f'channel_1_64bit.fits', g.astype(np.float64), header, overwrite=True)
                 fits.writeto(f'channel_2_64bit.fits', r.astype(np.float64), header, overwrite=True)
           
                 sysargv2  = "channel_0_64bit.fits"
                 sysargv2g  = "channel_1_64bit.fits"
                 sysargv2r  = "channel_2_64bit.fits"
                 sysargv4a  = input("Enter the gausian blur 1.313, 2 etc.  -->")
                 sysargv4 = float(sysargv4a)
           
                 def apply_gaussian_blur(image, sigma):
                   # Apply Gaussian blur with specified sigma
                   blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
                   return blurred_image
           
                 def read_fits(file):
                       hdul = fits.open(file)
                       header = hdul[0].header
                       data = hdul[0].data
                       hdul.close()
                       return data, header
           
                   # Load the FITS blue file
                 file1 = sysargv2
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
           
                 # Normalize the image data
                 image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)
           
                 # Apply the Gaussian blur
                 sigma = sysargv4  # Standard deviation in X and Y directions
                 blurred_image = apply_gaussian_blur(image_data, sigma)
           
                 # Save the final image as a FITS file
                 fits.writeto(f'channel_0_64bitGB.fits', blurred_image.astype(np.float64), header, overwrite=True)
                 # Load the FITS green file
                 file1 = sysargv2g
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
           
                 # Normalize the image data
                 image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)
           
                 # Apply the Gaussian blur
                 sigma = sysargv4  # Standard deviation in X and Y directions
                 blurred_image = apply_gaussian_blur(image_data, sigma)
           
                 # Save the final image as a FITS file
                 fits.writeto(f'channel_1_64bitGB.fits', blurred_image.astype(np.float64), header, overwrite=True)    # Load the FITS red file
                 file1 = sysargv2r
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
                 # Normalize the image data
                 image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)
           
                 # Apply the Gaussian blur
                 sigma = sysargv4  # Standard deviation in X and Y directions
                 blurred_image = apply_gaussian_blur(image_data, sigma)
           
                 # Save the final image as a FITS file
                 fits.writeto(f'channel_2_64bitGB.fits', blurred_image.astype(np.float64), header, overwrite=True)
           
                 sysargv1  = "channel_0_64bitGB.fits"
                 sysargv2  = "channel_1_64bitGB.fits"
                 sysargv3  = "channel_2_64bitGB.fits"
                 sysargv4  = "channel_RGB_64bitGB.fits"
           
                 old_data, old_header = read_fits(file1)
                 old_data = old_data.astype(np.float64)
              
                 # Function to read FITS file and return data
                 def read_fits(file):
                   with fits.open(file, mode='update') as hdul:#
                     data = hdul[0].data
                     hdul.close()
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
                 header['NAXIS1'] = RGB_Image.shape[0]
                 header['NAXIS2'] = RGB_Image.shape[1]
                 header['NAXIS3'] = RGB_Image.shape[2]
           
                 # Ensure the data type is correct 
                 newRGB_Image = RGB_Image.astype(np.float64)
           
                 print("newRGB_Image shape:", newRGB_Image.shape)
           
                 sysargv4 = "channel_RGB_64bitGB.fits"
                 fits.writeto( sysargv4, newRGB_Image, overwrite=True)
                 # Save the RGB image as a new FITS file with the correct header
                 hdu = fits.PrimaryHDU(data=newRGB_Image, header=header)
                 hdu.writeto(sysargv4, overwrite=True)
           
                 # Read and verify the saved FITS file
                 with fits.open(sysargv4) as hdul:
                   data = hdul[0].data
           
                 # Verify the saved RGB image
                 print("Verified image shape:", data.shape)
           
                 currentDirectory = os.path.abspath(os.getcwd())
           
                 # Define the file to delete
                 file_to_delete = "channel_0_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_1_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_2_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_0_64bitGB.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_1_64bitGB.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                       print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_2_64bitGB.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")

      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def FFT():

  try:

      def main():

                 sysargv2  = input("Enter the Color Image for FFT  -->")
           
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
                 b, g, r = image_data[2, :, :], image_data[1, :, :], image_data[0, :, :] 
           
           
                 # Save each channel as a separate file
                 fits.writeto(f'channel_0_64bit.fits', b.astype(np.float64), header, overwrite=True)
                 fits.writeto(f'channel_1_64bit.fits', g.astype(np.float64), header, overwrite=True)
                 fits.writeto(f'channel_2_64bit.fits', r.astype(np.float64), header, overwrite=True)
           
                 sysargv2  = "channel_0_64bit.fits"
                 sysargv2g  = "channel_1_64bit.fits"
                 sysargv2r  = "channel_2_64bit.fits"
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
           
                 def read_fits(file):
                       hdul = fits.open(file)
                       header = hdul[0].header
                       data = hdul[0].data
                       hdul.close()
                       return data, header
           
                   # Load the FITS blue file
                 file1 = sysargv2
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
           
                 # Normalize the image data
                 image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)
           
                 # Apply high-pass filter
                 filtered_image = high_pass_filter(image_data, cutoff=(int(sysargv4)/int(sysargv6)), weight=(int(sysargv5)/int(sysargv6)))
           
                 # Apply feathering
                 final_image = feather_image(filtered_image, radius=int(sysargv7), distance=int(sysargv8))
           
                 # Save the final image as a FITS file
                 fits.writeto(f'channel_0_64bitfft.fits', final_image.astype(np.float64), header, overwrite=True)
                 # Load the FITS green file
                 file1 = sysargv2g
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
           
                 # Normalize the image data
                 image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)
           
                 # Apply high-pass filter
                 filtered_image = high_pass_filter(image_data, cutoff=(int(sysargv4)/int(sysargv6)), weight=(int(sysargv5)/int(sysargv6)))
           
                 # Apply feathering
                 final_image = feather_image(filtered_image, radius=int(sysargv7), distance=int(sysargv8))
           
                 # Save the final image as a FITS file
                 fits.writeto(f'channel_1_64bitfft.fits', final_image.astype(np.float64), header, overwrite=True)    # Load the FITS red file
                 file1 = sysargv2r
                 image_data, header = read_fits(file1)
                 image_data = image_data.astype(np.float64)
                 # Normalize the image data
                 image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 1)).astype(np.float32)
           
                 # Apply high-pass filter
                 filtered_image = high_pass_filter(image_data, cutoff=(int(sysargv4)/int(sysargv6)), weight=(int(sysargv5)/int(sysargv6)))
           
                 # Apply feathering
                 final_image = feather_image(filtered_image, radius=int(sysargv7), distance=int(sysargv8))
           
                 # Save the final image as a FITS file
                 fits.writeto(f'channel_2_64bitfft.fits', final_image.astype(np.float64), header, overwrite=True)
           
                 sysargv1  = "channel_0_64bitfft.fits"
                 sysargv2  = "channel_1_64bitfft.fits"
                 sysargv3  = "channel_2_64bitfft.fits"
                 sysargv4  = "channel_RGB_64bitfft.fits"
           
                 old_data, old_header = read_fits(file1)
                 old_data = old_data.astype(np.float64)
              
                 # Function to read FITS file and return data
                 def read_fits(file):
                   with fits.open(file, mode='update') as hdul:#
                     data = hdul[0].data
                     hdul.close()
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
                 header['NAXIS1'] = RGB_Image.shape[0]
                 header['NAXIS2'] = RGB_Image.shape[1]
                 header['NAXIS3'] = RGB_Image.shape[2]
           
                 # Ensure the data type is correct 
                 newRGB_Image = RGB_Image.astype(np.float64)
           
                 print("newRGB_Image shape:", newRGB_Image.shape)
           
                 sysargv4 = "channel_RGB_64bitfft.fits"
                 fits.writeto( sysargv4, newRGB_Image, overwrite=True)
                 # Save the RGB image as a new FITS file with the correct header
                 hdu = fits.PrimaryHDU(data=newRGB_Image, header=header)
                 hdu.writeto(sysargv4, overwrite=True)
           
                 # Read and verify the saved FITS file
                 with fits.open(sysargv4) as hdul:
                   data = hdul[0].data
           
                 # Verify the saved RGB image
                 print("Verified image shape:", data.shape)
           
                 currentDirectory = os.path.abspath(os.getcwd())
           
                 # Define the file to delete
                 file_to_delete = "channel_0_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_1_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_2_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_0_64bitfft.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_1_64bitfft.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                       print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_2_64bitfft.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def  LrDeconv():

  try:
 
      def richardson_lucy(image, psf, iterations=30):
          """
          Perform Richardson–Lucy deconvolution.
    
          Parameters:
            image : 2D ndarray
                The observed (blurred) image.
            psf : 2D ndarray
                The point-spread function; should be normalized.
            iterations : int
                Number of iterations to perform.
    
          Returns:
            deconv : 2D ndarray
                The deconvolved image.
          """
          image = image.astype(np.float64)
          im_est = image.copy()  # initial estimate
          psf_mirror = psf[::-1, ::-1]  # mirror PSF

          for i in range(iterations):
              conv_est = convolve_fft(im_est, psf, normalize_kernel=True)
              conv_est[conv_est == 0] = 1e-7  # Avoid divide-by-zero
              relative_blur = image / conv_est
              correction = convolve_fft(relative_blur, psf_mirror, normalize_kernel=True)
              im_est *= correction
          return im_est

      def extract_psf(image, position, size):
          """
          Extract a PSF from the image by making a cutout around a bright source.
    
          Parameters:
            image : 2D ndarray
                Input image.
            position : tuple
                (x, y) coordinates (in pixels) of the center of the bright star.
            size : int or tuple
                Size of the cutout; can be an integer or a (width, height) tuple.
    
          Returns:
            psf : 2D ndarray
                The normalized PSF extracted from the image.
          """
          cutout = Cutout2D(image, position, size)
          psf = cutout.data.copy()
          psf -= np.median(psf)
          psf[psf < 0] = 0
          psf /= psf.sum()
          return psf

      def main():
          sysargv1  = input("Enter the Image name  -->")
          sysargv2  = input("Enter the name of deconvoluted image name  -->")

          # Load the observed image from a FITS file.
          input_file = sysargv1  # Replace with your actual FITS file.
          with fits.open(input_file) as hdul:
              image = hdul[0].data

          # Check if the image is color.
          # Since we assume a color image has shape (3, height, width)
          if image.ndim == 3 and image.shape[0] == 3:
              is_color = True
          else:
              is_color = False

          # Visualization of the image to help pick the PSF region.
          if not is_color:
              norm = simple_norm(image, 'sqrt', percent=99)
              plt.figure(figsize=(6, 5))
              plt.imshow(image, norm=norm, origin='lower', cmap='gray')
              plt.title("Input Grayscale Image")
              plt.colorbar()
              plt.show()
          else:
              plt.figure(figsize=(6, 5))
              # Rearrange image from (3, height, width) to (height, width, 3) for display.
              plt.imshow(np.moveaxis(image, 0, -1), origin='lower')
              plt.title("Input Color Image")
              plt.show()

          # Choose PSF extraction method:
          #   Option A: Use an analytical PSF (Gaussian 2D kernel)
          #   Option B: Extract the PSF from the image.
          use_analytical_psf = input("Use analytical PSF [g] or extract from image [e]? ").strip().lower()
    
          if use_analytical_psf == 'g':
              sigma = 2.0           # Adjust sigma as needed
              kernel_size = 25      # Size of the PSF kernel
              gauss_kernel = Gaussian2DKernel(sigma, x_size=kernel_size, y_size=kernel_size)
              psf = gauss_kernel.array
              psf /= psf.sum()      # Normalize the PSF
              print("Using an analytical Gaussian PSF.")
    
          elif use_analytical_psf == 'e':
              x = float(input("Enter x coordinate of the PSF center: "))
              y = float(input("Enter y coordinate of the PSF center: "))
              size = float(input("Enter the size of the PSF cutout (in pixels): "))
              # For color images in shape (3, height, width), extract from the first channel.
              if is_color:
                  psf = extract_psf(image[0, :, :], (x, y), size)
              else:
                  psf = extract_psf(image, (x, y), size)
              print("Using the PSF extracted from the image.")
        
              plt.figure(figsize=(4, 4))
              plt.imshow(psf, origin='lower', cmap='viridis')
              plt.title("Extracted PSF")
              plt.colorbar()
              plt.show()
          else:
              print("Invalid option selected. Exiting.")
              return

          # Perform Richardson–Lucy deconvolution with the chosen PSF.
          iterations = int(input("Enter the number of iterations (e.g., 30): "))
    
          if not is_color:
              deconv_image = richardson_lucy(image, psf, iterations=iterations)
              output_file = 'deconvolved_image.fits'
              fits.writeto(output_file, deconv_image, overwrite=True)
              print("Deconvolved image saved to", output_file)
    
              plt.figure(figsize=(12, 5))
              plt.subplot(1, 2, 1)
              plt.imshow(image, origin='lower', cmap='gray', norm=simple_norm(image, 'sqrt', percent=99))
              plt.title("Original Grayscale Image")
              plt.colorbar()
    
              plt.subplot(1, 2, 2)
              plt.imshow(deconv_image, origin='lower', cmap='gray', norm=simple_norm(deconv_image, 'sqrt', percent=99))
              plt.title("Deconvolved Image")
              plt.colorbar()
              plt.tight_layout()
              plt.show()
    
          else:
              # For color images assuming shape (3, height, width)
              deconv_image = np.empty_like(image)
              print("Processing each channel (assuming image shape is (3, height, width)):")
              # Iterate over axis 0 (channels)
              for c in range(image.shape[0]):
                  print(f"Deconvolving channel {c}...")
                  deconv_image[c, :, :] = richardson_lucy(image[c, :, :], psf, iterations=iterations)
              output_file = sysargv2
              fits.writeto(output_file, deconv_image, overwrite=True)
              print("Deconvolved color image saved to", output_file)
    
              plt.figure(figsize=(12, 5))
              plt.subplot(1, 2, 1)
              # Display original color image by moving axis 0 to the end.
              plt.imshow(np.moveaxis(image, 0, -1), origin='lower')
              plt.title("Original Color Image")
    
              plt.subplot(1, 2, 2)
              plt.imshow(np.moveaxis(deconv_image, 0, -1), origin='lower')
              plt.title("Deconvolved Color Image")
              plt.tight_layout()
              plt.show()

      if __name__ == '__main__':
          main()
  
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()


  return sysargv1
  menue()

def erosion():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def jpgcomp():

  try:

      def main():

                 sysargv2  = input("Enter the input file name --> ")
                 sysargv3  = input("Enter number percent to compress to (10) ")
                 sysargv4  = input("Enter the output file name --> ")
           
                 # Read the image as grayscale
                 image = cv2.imread(sysargv2)
                 cv2.imwrite(sysargv4, image, [cv2.IMWRITE_JPEG_QUALITY, int(sysargv3)])

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()


def dilation():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def imgcrop1():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def imghiststretch():

  try:

      # ------------------------------
      # Histogram Specification Functions
      # ------------------------------

      def rayleigh_specification(image, sigma):
          """
          Transform the image so its histogram matches a Rayleigh distribution.
    
          Inverse Rayleigh CDF: F⁻¹(p; σ) = σ * sqrt(-2 * ln(1-p))
          """
          vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
          cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
          epsilon = 1e-10
          safe_vals = np.clip(1 - cdf, epsilon, None)
          new_vals = sigma * np.sqrt(-2 * np.log(safe_vals))
          spec_img = new_vals[inv_idx].reshape(image.shape)
          return spec_img

      def gaussian_specification(image, mu, sigma):
          """
          Transform the image so its histogram matches a Gaussian distribution.
    
          Inverse Gaussian CDF: F⁻¹(p; μ,σ) = μ + σ * sqrt(2) * erfinv(2p - 1)
          """
          vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
          cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
          epsilon = 1e-10
          safe_cdf = np.clip(cdf, epsilon, 1 - epsilon)
          new_vals = mu + sigma * np.sqrt(2) * erfinv(2 * safe_cdf - 1)
          spec_img = new_vals[inv_idx].reshape(image.shape)
          return spec_img

      def uniform_specification(image, lower, upper):
          """
          Transform the image so its histogram is uniformly distributed.
    
          Inverse Uniform CDF: F⁻¹(p) = lower + p * (upper - lower)
          """
          vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
          cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
          new_vals = lower + cdf * (upper - lower)
          spec_img = new_vals[inv_idx].reshape(image.shape)
          return spec_img

      def exponential_specification(image, lamb):
          """
          Transform the image so its histogram matches an exponential distribution.
    
          Inverse Exponential CDF: F⁻¹(p; λ) = - (1/λ) * ln(1 - p)
          """
          vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
          cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
          epsilon = 1e-10
          safe_vals = np.clip(1 - cdf, epsilon, None)
          new_vals = - (1 / lamb) * np.log(safe_vals)
          spec_img = new_vals[inv_idx].reshape(image.shape)
          return spec_img

      def lognormal_specification(image, mu, sigma_ln):
          """
          Transform the image so its histogram matches a lognormal distribution.
    
          Inverse Lognormal CDF: 
             F⁻¹(p; μ,σ) = exp( μ + σ * sqrt(2) * erfinv(2p - 1) )
          """
          vals, inv_idx, counts = np.unique(image, return_inverse=True, return_counts=True)
          cdf = np.cumsum(counts).astype(np.float64) / counts.sum()
          epsilon = 1e-10
          safe_cdf = np.clip(cdf, epsilon, 1 - epsilon)
          new_vals = np.exp(mu + sigma_ln * np.sqrt(2) * erfinv(2 * safe_cdf - 1))
          spec_img = new_vals[inv_idx].reshape(image.shape)
          return spec_img

      # ------------------------------
      # Main Routine
      # ------------------------------

      def main():
          # Load the input FITS image (update the path as needed)
          sysargv2  = input("Enter the image(fits Siril)  -->")
          input_fits_file = sysargv2
          with fits.open(input_fits_file) as hdul:
              image = hdul[0].data.astype(np.float64)
    
          # Choose the transformation type.
          prompt = (
              "Choose histogram specification type:\n"
              "  (r) Rayleigh\n"
              "  (g) Gaussian\n"
              "  (u) Uniform\n"
              "  (e) Exponential\n"
              "  (l) Lognormal\n"
              "Enter one of r, g, u, e, l: "
          )
          transform_type = input(prompt).strip().lower()
    
          # Compute parameters from the image or ask the user.
          if transform_type == 'r':
              sigma_val = np.std(image)
              print(f"Applying Rayleigh specification with sigma = {sigma_val:.3f}")
              specified_image = rayleigh_specification(image, sigma_val)
              output_fits_file = 'image_rayleigh_specified.fits'
    
          elif transform_type == 'g':
              mu_val = np.mean(image)
              sigma_val = np.std(image)
              print(f"Applying Gaussian specification with mu = {mu_val:.3f} and sigma = {sigma_val:.3f}")
              specified_image = gaussian_specification(image, mu_val, sigma_val)
              output_fits_file = 'image_gaussian_specified.fits'
    
          elif transform_type == 'u':
              # For uniform, specify lower and upper limits.
              # For example, map to the range of the original image.
              lower = float(input("Enter lower bound (e.g., 0): "))
              upper = float(input("Enter upper bound (e.g., 1): "))
              print(f"Applying Uniform specification with lower = {lower} and upper = {upper}")
              specified_image = uniform_specification(image, lower, upper)
              output_fits_file = 'image_uniform_specified.fits'
    
          elif transform_type == 'e':
              lamb = float(input("Enter lambda (e.g., 0.1): "))
              print(f"Applying Exponential specification with lambda = {lamb}")
              specified_image = exponential_specification(image, lamb)
              output_fits_file = 'image_exponential_specified.fits'
    
          elif transform_type == 'l':
              mu_val = float(input("Enter mu for lognormal (e.g., 0): "))
              sigma_ln = float(input("Enter sigma for lognormal (e.g., 0.5): "))
              print(f"Applying Lognormal specification with mu = {mu_val} and sigma = {sigma_ln}")
              specified_image = lognormal_specification(image, mu_val, sigma_ln)
              output_fits_file = 'image_lognormal_specified.fits'
    
          else:
              print("Invalid choice. Exiting.")
              return
    
          # Save the specified image to a new FITS file.
          fits.writeto(output_fits_file, specified_image, overwrite=True)
          print(f"Specified image saved as {output_fits_file}")
    
          # Plot histograms for comparison
          fig, axes = plt.subplots(1, 2, figsize=(12, 5))
          axes[0].hist(image.ravel(), bins=256, color='blue', histtype='step')
          axes[0].set_title("Original Image Histogram")
          axes[0].set_xlabel("Intensity")
          axes[0].set_ylabel("Frequency")
    
          axes[1].hist(specified_image.ravel(), bins=256, color='red', histtype='step')
    
          title_dict = {
              'r': "Rayleigh Specified Histogram",
              'g': "Gaussian Specified Histogram",
              'u': "Uniform Specified Histogram",
              'e': "Exponential Specified Histogram",
              'l': "Lognormal Specified Histogram"
          }
          axes[1].set_title(title_dict.get(transform_type, "Specified Histogram"))
          axes[1].set_xlabel("Intensity")
          axes[1].set_ylabel("Frequency")
    
          plt.tight_layout()
          plt.show()

      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def gif():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def video():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()
  
  return sysargv1
  menue()

def alingimg():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def gamma():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def add2images():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def subtract2images():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def clahe():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()
  
  return sysargv1
  menue()

def pm_vector_line():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()
  
  return sysargv1
  menue()

def hist_match():

  try:

      def main():

                 sysargv1  = input("Enter the reference Image  -->")
                 sysargv3  = input("Enter the Image  -->")
                 sysargv4  = input("Enter the filename of the added images to save  -->")
           
                 # Load example images
                 reference = cv2.imread(sysargv1, -1)
                 image = cv2.imread(sysargv3, -1)
           
                 # Perform histogram matching
                 matched = match_histograms(image, reference, channel_axis=-1)
           
                 cv2.imwrite(sysargv4, matched)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def distance():

  try:

      def main():

                 sysargv1  = float(input("parallax angle in milliarcseconds  -->"))
           
                 distancepar = 1 / (sysargv1 / 1000)
                 distanceltyr = 3.26 * distancepar
                 print('distance parsecs', distancepar)
                 print('distance light year', distanceltyr)
           
      if __name__ == "__main__":
          main()           
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def edgedetect():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def mosaic():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def imgqtr():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def CpyOldHdr():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def binimg():

  try:

      def main():
  
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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()


def autostr():

  try:

      def smh_stretch(data, lower_percent=0.5, upper_percent=99.5):
        """
        Perform a shadows/midtones/highlights (SMH) style histogram stretch.
    
        This function computes:
          - low: the shadow threshold (at the lower_percent percentile),
          - high: the highlight threshold (at the upper_percent percentile),
          - med: the midtone (50th percentile),
        and computes gamma such that the normalized midtone maps to 0.5, i.e.,
             ( (med - low)/(high - low) )^gamma = 0.5.
    
        Then, the data is normalized and the gamma correction applied.
    
        Parameters:
          data          : NumPy array containing the image pixel values.
          lower_percent : Lower percentile for shadows (default is 0.5).
          upper_percent : Upper percentile for highlights (default is 99.5).
    
        Returns:
          stretched : The processed image (values in [0, 1]).
          low       : The computed shadow threshold.
          med       : The computed midtone value.
          high      : The computed highlight threshold.
          gamma     : The gamma value applied.
        """
        # Compute shadow and highlight thresholds from percentiles.
        low = np.percentile(data, lower_percent)
        high = np.percentile(data, upper_percent)
    
        # Compute the median (midtone) of the image.
        med = np.percentile(data, 50)
        # Ensure the midtone lies within [low, high]
        med = np.clip(med, low, high)
    
        # Avoid division by zero in case high equals low.
        if high == low:
          return np.zeros_like(data), low, med, high, 1.0
    
        # Normalize the midtone between 0 and 1.
        m = (med - low) / (high - low)
        # Avoid a zero or extremely small value for m.
        if m <= 0:
            m = 0.001
    
        # Determine gamma such that m^gamma = 0.5.
        gamma = np.log(0.5) / np.log(m)
    
        # Normalize data to the range [0, 1] based on computed thresholds.
        normalized = (data - low) / (high - low)
        normalized = np.clip(normalized, 0, 1)
    
        # Apply the gamma correction.
        stretched = normalized ** gamma
    
        return stretched, low, med, high, gamma

      def main():

                 sysargv3  = input("Enter file name of image to auto_str  -->")
                 sysargv4  = input("Enter file name of output image -->")
                 lower_percent  = 0.5
                 upper_percent  = 99.5
           
                 with fits.open(sysargv3) as hdul:
                     hdu = hdul[0]
                     data = hdu.data.astype(np.float64)
                     header = hdu.header
           
                 # Apply the SMH stretch.
                 stretched, low, med, high, gamma = smh_stretch(data, lower_percent, upper_percent)
               
                 # Print out the computed parameters.
                 print(f"Shadow threshold (lower {lower_percent}th percentile): {low}")
                 print(f"Midtone (50th percentile): {med}")
                 print(f"Highlight threshold (upper {upper_percent}th percentile): {high}")
                 print(f"Applied gamma: {gamma}\n")
               
                 # Update the header to record stretch information.
                 header['STRETCH'] = ('SMH', 'Shadows/Midtones/Highlights histogram stretch')
                 header['LOWPCT'] = (lower_percent, 'Lower percentile for shadow threshold')
                 header['HIGPCT'] = (upper_percent, 'Upper percentile for highlight threshold')
                 header['SHADOW'] = (low, 'Shadow threshold value')
                 header['MIDTONE'] = (med, 'Midtone (median) value')
                 header['HIGHLT'] = (high, 'Highlight threshold value')
                 header['GAMMA'] = (gamma, 'Gamma value used')
               
                 # Save the stretched image as a new FITS file.
                 hdu_new = fits.PrimaryHDU(data=stretched, header=header)
                 hdu_new.writeto(sysargv4, overwrite=True)
               
                 print(f"Stretched FITS image has been saved to: {sysargv4}")
           
      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def LocAdapt():

  try:
      
      def main():
          
                 sysargv2  = input("Enter the Color Image for LA -->")
                 sysargv6  = input("Enter neighborhood_size(15) -->")
                 sysargv6int  = int(sysargv6)
           
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
                 image_data = image_data.astype(float)
           
                 # Split the color image into its individual channels
                 #b, g, r = cv2.split(image_data)
                 b, g, r = image_data[2, :, :], image_data[1, :, :], image_data[0, :, :] 
           
           
                 # Save each channel as a separate file
                 fits.writeto(f'channel_0_64bit.fits', b.astype(float), header, overwrite=True)
                 fits.writeto(f'channel_1_64bit.fits', g.astype(float), header, overwrite=True)
                 fits.writeto(f'channel_2_64bit.fits', r.astype(float), header, overwrite=True)
           
                 sysargv2  = "channel_0_64bit.fits"
                 sysargv2g  = "channel_1_64bit.fits"
                 sysargv2r  = "channel_2_64bit.fits"
           
                 sysargv4a  = input("Enter the Contrast as (50) with no decimal  -->")
                 sysargv4 = int(sysargv4a)
                 sysargv4b  = input("Enter the feather_distance as (5) with no decimal  -->")
                 sysargv4c = int(sysargv4b)
           
                 def compute_optimum_contrast_percentage(image, target_std):
                   """
                   Compute an optimum contrast percentage based on the image's current standard deviation.
               
                   Parameters:
                   image (numpy.ndarray): The input image as a 2D NumPy array.
                   target_std (float): A chosen target standard deviation.
                                      The optimum contrast factor is computed as target_std / current_std.
               
                   Returns:
                       float: The computed optimum contrast percentage.
                       For example, a return value of 150 means 150% (i.e., a factor of 1.5).
                   """
                   current_std = np.std(image)
                   if current_std == 0:
                   # For a flat image, no change is applied.
                     return 100.0
                   contrast_factor = target_std / current_std
                   contrast_percentage = contrast_factor * 100.0
                   return contrast_percentage
           
                 def compute_optimum_feather_distance(image, neighborhood_size, factor=1.0):
                   """
                   Compute an optimum feather distance based on the local standard deviation of the image.
                
                   This function computes the local mean and local mean of the squared image within a window,
                   then derives the local standard deviation. The median of these local standard deviations
                   is taken as the baseline optimum feather distance, which can be adjusted by a scaling factor.
                       Parameters:
                       image (numpy.ndarray): The input image as a 2D NumPy array.
                       neighborhood_size (int): The window size used for calculating local statistics.
                       factor (float): A multiplicative factor to adjust the optimum feather distance (default is 1.0).
               
                   Returns:
                       float: The computed optimum feather distance.
                   """
                   # Adjust the window size by reducing it by 1 (ensuring an effective kernel size of at least 1)
                   adjusted_size = max(1, neighborhood_size - 1)
                   kernel = np.ones((adjusted_size, adjusted_size), dtype=np.float32) / (adjusted_size * adjusted_size)
               
                   # Compute the local mean using a convolution
                   local_mean = convolve(image, kernel, mode='reflect')
               
                   # Compute the local mean of the squared image
                   local_mean_sq = convolve(image**2, kernel, mode='reflect')
               
                   # Calculate local standard deviation: sqrt(local_mean_sq - local_mean**2)
                   local_std = np.sqrt(np.abs(local_mean_sq - local_mean**2))
               
                   # Use the median of the local standard deviations as the baseline, then scale.
                   optimum_feather = factor * np.median(local_std)
                   return optimum_feather
           
               #--------------------------------------------------------------------------------------
                 def contrast_filter(image, neighborhood_size, contrast_factor, feather_distance):
                   """
                   Apply a local adaptive contrast filter with feathering.
               
                   This function computes a local mean (using an adjusted neighborhood size) and then
                   enhances the contrast by scaling the deviation from the local mean. Feathering is implemented 
                   by blending the enhanced image with the original image in areas where the local standard deviation 
                   is below a specified feather_distance threshold.
                     
                   Parameters:
                       image (ndarray): 2D array of the input image.
                       neighborhood_size (int): Size of the window for computing local statistics.
                       contrast_factor (float): Factor to scale the local contrast.
                       feather_distance (float): Threshold (in intensity units) that defines the feathering effect.
                                             Regions with local standard deviation lower than this value are blended
                                             with the original image.
               
                   Returns:
                         final_image (ndarray): The contrast-enhanced image.
                   """
                   # Adjust the kernel size (reducing the provided neighborhood size by 1, but never below 1)
                   adjusted_size = max(1, neighborhood_size - 1)
                   kernel = np.ones((adjusted_size, adjusted_size), dtype=float)
                   kernel /= kernel.size
           
                   # Compute the local mean using convolution
                   local_mean = convolve(image, kernel, mode='reflect')
               
                   # Compute the enhanced image using the standard contrast adjustment formula:
                   #   enhanced = (image - local_mean) * contrast_factor + local_mean
                   enhanced_image = (image - local_mean) * contrast_factor + local_mean
           
                   # For feathering, compute the local standard deviation:
                   # First, get the local mean of the squared image.
                   squared_image = np.square(image)
                   local_mean_squared = convolve(squared_image, kernel, mode='reflect')
                   # Then compute the standard deviation (making sure to safeguard against small negative values)
                   local_std = np.sqrt(np.abs(local_mean_squared - np.square(local_mean)))
               
                   # Create a feathering weight: when local_std is below feather_distance the weight is <1.
                   # This weight is 1 if local_std >= feather_distance.
                   weight = np.clip(local_std / feather_distance, 0, 1)
               
                   # Blend the original and enhanced images using the weight map:
                   #   In low-contrast regions (low local_std) the enhanced effect is partially reduced.
                   final_image = weight * enhanced_image + (1 - weight) * image
               
                   return final_image
           
                 # -------------------------
                 # Hard-coded Parameters
                 # -------------------------
                 input_fits  = 'channel_0_64bit.fits'   # Name/path for the input FITS file.
                 output_fits = 'channel_0_64bitLA.fits'  # Name/path for the output FITS file.
                 neighborhood_size = sysargv6int      # Size of the region used to compute local statistics.
                 target_std = int(sysargv4) 
                 # Factor by which to scale the local deviations.
                 feather_distance = int(sysargv4c)       # Feather threshold: lower values produce more feathering in smooth areas.
                 feather_factor = feather_distance
           
                 # -------------------------
                 # Read the FITS File
                 # -------------------------
                 hdulist = fits.open(input_fits)
                 # Convert image data to float32 for accurate arithmetic and convolution
                 image_data = hdulist[0].data.astype(float)
                 header = hdulist[0].header
           
                 optimum_contrast_percentage = compute_optimum_contrast_percentage(image_data, target_std)
                 contrast_factor = optimum_contrast_percentage * 100
                 optimum_feather_distance = compute_optimum_feather_distance(image_data, neighborhood_size, feather_factor)
                 feather_distance = optimum_feather_distance / 100
           
                 # -------------------------
                 # Apply the Contrast Filter with Feathering
                 # -------------------------
                 enhanced_data = contrast_filter(image_data, neighborhood_size, contrast_factor, feather_distance)
           
                 # -------------------------
                 # Write the Enhanced Image to a New FITS File
                 # -------------------------
                 hdu = fits.PrimaryHDU(data=enhanced_data, header=header)
                 hdu.writeto(output_fits, overwrite=True)
                 hdulist.close()
           
                 print(f"Enhanced FITS file with feathering saved as {output_fits}")
           
                 # -------------------------
                 # Hard-coded Parameters
                 # -------------------------
                 input_fits  = 'channel_1_64bit.fits'   # Name/path for the input FITS file.
                 output_fits = 'channel_1_64bitLA.fits'  # Name/path for the output FITS file.
                 neighborhood_size = sysargv6int       # Size of the region used to compute local statistics.
                 target_std = int(sysargv4) 
                 # Factor by which to scale the local deviations.
                 feather_distance = int(sysargv4c)       # Feather threshold: lower values produce more feathering in smooth 
                 feather_factor = feather_distance
           
                 # -------------------------
                 # Read the FITS File
                 # -------------------------
                 hdulist = fits.open(input_fits)
                 # Convert image data to float64 for accurate arithmetic and convolution
                 image_data = hdulist[0].data.astype(float)
                 header = hdulist[0].header
           
                 optimum_contrast_percentage = compute_optimum_contrast_percentage(image_data, target_std)
                 contrast_factor = optimum_contrast_percentage * 100
                 optimum_feather_distance = compute_optimum_feather_distance(image_data, neighborhood_size, feather_factor)
                 feather_distance = optimum_feather_distance / 100
           
                 # -------------------------
                 # Apply the Contrast Filter with Feathering
                 # -------------------------
                 enhanced_data = contrast_filter(image_data, neighborhood_size, contrast_factor, feather_distance)
           
                 # -------------------------
                 # Write the Enhanced Image to a New FITS File
                 # -------------------------
                 hdu = fits.PrimaryHDU(data=enhanced_data, header=header)
                 hdu.writeto(output_fits, overwrite=True)
                 hdulist.close()
           
                 print(f"Enhanced FITS file with feathering saved as {output_fits}")
           
                 # -------------------------
                 # Hard-coded Parameters
                 # -------------------------
                 input_fits  = 'channel_2_64bit.fits'   # Name/path for the input FITS file.
                 output_fits = 'channel_2_64bitLA.fits'  # Name/path for the output FITS file.
                 neighborhood_size = sysargv6int       # Size of the region used to compute local statistics.
                 target_std = int(sysargv4) 
                 # Factor by which to scale the local deviations.
                 feather_distance = int(sysargv4c)       # Feather threshold: lower values produce more feathering in smooth 
                 feather_factor = feather_distance
           
                 # -------------------------
                 # Read the FITS File
                 # -------------------------
                 hdulist = fits.open(input_fits)
                 # Convert image data to float64 for accurate arithmetic and convolution
                 image_data = hdulist[0].data.astype(float)
                 header = hdulist[0].header
           
                 optimum_contrast_percentage = compute_optimum_contrast_percentage(image_data, target_std)
                 contrast_factor = optimum_contrast_percentage * 100
                 optimum_feather_distance = compute_optimum_feather_distance(image_data, neighborhood_size, feather_factor)
                 feather_distance = optimum_feather_distance / 100
           
                 # -------------------------
                 # Apply the Contrast Filter with Feathering
                 # -------------------------
                 enhanced_data = contrast_filter(image_data, neighborhood_size, contrast_factor, feather_distance)
           
                 # -------------------------
                 # Write the Enhanced Image to a New FITS File
                 # -------------------------
                 hdu = fits.PrimaryHDU(data=enhanced_data, header=header)
                 hdu.writeto(output_fits, overwrite=True)
                 hdulist.close()
           
                 print(f"Enhanced FITS file with feathering saved as {output_fits}")
           
               #--------------------------------------------------------------------------------------
           
                 sysargv1  = "channel_0_64bitLA.fits"
                 sysargv2  = "channel_1_64bitLA.fits"
                 sysargv3  = "channel_2_64bitLA.fits"
                 sysargv4  = "channel_RGB_64bitLA.fits"
           
                 old_data, old_header = read_fits(file1)
                 old_data = old_data.astype(float)
              
                 # Function to read FITS file and return data
                 def read_fits(file):
                   with fits.open(file, mode='update') as hdul:#
                     data = hdul[0].data
                     hdul.close()
                   return data
           
                 # Read the FITS files
                 file1 = sysargv1
                 file2 = sysargv2
                 file3 = sysargv3
           
                 # Read the image data from the FITS file
                 blue = read_fits(file1)
                 green = read_fits(file2)
                 red = read_fits(file3)
           
                 blue = blue.astype(float)
                 green = green.astype(float)
                 red = red.astype(float)
           
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
                 header['NAXIS1'] = RGB_Image.shape[0]
                 header['NAXIS2'] = RGB_Image.shape[1]
                 header['NAXIS3'] = RGB_Image.shape[2]
           
                 # Ensure the data type is correct 
                 newRGB_Image = RGB_Image.astype(float)
           
                 print("newRGB_Image shape:", newRGB_Image.shape)
           
                 sysargv4 = "channel_RGB_64bitLA.fits"
                 fits.writeto( sysargv4, newRGB_Image, overwrite=True)
                 # Save the RGB image as a new FITS file with the correct header
                 hdu = fits.PrimaryHDU(data=newRGB_Image, header=header)
                 hdu.writeto(sysargv4, overwrite=True)
           
                 # Read and verify the saved FITS file
                 with fits.open(sysargv4) as hdul:
                   data = hdul[0].data
           
                 # Verify the saved RGB image
                 print("Verified image shape:", data.shape)
           
                 currentDirectory = os.path.abspath(os.getcwd())
           
                 # Define the file to delete
                 file_to_delete = "channel_0_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_1_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_2_64bit.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_0_64bitLA.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_1_64bitLA.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                       print(f"File '{file_to_delete}' does not exist.")
           
                 # Define the file to delete
                 file_to_delete = "channel_2_64bitLA.fits"
                 file_to_delete = (os.path.join(currentDirectory, file_to_delete))
                 # Check if the file exists
                 if os.path.exists(file_to_delete):
                     os.remove(file_to_delete)  # Delete the file
                     print(f"File '{file_to_delete}' has been deleted.")
                 else:
                     print(f"File '{file_to_delete}' does not exist.")


      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def WcsOvrlay():

  try:

      def main():

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

      if __name__ == "__main__":
          main()


  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def WcsStack():

  try:
      
      def main():

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
                 ref_data = ref_hdul[0].data.astype(float)  # Ensure float64 precision
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
                         target_data = hdul[0].data.astype(float)
           
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
                 stacked_image = np.mean(aligned_images, axis=0).astype(float)
           
                 # Save the stacked image to a new FITS file
                 stacked_hdu = fits.PrimaryHDU(stacked_image, header=ref_header)
                 stacked_hdu.writeto(sysargv3, overwrite=True)
           
                 print("Stacked image saved as " + sysargv3 )

      if __name__ == "__main__":
          main()

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def combinelrgb():

  try:

      def main():

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

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def MxdlAstap():

  try:

      def main():

                 sysargv0  = input("Enter MaxDl fits file for Astap  -->")
           
                   # Set mpmath precision to 50 decimal places
                 mp.dps = 50
           
                 # Open the FITS file in update mode so we can modify its header.
                 filename = sysargv0    # Replace with your FITS file name
                 hdul = fits.open(filename, mode="update")
                 hdr = hdul[0].header
           
                 # --- Step 1. Read the astrometry.net keywords ---
                 # These keywords come from the astrometry.net solution in the header.
                 objctra = hdr.get("OBJCTRA")      # e.g., '12 30 50.3'
                 objctdec = hdr.get("OBJCTDEC")     # e.g., '+12 23 13'
                 cdelt1  = hdr.get("CDELT1")        # e.g., 1.466251413027E-04
                 cdelt2  = hdr.get("CDELT2")        # e.g., 1.466251413027E-04
           
                 if objctra is None or objctdec is None or cdelt1 is None or cdelt2 is None:
                     raise ValueError("Required keywords OBJCTRA, OBJCTDEC, or CDELT* are missing.")
           
                 # --- Step 2. Convert center coordinates to degrees ---
                 # OBJCTRA is in "HH MM SS.s" [hourangle] and OBJCTDEC is in "DD MM SS".
                 ra_deg  = Angle(objctra, unit="hourangle").degree
                 dec_deg = Angle(objctdec, unit="deg").degree
           
                 # --- Step 3. Determine the reference pixel (typically the image center) ---
                 naxis1 = hdr.get("NAXIS1")
                 naxis2 = hdr.get("NAXIS2")
                 if naxis1 is None or naxis2 is None:
                     raise ValueError("Image dimensions (NAXIS1, NAXIS2) are missing.")
           
                 # Following the common convention, the reference pixel is placed at the center.
                 crpix1 = (naxis1 + 1) / 2.0
                 crpix2 = (naxis2 + 1) / 2.0
           
                 # --- Step 4. Calculate the new WCS parameters with high precision ---
                 # Convert cdelt1 to a high-precision mpmath float:
                 cdelt1_mp = mp.mpf(cdelt1)
                 cdelt2_mp = mp.mpf(cdelt2)
           
                 # Mimic a solved CD1_1 value (from an astrometric solution)
                 solved_cd1_1 = mp.mpf(-3.542246e-4)
                 # Compute the factor by which the effective scale changes:
                 scale_factor = abs(solved_cd1_1) / cdelt1_mp  # e.g., ~2.416
                 # New effective pixel scale (in deg/pixel) – it should be nearly the absolute value of solved_cd1_1.
                 scale = scale_factor * cdelt1_mp
           
                 # Assume a (solved) image rotation, e.g., 179.7°
                 rotation_deg = mp.mpf(179.7)
                 # Convert degrees to radians using mpmath:
                 theta = rotation_deg * (mp.pi / 180)
           
                 # Calculate the CD matrix components using high-precision mpmath functions:
                 cd1_1 = -scale * mp.cos(theta)
                 cd1_2 =  scale * mp.sin(theta)
                 cd2_1 =  scale * mp.sin(theta)
                 cd2_2 =  scale * mp.cos(theta)
           
                 # --- Step 5. Write the computed WCS keywords into the FITS header ---
                 hdr["CRPIX1"]   = crpix1
                 hdr["CRPIX2"]   = crpix2
                 hdr["CRVAL1"]   = ra_deg    # Fitted RA center in degrees
                 hdr["CRVAL2"]   = dec_deg   # Fitted DEC center in degrees
                 hdr["CROTA1"]   = float(rotation_deg)  # Convert high-precision number to float
                 hdr["CROTA2"]   = float(rotation_deg)  # Often nearly the same value as CROTA1
                 hdr["CD1_1"]    = float(cd1_1)
                 hdr["CD1_2"]    = float(cd1_2)
                 hdr["CD2_1"]    = float(cd2_1)
                 hdr["CD2_2"]    = float(cd2_2)
                 hdr["PLTSOLVD"] = True   # Flag indicating that the plate solution has been applied
                 # Additional WCS keywords
                 hdr["CTYPE1"]  = "RA---TAN"   # first parameter RA, projection TANgential   
                 hdr["CTYPE2"]  = "DEC--TAN"   # second parameter DEC, projection TANgential   
                 hdr["CUNIT1"]  = "deg"        # Unit of coordinates                            
                 hdr["EQUINOX"] = 2000.0       # Equinox of coordinates                         
           
           
                 # Save (flush) the changes and close the file.
                 hdul.flush()
                 hdul.close()
           
                 print("WCS keywords calculated and written to the FITS header.")


      if __name__ == "__main__":
          main()           

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()


def CentRatio():

  try:

      def main():

                 sysargv0  = input("Enter star1_image1 x  -->")
                 sysargv1  = input("Enter star1_image1 y  -->")
                 sysargv2  = input("Enter star2_image1 x  -->")
                 sysargv3  = input("Enter star2_image1 y  -->")
                 sysargv4  = input("Enter star1_image2 x  -->")
                 sysargv5  = input("Enter star1_image2 y  -->")
                 sysargv6  = input("Enter star2_image2 x  -->")
                 sysargv7  = input("Enter star2_image2 y  -->")
           
           
                 def euclidean_distance(x1, y1, x2, y2):
                   """
                   Calculate the Euclidean distance between two points.
                   """
                   return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
           
                 # Image One coordinates (in pixels)
                 star1_image1 = (float(sysargv0), float(sysargv1))
                 star2_image1 = (float(sysargv2), float(sysargv3))
           
                 # Calculate distance1 for image one
                 distance1 = euclidean_distance(star1_image1[0], star1_image1[1], star2_image1[0], star2_image1[1])
           
                 # Image Two coordinates (in pixels)
                 star1_image2 = (float(sysargv4), float(sysargv5))
                 star2_image2 = (float(sysargv6), float(sysargv7))
           
                 # Calculate distance2 for image two
                 distance2 = euclidean_distance(star1_image2[0], star1_image2[1], star2_image2[0], star2_image2[1])
           
                 # Calculate the ratio of distances
                 ratio = distance1 / distance2
           
                 # Print out the results
                 print(f"Distance in Image One: {distance1:.5f} pixels")
                 print(f"Distance in Image Two: {distance2:.5f} pixels")
                 print(f"Ratio (distance1 / distance2): {ratio:.5f}")

      if __name__ == "__main__":
          main()           

  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

  return sysargv1
  menue()

def HpMore():

  try:

      def main():

                 sysargv0  = input("Enter fits image  -->")
                 sysargv1  = input("Enter Hp output fits file  -->")
           
                 # Step 1: Read the color FITS file.
                 with fits.open(sysargv0) as hdul:
                     data = hdul[0].data
           
                 print("Original data shape:", data.shape)
           
                 # Ensure the data is in a 3D array representing three color channels.
                 # If the data shape is (ny, nx, 3), transpose it so that channels come first: (3, ny, nx).
                 if data.ndim == 3:
                     if data.shape[-1] == 3:  # shape is likely (ny, nx, 3)
                         data = np.transpose(data, (2, 0, 1))
                         print("Transposed data shape:", data.shape)
                     elif data.shape[0] == 3:
                         # Already in (3, ny, nx) order.
                         print("Data already in (channels, ny, nx) format.")
                     else:
                         raise ValueError("Unexpected color image shape. Expected channel count of 3.")
                 else:
                     raise ValueError("Input FITS does not appear to be a color (3 channel) image.")
           
                 # Step 2: Define the 5x5 high-pass (Laplacian-style) kernel.
                 # This kernel is designed to sum to zero: the negative weights in the outer parts subtract the local average,
                 # while the positive weights at and near the center emphasize high-frequency details.
                 hp_kernel_5x5 = np.array([
                     [-1, -1, -1, -1, -1],
                     [-1,  1,  2,  1, -1],
                     [-1,  2,  4,  2, -1],
                     [-1,  1,  2,  1, -1],
                     [-1, -1, -1, -1, -1]
                 ], dtype=float)
           
                 print("5x5 High-Pass Kernel:\n", hp_kernel_5x5)
           
                 # Step 3: Initialize an output array for the filtered data.
                 # We'll process each channel separately.
                 filtered_channels = np.empty_like(data)
           
                 # Loop over each channel and apply the range-restricted high-pass filter.
                 # For each channel, we compute the 5x5 convolution and then add its values back only where pixel values exceed a threshold.
                 for i in range(data.shape[0]):
                     channel = data[i]
               
                     # Apply the high-pass filter using 2D convolution.
                     # 'same' ensures the output matches the input dimensions,
                     # and 'symm' uses symmetric boundary handling.
                     highpass = convolve2d(channel, hp_kernel_5x5, mode='same', boundary='symm')
               
                     # Create a range restriction mask by choosing pixels above the 70th percentile.
                     threshold = np.percentile(channel, 70)
                     mask = channel > threshold
           
                     # Combine: add the high-pass detail to the original image only in pixels where the mask is True.
                     filtered_channel = channel.copy()
                     filtered_channel[mask] = channel[mask] + highpass[mask]
               
                     filtered_channels[i] = filtered_channel
           
                 # Step 4: Convert the filtered result back for visualization.
                 # Many plotting functions expect color images in (ny, nx, 3) order.
                 filtered_rgb = np.transpose(filtered_channels, (1, 2, 0))
                 original_rgb = np.transpose(data, (1, 2, 0))
           
           
                 # Step 6: Write the filtered image to a new FITS file.
                 # Here we save in the original channel-first ordering (3, ny, nx).
                 fits.writeto( sysargv1, filtered_channels, overwrite=True)

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()

def CombBgrAlIm():
  try:

      def main():

                 sysargv0  = input("Enter wildcard fits image*.fit  -->")
           
                 # List the filenames of your six reprojected FITS images.
                 # It is assumed that each image is on the same canvas and black (0) indicates no data.
           
           
                 # Collect all files with names starting with 'reproj' and ending with '.fits'
                 filenames = sorted(glob.glob("m16_*_pspcchannel_BGR_64bitAL.fits"))
           
                 print("Files found:", filenames)
           
           
                 # Open the first image to get the shape of the common canvas.
                 with fits.open(filenames[0]) as hdul:
                     shape = hdul[0].data.shape
           
                 # Initialize two arrays with the same shape:
                 #   composite: to accumulate the pixel values
                 #   weights: to count how many images contributed at each pixel
                 composite = np.zeros(shape, dtype=np.float32)
                 weights = np.zeros(shape, dtype=np.float32)
           
                 # Loop over each reprojected image
                 for fname in filenames:
                     with fits.open(fname) as hdul:
                         # Convert the image data to float32 for precision.
                         data = hdul[0].data.astype(np.float32)
               
                     # Define a mask for valid data; here we assume that a pixel value > 0 is a valid contribution.
                     # (If your images might have valid zero values, you might need to use a quality mask instead.)
                     mask = data > 0
               
                     # Only add the data where the image actually contributes.
                     composite[mask] += data[mask]
                     weights[mask] += 1
           
                 # Determine the maximum weight on the canvas—we assume this is the maximum number of overlapping images.
                 max_weight = np.max(weights)
                 print(f"Maximum contributions across the canvas: {max_weight}")
           
                 # Now compute the final composite image.
                 # For each pixel, first get the average (composite value divided by the weight) and then
                 # boost it to the level of maximum overlap.
                 # That is, for every pixel: final_pixel = (composite / weight) * sqrt(max_weight)
                 #
                 # This means that if a pixel is observed only once (weight==1) it will get multiplied by sqrt(max_weight),
                 # whereas a pixel observed max_weight times would be scaled by √(max_weight)/√(max_weight) = 1.
                 final = np.zeros_like(composite)
                 valid = weights > 0  # to avoid any division by zero issues
                 final[valid] = (composite[valid] / weights[valid]) * np.sqrt(max_weight)
           
                 # Save out the final image to a new FITS file.
                 hdu = fits.PrimaryHDU(final)
                 hdu.writeto('final_wgted_Al_Img.fits', overwrite=True)
           
                 print("Finished writing final_wgted_Al_Img.fits")

      if __name__ == "__main__":
          main()
           
  except Exception as e:
      print(f"An error occurred: {e}")
      print("Returning to the Main Menue...")
      return sysargv1
      menue()




def menue(sysargv1):
  sysargv1 = input("Enter \n>>1<< AffineTransform(3pts) >>2<< Mask an image >>3<< Mask Invert >>4<< Add2images(fit)  \n>>5<< Split tricolor >>6<< Combine Tricolor >>7<< Create Luminance(2ax) >>8<< Align2img \n>>9<< Plot_16-bit_img to 3d graph(2ax) >>10<< Centroid_Custom_filter(2ax) >>11<< UnsharpMask \n>>12<< FFT-(RGB) >>13<< Img-DeconvClr >>14<< Centroid_Custom_Array_loop(2ax) \n>>15<< Erosion(2ax) >>16<< Dilation(2ax) >>17<< DynamicRescale(2ax) >>18<< GausBlur  \n>>19<< DrCntByFileType >>20<< ImgResize >>21<< JpgCompress >>22<< subtract2images(fit)  \n>>23<< multiply2images >>24<< divide2images >>25<< max2images >>26<< min2images \n>>27<< imgcrop >>28<< imghiststretch >>29<< gif  >>30<< aling2img(2pts) >>31<< Video \n>>32<< gammaCor >>33<< ImgQtr >>34<< CpyOldHdr >>35<< DynReStr(RGB) \n>>36<< clahe >>37<< pm_vector_line >>38<< hist_match >>39<< distance >>40<< EdgeDetect \n>>41<< Mosaic(4) >>42<< BinImg >>43<< autostr >>44<< LocAdapt >>45<< WcsOvrlay \n>>46<< WcsStack >>47<< CombineLRGB >>48<< MxdlAstap >>49<< CentRatio >>50<< ResRngHp \n>>51<< CombBgrAlIm \n>>1313<< Exit --> ")
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
    currentDirectory = os.path.abspath(os.getcwd())

    # Define the file to delete
    file_to_delete = "img_enlarged_25x.fit"
    file_to_delete = (os.path.join(currentDirectory, file_to_delete))
    # Check if the file exists
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)  # Delete the file
        print(f"File '{file_to_delete}' has been deleted.")
    else:
        print(f"File '{file_to_delete}' does not exist.")

    currentDirectory = os.path.abspath(os.getcwd())

    # Define the file to delete
    file_to_delete = "channel_0_64bit.fits"
    file_to_delete = (os.path.join(currentDirectory, file_to_delete))
    # Check if the file exists
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)  # Delete the file
        print(f"File '{file_to_delete}' has been deleted.")
    else:
        print(f"File '{file_to_delete}' does not exist.")

    # Define the file to delete
    file_to_delete = "channel_1_64bit.fits"
    file_to_delete = (os.path.join(currentDirectory, file_to_delete))
    # Check if the file exists
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)  # Delete the file
        print(f"File '{file_to_delete}' has been deleted.")
    else:
        print(f"File '{file_to_delete}' does not exist.")

    # Define the file to delete
    file_to_delete = "channel_2_64bit.fits"
    file_to_delete = (os.path.join(currentDirectory, file_to_delete))
    # Check if the file exists
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)  # Delete the file
        print(f"File '{file_to_delete}' has been deleted.")
    else:
        print(f"File '{file_to_delete}' does not exist.")

    # Define the file to delete
    file_to_delete = "channel_RGB_64bit_binned_gamma_corrected_drs_B.fit"
    file_to_delete = (os.path.join(currentDirectory, file_to_delete))
    # Check if the file exists
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)  # Delete the file
        print(f"File '{file_to_delete}' has been deleted.")
    else:
        print(f"File '{file_to_delete}' does not exist.")

    # Define the file to delete
    file_to_delete = "channel_RGB_64bit_binned_gamma_corrected_drs_G.fit"
    file_to_delete = (os.path.join(currentDirectory, file_to_delete))
    # Check if the file exists
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)  # Delete the file
        print(f"File '{file_to_delete}' has been deleted.")
    else:
          print(f"File '{file_to_delete}' does not exist.")

    # Define the file to delete
    file_to_delete = "channel_RGB_64bit_binned_gamma_corrected_drs_R.fit"
    file_to_delete = (os.path.join(currentDirectory, file_to_delete))
    # Check if the file exists
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)  # Delete the file
        print(f"File '{file_to_delete}' has been deleted.")
    else:
        print(f"File '{file_to_delete}' does not exist.")

  
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

  if sysargv1 == '48':
    MxdlAstap()

  if sysargv1 == '49':
    CentRatio()

  if sysargv1 == '50':
    HpMore()

  if sysargv1 == '51':
    CombBgrAlIm()

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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      