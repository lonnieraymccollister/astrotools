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
from astropy.io import fits


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
  sysargv1  = input("Enter the first masked Image  -->")
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
  image = cv2.imread(sysargv1)
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
  sysargv5  = input("Enter the final image name fits  -->") 
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
      rescaled = (65535.0 / (my_data1.max()+1) * ((my_data1+1) - my_data1.min())).astype(np.uint16)
      my_data[xw:(xw+int(sysargv2)), yh:(yh+int(sysargv2))] = rescaled

  hdu = fits.PrimaryHDU(my_data)
  # Create an HDU list and add the primary HDU
  hdulist = fits.HDUList([hdu])
  # Specify the output FITS file path
  output_fits_filename = sysargv5
  # Write the HDU list to the FITS file
  hdulist.writeto(output_fits_filename, overwrite=True)
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

def menue(sysargv1):
  sysargv1 = input("Enter \n>>1<< AffineTransform >>2<< Mask an image >>3<< Mask Invert >>4<< Add2images  \n>>5<< Split tricolor >>6<< Combine Tricolor >>7<< Create Luminance(2ax) >>8<< Align2img \n>>9<< Plot_16-bit_img to 3d graph(2ax) >>10<< Centroid_Custom_filter(2ax) >>11<< UnsharpMask \n>>12<< FFT-Bandpass(2ax) >>13<< Img-DeconvClr >>14<< Centroid_Custom_Array(2ax) \n>>15<< Erosion(2ax) >>16<< Dilation(2ax) >>17<< DynamicRescale(2ax) >>18<< Gaussian \n>>1313<< Exit --> ")
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
