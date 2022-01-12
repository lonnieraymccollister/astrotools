# astrotools  
 Used to view objects like Messier - 87.  Removes symetric light from galaxy etc.  Find the centroid of the star ---> galaxy and crop the object to a square making certain the centroid is in the exact center of the image and the image has an odd number of pixels. 
step 1
((python astrotools.py) or (astrotools.exe(windows10/11))) 1 luminance.png
create csv img_pixel.csv

examine image in csv and edit/change position if neccessary etc.

step (2 - 8bit_im_out)/(4 - 16bit)
((python astrotools.py) or (astrotools.exe(windows10/11))) 2
create img_pixels2.csv from img_pixel.csv then enter the above command which then creates img_pixels2.png and shows img_pixels2.png

step (3 - 8bit_im)/(5 - 16bit)
((python astrotools.py) or (astrotools.exe(windows10/11))) 3 img_pixels2.png
shows:  x - y - z(intensity in color and 3d)

The following will create an image without using a spreadsheet.  The output file is img_pixels2.png

((python astrotools.py) or (astrotools.exe(windows10/11))) 6 luminance.png 128
128 is the width of square image - 1 so the image is 129 pixels.


====================================================================================
The python program is also compiled as a windows program which is nearly 400 meg and can be downloaded at https://lonnieraymccollister.info/Misc/astrotools.zip
====================================================================================