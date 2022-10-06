# astrotools  
 Used to view objects like Messier - 87.  Removes symetric light from galaxy etc.  Find the centroid of the star ---> galaxy and crop the object to a square making certain the centroid is in the exact center of the image and the image has an odd number of pixels(H121-W121). 
step 1
((python astrotools.py) or (astrotools.exe(windows10/11))) 1 luminance.png
create csv img_pixel.csv

examine image in csv and edit/change position if neccessary etc.

step (2 - 8bit_im_out)/(4 - 16bit)
((python astrotools.py) or (astrotools.exe(windows10/11))) 2
create img_pixels2.csv from img_pixel.csv then enter the above command which then creates img_pixels2.png and shows img_pixels2.png

step (3 - 8bit_im)/(5 - 16bit)
((python astrotools.py) or (astrotools.exe(windows10/11))) 3 img_pixels2.png
shows graph:  x - y - z(intensity in color and 3d)

The following will create an image and display image without using a spreadsheet.  The output file is img_pixels2.png ---

((python astrotools.py) or (astrotools.exe(windows10/11))) 6 luminance.png 60 893 687
 
60 is radius to be displayed.  --  893 687 --x y cordinates of the image centroid to be removed.

The following will crop an image and display image without using a spreadsheet.  The output file is crop.png ---

((python astrotools.py) or (astrotools.exe(windows10/11))) 7 luminance.png 60 893 687
 
60 is radius to be displayed.  --  893 687 --x y cordinates of the image centroid to be removed.

The following will dynamically remove the background of an image and display image without using a spreadsheet.  The output file is img_pixels2.png ---

((python astrotools.py) or (astrotools.exe(windows10/11))) 8 luminance.png 50 3

50 -- mesh size    3 -- filter size 

The following will use Lucy Richardson deconvolution on an image.  The output file is img_pixels2.png ---

((python astrotools.py) or (astrotools.exe(windows10/11))) 9 RGB.png 

The psf of the star is generated internally by the program. (Requires RGB image) 30 iterations are run to produce a low contrast output file.

The following will increase s/n create an image running astrotoolsa.py as a tile of radius 100/(s/n computation level) pixels. Stack the created images for the final image Deep Sky Stacker recomended) --- 

((python astrotools_tileloop.py) luminance.png (image width in pixels) (image width in pixels) (image height in pixels)(x-column start pixel) (s/n computation level use 1,2,4,5,10)

====================================================================================
The python program is also compiled as a windows program which is nearly 400 meg and can be downloaded at https://lonnieraymccollister.info/Misc/astrotools.zip
====================================================================================