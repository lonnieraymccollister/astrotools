import sys, os
from PIL import Image
import glob

fGIF = "animGIF.gif"
H = 720
W = 1270
n = 1
# Create the frames
frames = []
images = glob.glob("*.jpg")

for i in images:
    newImg = Image.open(i)
#    if (len(sys.argv) < 2 and n > 0):
#        newImg = newImg.resize((W, H))
    frames.append(newImg)
 
# Save into a GIF file that loops forever: duration is in milli-second
frames[0].save(fGIF, format='GIF', append_images=frames[1:],
    save_all=True, duration=200, loop=0)