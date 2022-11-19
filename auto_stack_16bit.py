import os
import sys
import cv2
import numpy as np

# stack images
    ### -- example -- python auto_stacks_16bit.py H:\lonnielaptop\Astrotools\StarNetv2GUI_Win\image_stacking-master\image_stacking-master\test resultnew.png

file_list = os.listdir(sys.argv[1])
file_list = [os.path.join(sys.argv[1], x)
                 for x in file_list if x.endswith(('.png'))]
M = np.eye(3, 3, dtype=np.float32)
first_image = None
stacked_image = None
for file in file_list:
    image = cv2.imread(file, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535
    print(file)
    if first_image is None:
        first_image = image
        stacked_image = image
    else:
        stacked_image += image
stacked_image /= len(file_list)   
stacked_image = (stacked_image*65535).astype(np.uint16)
file_list = [os.path.join(sys.argv[1], x)
                 for x in file_list if x.endswith(('.jpg', '.png','.bmp'))]
cv2.imwrite(sys.argv[2], stacked_image)
print("===>Stacking complete<===")
print("===>Stacking complete<===")
print("===>Stacking complete<===")






