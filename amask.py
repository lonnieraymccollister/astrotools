import cv2

sysargv1  = input("Enter the Image1  -->")
#sysargv2  = input("Enter the Image2  -->")
sysargv3  = input("Enter the Mask for image  white- and black  -->")
sysargv4  = input("Enter the filename of the mask image to save  -->")

image = cv2.imread(sysargv1)
#image1 = cv2.imread(sysargv2)
mask = cv2.imread(sysargv3)


# Apply the mask to the image
masked_image = cv2.bitwise_and(image, mask)

cv2.imwrite(sysargv4, masked_image)