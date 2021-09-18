# A computer vision script to detect common objects in a specified image. Uses OpenCV and YOLO data set.
# Made by Andrew Banonis | @andrewbanonis

# Before you start, install the prerequistes:
#   opencv-python
#   cvlib
#   matplotlib
#   tensorflow

import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

# **Specify image here!**
image = "your image.jpg" #image file to be used
print("Got the image specified...")

#OpenCV reads file specified in 'image' var
im = cv2.imread(image)
print("Reading image...")

#detects stuff in image specified
bbox, label, conf = cv.detect_common_objects(im)
print("Detecting stuff in the image...")

#draws the boxes with labels
output_image = draw_bbox(im, bbox, label, conf)
print("Drawing boxes and writing labels...")

#gets the image output
plt.imshow(output_image)
print("Outputting the image...")

#shows the image
plt.show()
print("Displaying image!")
