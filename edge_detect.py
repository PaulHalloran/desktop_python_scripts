import cv2
import sys

# Get user supplied values
directory = '/home/ph290/Documents/open_cv_stuff/'
imagePath = directory+'people.jpg'

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,10,200)

cv2.imshow("edges highlighted", edges)
cv2.waitKey(0)

