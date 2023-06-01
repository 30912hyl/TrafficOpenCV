import cv2 as cv
from rescale import *

img = cv.imread('Photos/london.jpg') 
scaled = rescaleFrame(img, 0.75)
cv.imshow('London', scaled)

# Convert to grayscale
gray = cv.cvtColor(scaled, cv.COLOR_BGR2GRAY)  
#cv.imshow('Gray', gray)

# Blur, increase (n,n) for more blur, n must be odd
blur = cv.GaussianBlur(scaled,(3,3), cv.BORDER_DEFAULT) 
#cv.imshow('Blur', blur)

# Edge Cascade, use blur>scaled to reduce edges
canny = cv.Canny(scaled, 125, 175)
#cv.imshow('Canny Edges', canny)

# Dilating(smooth/round edges) the image 
dilated = cv.dilate(canny, (3,3), iterations=1)
#cv.imshow('Dilated', dilated)

# Eroding(anti-dilate)
eroded = cv.erode(dilated, (3,3), iterations=1)
#cv.imshow('Eroded', eroded)

# Resize 
'''
Resize function has a third arg interpolation=cv.INTER_AREA by default
use cv.INTER_AREA for downscaling, _LINEAR for upscaling, _CUBIC for high quality
'''
resized = cv.resize(scaled, (500,500))
#cv.imshow('Resized', resized)

# Cropping, images are arrays so array slice
cropped = scaled[50:200,200:400]
#cv.imshow('Cropped', cropped)

cv.waitKey(0) 