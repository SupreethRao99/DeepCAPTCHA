import cv2 as cv
import sys
import numpy as np

image = cv.imread('test1.png')
img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

if img is None:
    sys.exit('Could not read the image')

ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)



contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 1)
print(hierarchy)
cv.imshow("Display Window", image)
k = cv.waitKey(0)
