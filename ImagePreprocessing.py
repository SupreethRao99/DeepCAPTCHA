import sys
import cv2 as cv

image = cv.imread('index.png')

if image is None:
    sys.exit('Could not read the image')

img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    if cv.contourArea(cnt, True) > 64:  # heuristically determined value
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv.imshow("Display Window", image)
k = cv.waitKey(0)
