import sys
import cv2 as cv

white = [255, 255, 255]
green = [0, 255, 0]
image = cv.imread('SampleImages/sample-003.png')

if image is None:
    sys.exit('Could not read the image')

img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
padded = cv.copyMakeBorder(thresh, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=white)

contours, hierarchy = cv.findContours(padded, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    if cv.contourArea(cnt, True) > 50:  # heuristically determined value
        cv.rectangle(padded, (x, y), (x + w, y + h), color=green, thickness=1)

cv.imshow("Display Window", padded)
k = cv.waitKey(0)
