import sys
import cv2 as cv
import os

white = [255, 255, 255]
green = [0, 255, 0]
image = cv.imread('SampleImages/sample-003.png')
OUTPUT_FOLDER = "extracted_letter_images"

if image is None:
    sys.exit('Could not read the image')

img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
padded = cv.copyMakeBorder(thresh, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=white)

contours, hierarchy = cv.findContours(padded,
                                      cv.RETR_TREE,
                                      cv.CHAIN_APPROX_NONE)
letter_image_regions = []

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    if cv.contourArea(cnt, True) > 50:  # heuristically determined value
        letter_image_regions.append((x, y, w, h))

letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
count = 0

for letter_bounding_box in letter_image_regions:
    count = count + 1
    # Grab the coordinates of the letter in the image
    x, y, w, h = letter_bounding_box
    # Extract the letter from the original image with a 2-pixel
    # margin around the edge
    letter_image = padded[y - 2:y + h + 2, x - 2:x + w + 2]
    p = os.path.join(OUTPUT_FOLDER, '{}.png'.format(str(count).zfill(3)))
    cv.imwrite(p, letter_image)
