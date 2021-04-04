import cv2 as cv
import imutils


def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv.copyMakeBorder(image, padH, padH, padW, padW,
                              cv.BORDER_REPLICATE)
    image = cv.resize(image, (width, height))

    # return the pre-processed image
    return image


import cv2 as cv

img = cv.imread('extracted_letter_images/001.png')
width = 32
height = 32
res = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)
cv.imwrite('resized_002.png', res)
