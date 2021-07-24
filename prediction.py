import tensorflow as tf
import cv2 as cv
from resize import resize_to_fit
import numpy as np

image = cv.imread('extracted_letter_images/005.png')
image_resized = resize_to_fit(image, 32, 32)
# cv.imwrite('ResizedStorage/005_resized.png', image_resized)
# image_resized = np.expand_dims(image_resized, axis=0)

model = tf.keras.models.load_model('Model/CAPTCHA-Model')
model.summary()
# print(model.predict('ResizedStorage/005_resized.png'))
