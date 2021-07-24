import tensorflow as tf
import numpy as np

img = tf.keras.preprocessing.image.load_img(
    'ResizedStorage/005_resized.png',
    target_size=(32, 32)
)

img_nparray = tf.keras.preprocessing.image.img_to_array(img)
input_batch = np.array([img_nparray])

model = tf.saved_model.load('CAPTCHA-Model-New')
print(model.evaluate(input_batch))
