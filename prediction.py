import tensorflow as tf

model = tf.keras.models.load_model('Model/CAPTCHA-Model')
model.summary()
