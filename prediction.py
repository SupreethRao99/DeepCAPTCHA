import tensorflow as tf

model = tf.keras.models.load_model('content/my_model')
model.predict('sample.png')
