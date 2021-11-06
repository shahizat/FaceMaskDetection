import tensorflow as tf

model = tf.keras.models.load_model('mymodel.model')
tf.saved_model.save(model, '/home/jetson/Desktop/facemask/tf')
