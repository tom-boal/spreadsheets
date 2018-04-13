import tensorflow as tf
import numpy as np

np.set_printoptions(precision=3)

matrix = tf.constant([[1.24, 2.35]])
negMatrix = tf.negative(matrix)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    result = sess.run(negMatrix)

print(result)
