import tensorflow as tf
import numpy as np

np.set_printoptions(precision=3)


x = tf.constant([[1.2, 2.3]])
neg_x = tf.negative(x)

print(neg_x)

with tf.Session() as sess:
    result = sess.run(neg_x)
print(result)
np_x = tf.contrib.util.constant_value(x)
print(type(x))
print(ret)
print(avalue)

