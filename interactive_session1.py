import tensorflow as tf
import numpy as np
np.set_printoptions(precision=3)

sess = tf.InteractiveSession()

matrix = tf.constant([[1.1, 2.3]])
negMatrix = tf.negative(matrix)

result = negMatrix.eval()
print(result)
print(type(result))



sess.close()
