import tensorflow as tf

# Math with constant tensors
const_a = tf.constant(3.6)
const_b = tf.constant(1.2)
const_c = tf.constant([1.2])
total = const_a + const_b
quot = tf.div(const_a, const_b)
t1 = tf.constant([1.2, 2.3, 3.4, 4.5])
t6 = tf.constant([1.2, 2.3])

# Math with random tensors
rand_a = tf.random_normal([3], 2.0)
rand_b = tf.random_uniform([3], 1.0, 4.0)
diff = tf.subtract(rand_a, rand_b)

# Vector multiplication
vec_a = tf.linspace(0.0, 3.0, 4)
vec_aa = tf.linspace(0.0, 3.1, 4)
vec_b = tf.fill([4, 1], 2.0)
prod = tf.multiply(vec_a, vec_b)
dot = tf.tensordot(vec_a, vec_b, 1)

# Matrix multiplication
mat_a = tf.constant([[2, 3], [1, 2], [4, 5]])
mat_b = tf.constant([[6, 4, 1], [3, 7, 2]])
mat_prod = tf.matmul(mat_a, mat_b)

# Execute the operations
with tf.Session() as sess:
    print('const_a = ', const_a)
    print('const_a: %f\n' % sess.run(const_a))
    print('const_c: %f\n' % sess.run(const_c))
    print('sess.run(t6) = \n' , sess.run(t6))
    print('sess.run(t6) = ')
    print(sess.run(t6))
    print('sess.run(const_a) = \n' , sess.run(const_a))
    print('sess.run(const_a) = \n' , sess.run(const_a))
    print('sess.run(t1) = \n' , sess.run(t1))
    print('Sum: %f\n' % sess.run(total))
    print('Quotient: %f' % sess.run(quot))
    print('Difference: ', sess.run(diff))
    print('prod  = ', prod)
    print('Element-wise product = prod: ')
    print(sess.run(prod))
    print('Dot product: ', sess.run(dot))
    print('Matrix product: ')
    print(sess.run(mat_prod))
    print('vec_a =  ', vec_a)
    print('sess.run(vec_a) =  ', sess.run(vec_a))
    print('sess.run(vec_b) =  ', sess.run(vec_b))
    print('sess.run(vec_aa) =  ', sess.run(vec_aa))
    print(sess.run(mat_prod))
    #print('vec_a: %f\n' % sess.run(vec_a)) ERROR
    
