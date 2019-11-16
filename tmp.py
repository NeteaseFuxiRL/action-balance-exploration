import tensorflow as tf
import numpy as np


x = tf.placeholder(dtype=tf.float32, shape=[1, 1, 18])
mean_tmp, var_tmp = tf.nn.moments(x, axes=[-1])
nor_x = (x - mean_tmp) / tf.math.sqrt(var_tmp)

with tf.Session() as sess:
    b = np.array([0.020269574058359843, 0.020355916667331057, 0.020380554301728807, 0.02046636518546633, 0.020272864148952095, 0.02090490851213814, 0.020256753313654348, 0.020936352159718823, 0.020600762059475204, 0.020850010256676366, 0.020055667135736536, 0.020619159855189334, 0.02031368893322333, 0.020872680111282266, 0.020225164164845388, 0.020495997689547525, 0.020301612792561985, 0.020389019703832317])
    print(sess.run([mean_tmp, var_tmp], feed_dict={x: np.reshape(b, (1, 1, 18))}))
    print(sess.run(nor_x, feed_dict={x: np.reshape(b, (1, 1, 18))}))