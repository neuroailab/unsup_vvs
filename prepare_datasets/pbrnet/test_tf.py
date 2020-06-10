import numpy as np
import tensorflow as tf
from skimage import io

original = np.array([[1,4096],[15000,30000]],dtype=np.uint16)
io.imsave('test.png',original)

sk_im = io.imread('test.png')

image = tf.image.decode_png(tf.read_file('test.png'),dtype=tf.uint16)
sess = tf.Session()
tf_im = sess.run(image)

print sk_im
print tf_im
