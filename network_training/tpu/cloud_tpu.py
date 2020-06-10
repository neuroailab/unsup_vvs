import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

def axy_computation(a, x, y):
  return a * x + y

inputs = [
    3.0,
    tf.ones([3, 3], tf.float32),
    tf.ones([3, 3], tf.float32),
]

tpu_computation = tpu.rewrite(axy_computation, inputs)

tpu_cluster_resolver = (
        tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_names=["siming-tpu"],
            zone="us-central1-f",
            project="tpucloud-196821"))
tpu_grpc_url = tpu_cluster_resolver.get_master()

with tf.Session(tpu_grpc_url) as sess:
  sess.run(tpu.initialize_system())
  sess.run(tf.global_variables_initializer())
  output = sess.run(tpu_computation)
  print(output)
  sess.run(tpu.shutdown_system())

print('Done!')
