import tensorflow as tf
import os, sys
import numpy as np
import cPickle

from tensorflow.contrib.framework.python.framework import checkpoint_utils

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import resnet_utils

sys.path.append('../../no_tfutils/')
import vgg_preprocessing
import pdb

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


neural_image_folder = '/mnt/fs0/datasets/neural_data/img_split/V4IT/tf_records/images'
filenames = os.listdir(neural_image_folder)
str_list = ['block3_5', 'block3_6', 'block3_7', 'block3_8','block3_9','block3_10','block3_11','block3_12','block3_13','block3_14','block3_15','block3_16','block3_17','block3_18','block3_19','block3_20','block3_21','block3_22']
#print(len(filenames))

input_image = tf.placeholder(tf.float32, shape=[256, 256, 3])
#print(input_image.shape)
input_image_model = vgg_preprocessing.preprocess_for_eval(input_image, 224, 224, 224)
#print(input_image_model.shape)
input_image_model = tf.expand_dims(input_image_model, 0)

logits, activation_handles = resnet_v2.resnet_v2_101(input_image_model, scope='resnet_v2_101', is_training=False)

block_list = []
        
for block_str in str_list:
    block = activation_handles['resnet_v2_101/block%s/unit_%s/bottleneck_v2' % (block_str.split('_')[0][-1], block_str.split('_')[1])]
    block_list.append(block)

var_names = [x[0] for x in checkpoint_utils.list_variables('/data/simy/deepmind_model/model_release-7')]
#print(var_names)
var_list = ([v for v in tf.global_variables() if v.name[:-2].encode('ascii', 'ignore') in var_names])
#print(var_list)
variables_to_restore = {var.op.name: var for var in var_list}
#print(variables_to_restore)
#restore_op, restore_feed_dict = slim.assign_from_checkpoint('/data/simy/deepmind_model/model_release-7',variables_to_restore)
#print(restore_feed_dict.keys())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#print("Construction Done!")

# Create an initial assignment function.
#def InitAssignFn(sess):
#    sess.run(restore_op, restore_feed_dict)
saver_ = tf.train.Saver(variables_to_restore)
print("Construction Done!")
# Run training.
#my_log_dir = '/data/simy/'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver_.restore(sess, '/data/simy/deepmind_model/model_release-7')

    i = 0
    for filename in filenames:
        print(filename)
        if filename == 'meta.pkl':
            continue
        #print(filename.strip('-'))
        prefix = filename.split('_')[0]
        print(prefix)
        prefix_number = (filename.split('-')[0]).split('_')[1]
        print(prefix_number)
        
        reconstructed_imgs = []
        mlt_record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(neural_image_folder, filename))
        curr_index = 0
        writer_list = []
        
        for writer_str in str_list:
            os.system('mkdir -p /mnt/fs1/siming/Dataset/deepmind/%s' % writer_str)
            tfr_path = os.path.join('/mnt/fs1/siming/Dataset/deepmind/%s' % writer_str, "%s-%s.tfrecords" % (prefix, prefix_number))
            writer = tf.python_io.TFRecordWriter(tfr_path)
            writer_list.append(writer)


        for string_record in mlt_record_iterator:
            #if i > 2:
            #    break
            print(i)
            example = tf.train.Example()
            example.ParseFromString(string_record)
            reconstructed_mlt = (example.features.feature['images'].bytes_list.value[0])
            img_1d = np.fromstring(reconstructed_mlt, dtype=np.float32)
            img_1d = np.multiply(img_1d, 255.0)
            reconstructed_img = img_1d.reshape((256, 256, -1))
            
            for block in block_list:
                print(block.shape)

            save_list = sess.run(block_list, feed_dict={input_image:reconstructed_img})
        
            for save_block, writer, strs in zip(save_list, writer_list, str_list):
        
                save_block = np.squeeze(save_block, axis=0)
                save_block = save_block.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    strs: _bytes_feature(save_block)
                    }))
                writer.write(example.SerializeToString())
        

            '''
            writer.write(example.SerializeToString())'''
            i = i + 1
            
        for writer in writer_list:
            writer.close()

    #slim.learning.train(representation, my_log_dir, init_fn=InitAssignFn, init_feed_dict={input_image: reconstructed_imgs})


# Do the Neural Fitting


# the shape of "input_image" should be [batch_size, height, width, channel]
# setting "is_training" to be false means for the part of resnet, all the parameters should be freezed


'''
  print(representation.shape)
  curr_shape = representation.get_shape().as_list()
  spa_shape_x = curr_shape[1]
  spa_shape_y = curr_shape[2]
  cha_shape = curr_shape[3]
  out_shape = 88
  bias=0
  resh = tf.reshape(representation,[curr_shape[0], -1], name='reshape')
  spa_kernel = tf.get_variable(initializer=tf.variance_scaling_initializer(),
                               shape=[spa_shape_x, spa_shape_y, 1, out_shape],
                               dtype=tf.float32,
                               regularizer=tf.contrib.layers.l2_regularizer(0.),
                               name='spa_weights', trainable=True)
  cha_kernel = tf.get_variable(initializer=tf.variance_scaling_initializer(),
                               shape=[1, 1, cha_shape, out_shape],
                               dtype=tf.float32,
                               regularizer=tf.contrib.layers.l2_regularizer(0.),
                               name='cha_weights', trainable=True)
  kernel = spa_kernel * cha_kernel
  biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                           shape=[out_shape],
                           dtype=tf.float32,
                           regularizer=tf.contrib.layers.l2_regularizer(0.),
                           name='bias', trainable=True)

  kernel = tf.reshape(kernel, [-1, out_shape], name='ker_reshape')

  fcm = tf.matmul(resh, kernel)
  output = tf.nn.bias_add(fcm, biases, name='spa_disen_fc')
'''

