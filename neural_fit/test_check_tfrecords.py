import tensorflow as tf
import numpy as np

tfrec_to_check = '/data2/chengxuz/vm_response/tfrecords/encode_17-decode_1/train_0.tfrecords'

record_iterator = tf.python_io.tf_record_iterator(path=tfrec_to_check)
get_num = 0

for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    img_string = (example.features.feature['encode_17-decode_1']
                                  .bytes_list
                                  .value[0])
    img_array = np.fromstring(img_string)
    print(img_array.dtype)
    print(img_array[:10])
    print(img_array.shape)

