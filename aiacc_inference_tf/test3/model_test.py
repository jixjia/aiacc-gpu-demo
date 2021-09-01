import tensorflow as tf
from aiacc_inference_tf.libaiacc_inference_tf import *

graph_file = './model.h5'
output_model = './model.pb'
batch_size = [ 1 ]

optimize_keras_model(graph_file, output_model, batch_size, 'FP16', None, dynamic_op=False)
