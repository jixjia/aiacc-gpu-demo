import tensorflow as tf
from aiacc_inference_tf.libaiacc_inference_tf import *

export_dir = './saved_model/1'
graph_pb = './resnet_v2_50.pb'

input_names = ['input']
output_names = ['logits', 'classes']

input_tensor_names = ['input']
output_tensor_names = ['logits', 'classes']

signature_key = 'predict'
batch_size = [ 1 ]

optimize_tf_model_v2(graph_pb, export_dir, input_names, output_names,
                     input_tensor_names, output_tensor_names, signature_key,  batch_size, "FP16",
                     dynamic_op=False)
