import tensorflow as tf
from aiacc_inference_tf.libaiacc_inference_tf import *

model_fname = 'resnet_v2_50.pb'
output_names = ['logits', 'classes']
output_model = 'opt_resnet_v2_50.pb'

optimize_tf_model(model_fname, output_names, output_model,
                  batch_size=[1], precision='FP16', dynamic_op=False)
