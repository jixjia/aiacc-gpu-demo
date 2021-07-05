import tensorflow as tf
from aiacc_inference_tf2.libaiacc_inference_tf2 import *

input_model_dir = './resnet_v2_fp32_savedmodel_NHWC_jpg/1538687457'
saved_model_dir = './saved_model/1'
batch_size = [1]

optimize_tf_saved_model(input_model_dir, saved_model_dir,
                        batch_size, precision='FP16')
