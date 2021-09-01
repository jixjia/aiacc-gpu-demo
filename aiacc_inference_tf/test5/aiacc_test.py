import numpy as np
from PIL import Image
from aiacc_inference_tf.libaiacc_inference_tf import *

def input_fn(input_data):
    shape = [1, 299, 299, 3]
    image = Image.open(input_data)
    image = image.resize((shape[2],shape[1]))
  
    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    image_data = image_data.repeat(shape[0],axis=0)

    return image_data

model_fname = 'resnet_v2_50.pb'
input_names = ['input']
output_names = ['logits', 'classes']
output_model = 'opt_resnet_v2_50.pb'
batch_size = [1]
image_list = ["cat.jpg",  "dog.jpg", "elephant.jpg"]

optimize_tf_model(model_fname, output_names,
                  output_model, batch_size, 'INT8', True,
                  input_names, image_list, input_fn)
