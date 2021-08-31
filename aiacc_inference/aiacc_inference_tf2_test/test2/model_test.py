import numpy as np
from PIL import Image
from aiacc_inference_tf2.libaiacc_inference_tf2 import *

input_model_dir = './resnet_50_saved_model/1'
saved_model_dir = './saved_model/1'
batch_size = [1]
image_list = ["cat.jpg",  "dog.jpg", "elephant.jpg"]

def input_fn(input_data):
    shape = [1, 224, 224, 3]
    image = Image.open(input_data)
    image = image.resize((shape[2],shape[1]))

    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    image_data = image_data.repeat(shape[0],axis=0)

    return image_data

optimize_tf_saved_model(input_model_dir, saved_model_dir,
                        batch_size, 'INT8',
                        image_list, input_fn)
