# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Send JPEG image to tensorflow_model_server loaded with ResNet model.

"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import tensorflow.compat.v1 as tf
import numpy as np
import time
from PIL import Image
import aiacc_inference_tf2

img = 'cat.jpg'
img = 'dog.jpg'

def _get_class():
  path = 'resnet50_labels.txt'
  with open(path) as f:
    class_names = f.readlines()
  class_names = [c.strip() for c in class_names]
  return class_names

def input_fn(input_data):
    shape = [1, 224, 224, 3]
    image = Image.open(input_data)
    image = image.resize((shape[2],shape[1]))
  
    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    image_data = image_data.repeat(shape[0],axis=0)

    return image_data

def main(_):
  image_data = input_fn(img)

  SAVED_MODEL_DIR="./resnet_50_savedmodel/1/"
  SAVED_MODEL_DIR="./saved_model/1"

  tf_config = tf.ConfigProto()
  tf_config.gpu_options.allow_growth = True
  sess = tf.Session(config=tf_config)
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], SAVED_MODEL_DIR)

  if True:
    classes_tensor = sess.graph.get_tensor_by_name('PartitionedCall:0')

    for _ in range(5):
      result = sess.run([classes_tensor], {'serving_default_input_1:0': image_data})

    num_requests = 100
    start = time.time()
    for _ in range(num_requests):
      result = sess.run([classes_tensor], {'serving_default_input_1:0': image_data})
    total_time = time.time() - start
    print("average time ms", (total_time * 1000) / num_requests)

    class_id = np.argmax(result[0][0])
    class_names = _get_class()
    print('type:', class_names[class_id])

if __name__ == '__main__':
  tf.app.run()
