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

import tensorflow as tf
import numpy as np
from PIL import Image
import time
import aiacc_inference_tf

# The image URL is the location of the image we should send to the server
img = 'dog.jpg'
img = 'cat.jpg'
img = 'elephant.jpg'
model_file = 'resnet_v2_50.pb'
model_file = 'opt_resnet_v2_50.pb'

def _get_class():
  path = 'resnet50_labels.txt'
  with open(path) as f:
    class_names = f.readlines()
  class_names = [c.strip() for c in class_names]
  return class_names

def create_graph():
  with tf.gfile.FastGFile(model_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def main(_):
  shape = [1, 299, 299, 3]
  image = Image.open(img)
  image = image.resize((shape[2],shape[1]))
  
  image_data = np.array(image,dtype='float32')
  image_data /= 255.
  image_data = np.expand_dims(image_data, 0)
  image_data = image_data.repeat(shape[0],axis=0)

  create_graph()
  output_names = [ "classes", "logits" ]

  with tf.Session() as sess:
    classes_tensor = sess.graph.get_tensor_by_name('classes:0')
    logits_tensor  = sess.graph.get_tensor_by_name('logits:0')

    for _ in range(5):
      result = sess.run([classes_tensor, logits_tensor],
                            {'input:0': image_data})

    num_requests = 100
    start = time.time()
    for _ in range(num_requests):
      result = sess.run([classes_tensor, logits_tensor],
                            {'input:0': image_data})
    total_time = time.time() - start
    print("average time ms", (total_time * 1000) / num_requests)
    class_names = _get_class()
    print('type:', class_names[result[0][0]])

if __name__ == '__main__':
  tf.app.run()
