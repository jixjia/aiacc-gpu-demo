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

import grpc
import requests
import tensorflow as tf
import numpy as np
import time
import os

from PIL import Image
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# The image URL is the location of the image we should send to the server
img = 'cat.jpg'
img = 'dog.jpg'

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

def _get_class():
    path = 'resnet50_labels.txt'
    classes_path = os.path.expanduser(path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def main(_):
  shape = [1, 299, 299, 3]
  image = Image.open(img)
  image = image.resize((shape[2],shape[1]))
  
  #with open(FLAGS.image, 'rb') as f:
  #  data = f.read()
  #data = np.zeros(shape,dtype='float32')
  image_data = np.array(image,dtype='float32')
  image_data /= 255.
  image_data = np.expand_dims(image_data, 0)
  image_data = image_data.repeat(shape[0], axis=0)
  print(image_data.shape, image_data.nbytes)

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'resnet'
  request.model_spec.signature_name = 'predict'
  request.inputs['input'].CopyFrom(
      tf.contrib.util.make_tensor_proto(image_data, shape=shape))
  for _ in range(50):
      result = stub.Predict(request, 10)  # 10 secs timeout
  #print(request)

  import datetime
  num_requests = 100
  start = time.time()
  for _ in range(num_requests):
      result = stub.Predict(request, 10)  # 10 secs timeout
  total_time = time.time() - start
  print("average time ", (total_time * 1000) / num_requests)
  class_names = _get_class()
  print('type:', class_names[result.outputs['classes'].int64_val[0]])

if __name__ == '__main__':
  tf.app.run()
