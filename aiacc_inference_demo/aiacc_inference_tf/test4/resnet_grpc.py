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
import os

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS

def _get_class():
    path = 'resnet50_labels.txt'
    classes_path = os.path.expanduser(path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def main(_):
  FLAGS.image = 'dog.jpg'
  FLAGS.image = 'cat.jpg'
  with open(FLAGS.image, 'rb') as f:
    data = f.read()

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'resnet'
  request.model_spec.signature_name = 'predict'
  request.inputs['image_bytes'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data, shape=[1]))
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  class_id = result.outputs['classes'].int64_val[0] - 1
  class_names = _get_class()
  print(class_names[class_id])

if __name__ == '__main__':
  tf.app.run()
