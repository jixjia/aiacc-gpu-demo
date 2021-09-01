from __future__ import print_function

import grpc
import requests
import tensorflow as tf

from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import model_service_pb2_grpc

def main(_):
  server = 'localhost:8500'
  channel = grpc.insecure_channel(server)
  request = get_model_status_pb2.GetModelStatusRequest()
  request.model_spec.name = 'resnet'
  stub = model_service_pb2_grpc.ModelServiceStub(channel)
  result = stub.GetModelStatus(request, 5)  # 5 secs timeout
  print(result.model_version_status[0])

if __name__ == '__main__':
  tf.app.run()
