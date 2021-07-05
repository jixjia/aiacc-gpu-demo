import numpy as np
import aiaccix as aix
import time
import os
# %%init session
sess = aix.InferenceSession("./resnet50-v1-7.onnx")
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
print("input_name is %s, input_shape is %s"%(input_name,str(input_shape)))
# %%test image, image size == [1,3,224,224]
input_image = np.random.random((1,3,224,224)).astype(np.float32)

#warmup
for _ in range(10):
  pred_onnx = sess.run(None, {input_name: input_image})

#test the inference time of input_image
start = time.time()
for _ in range(1000):
  pred_onnx = sess.run(None, {input_name: input_image})
end = time.time()
print('shape is ',input_image.shape,', delta time: ',end-start,' ms')
