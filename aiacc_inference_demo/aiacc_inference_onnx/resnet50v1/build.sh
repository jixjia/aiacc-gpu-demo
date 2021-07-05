#假设c_api目录位于../c_api/
ONNX_PATH=../c_api/
ONNX_INCLUDE_PATH=${ONNX_PATH}/include/
ONNX_SESSION_PATH=${ONNX_INCLUDE_PATH}/onnxruntime/core/session/
ONNX_LIB_PATH=${ONNX_PATH}/lib/
g++ -std=c++11 test.cpp -o resnet50-test  -I ${ONNX_INCLUDE_PATH} -I ${ONNX_SESSION_PATH}  -L ${ONNX_LIB_PATH} -laiaccix
