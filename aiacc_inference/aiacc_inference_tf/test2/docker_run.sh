version=`python3 -c 'import tensorflow as tf; print(tf.__version__)'`

echo ${version}

docker run --gpus all -p 8500:8500 -p 8501:8501 -it --rm --mount type=bind,source=`pwd`/saved_model,target=/models/resnet -e MODEL_NAME=resnet -t registry.cn-beijing.aliyuncs.com/ai_beijing/tensorflow_serving:${version}-gpu
