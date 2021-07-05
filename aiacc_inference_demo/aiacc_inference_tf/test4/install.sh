TF_VER=( $(python3 -c 'import tensorflow as tf; print(tf.__version__)') )
echo ${TF_VER}

pip3 install requests
pip3 install tensorflow_serving_api_gpu==${TF_VER}
