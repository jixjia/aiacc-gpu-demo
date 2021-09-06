# AIACC GPU Demo
Demonstration of an accelerated training and inference using AIACC on Alibaba Cloud NVIDIA GPU

## Set up AIACC
Tick AIACC install options during ECS provisioning

![](materials/ecs_aiacc.gif)


Activate the auto-installed demo AIACC conda env:

![](materials/aiacc_conda.gif)


## How-To
Execute `**_horovod.py` for Horovod distributed training  
Execute `**_perseus.py` for AIACC boosted distributed training

Compare and contrast the training efficiency 

[Note] Ensure running test scripts in their corresponding conda environment


## Tested Environment

|  GPU Hardware  |    |
| ---- | ---- |
|  GPU  |  Nvidia V100  |
|  Driver  |  450.x.x  |
|  CUDA  |  10.1.x  |
|  CuDNN  |  7.6.5  |


| Dependencies   |    |
| ---- | ---- |
|  Tensorflow GPU |  2.1  |
|  Open MPI |  3.1.3  |
|  OpenCV |  4.5.1  |

