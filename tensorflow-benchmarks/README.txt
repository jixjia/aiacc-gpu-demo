benchmarks.1.12，benchmarks.1.13，benchmarks.1.14

以上三个文件夹分别对应tensorflow-1.12，tensorflow-1.13.1, tensorflow-1.14的测试脚本，进入文件夹第一层后执行以下命令即可： 

mpirun --allow-run-as-root --bind-to none -np 8 -npernode 8  \
       --mca btl_tcp_if_include eth0  \
       --mca orte_keep_fqdn_hostnames t   \
       -x HOROVOD_TIMELINE  \
       -x HOROVOD_FUSION_THRESHOLD  \
       -x NCCL_SOCKET_IFNAME=eth0   \
       -x LD_LIBRARY_PATH   \
       ./config-fp16-tf.sh


参数说明：
-np代表使用的总卡数，-npernode代表每个节点使用的卡数，因为是单机测试，保证np 等于 npernode即可。一般使用1，2，4，8作为卡数.
