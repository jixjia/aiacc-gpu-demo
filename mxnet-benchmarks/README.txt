进入测试脚本文件夹image-classification，执行以下命令： 

mpirun --allow-run-as-root -np 8 -npernode 8        --mca btl_tcp_if_include eth0        --mca orte_keep_fqdn_hostnames t        --bind-to none       --report-bindings        -x NCCL_IB_DISABLE=1        -x NCCL_SOCKET_IFNAME=eth0        -x LD_LIBRARY_PATH        ./resnet50.sh

参数说明：
-np代表使用的总卡数，-npernode代表每个节点使用的卡数，因为是单机测试，保证np 等于 npernode即可。一般使用1，2，4，8作为卡数.
