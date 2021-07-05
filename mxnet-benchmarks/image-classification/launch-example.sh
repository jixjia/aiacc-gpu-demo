#!/bin/bash

#if [ $# -ne 2 ]; then
#  echo "Usage: " $0  " total_worker_count  worker_count_per_node"
#  exit -1
#fi

if [ $# -ge 1 ] ; then
  workers=$1
fi
if [ $# -ge 2 ]; then
  gpus=$2
fi

let tot_gpus=${workers}*${gpus}
let n_per_node=${gpus}

runDir=`dirname $0`
cd $runDir
chmod +x ./resnet50.sh

mpirun --allow-run-as-root -np ${tot_gpus} -npernode ${n_per_node} \
       --mca btl_tcp_if_include eth0 \
       --mca orte_keep_fqdn_hostnames t \
       --bind-to none\
       --report-bindings \
       -x NCCL_IB_DISABLE=1 \
       -x NCCL_SOCKET_IFNAME=eth0 \
       -x LD_LIBRARY_PATH \
       ./resnet50.sh

#--hostfile hostfile \
