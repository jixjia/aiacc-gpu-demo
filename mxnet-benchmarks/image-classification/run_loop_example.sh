#!/bin/bash

if [ $# -ne 3 ] ; then
  echo "Usage: $0 job_name workers gpus_per_worker"
  exit -1
fi
arena submit mpijob --name=$1 \
             --workers=$2 \
             --gpus=$3 \
             --memory=200Gi \
             --cpu=52 \
             --image=registry-vpc.cn-huhehaote.aliyuncs.com/ai_hhht/perseus-mxnet:centos7-cuda10.0-1.2.0-1.4 \
             --data=pvc-cpfs:/mnt/newcpfs \
             "tail -f /dev/null"
