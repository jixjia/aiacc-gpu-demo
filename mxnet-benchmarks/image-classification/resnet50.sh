#!/bin/bash

let GPU=${OMPI_COMM_WORLD_RANK}%${OMPI_COMM_WORLD_LOCAL_SIZE}
export OMP_NUM_THREADS=4

#export NSIGHT_CUDA_DEBUGGER=1
export NCCL_DEBUG=INFO
#export NCCL_NTHREADS=128
#export PERSEUS_ALLREDUCE_MODE=1
#export PERSEUS_ALLREDUCE_DTYPE=2
#export PERSEUS_ALLREDUCE_STREAMS=2
#export PERSEUS_ALLREDUCE_FUSION=32

export DATA_DIR=/mnt/newcpfs/mxnet-data

MXNET_VISIBLE_DEVICE=$GPU python train_imagenet.py \
                                 --network resnet \
                                 --num-layers 50 \
                                 --data-nthreads 3 \
                                 --gpus $GPU \
                                 --batch-size 112 \
                                 --model ~/resnet50-test \
                                 --num-epochs 1 \
                                 --kv-store dist_sync_perseus \
                                 --benchmark 1 \
                                 --lr 0.1 --lr-step-epochs 30,60,80 --warmup-epochs 5 --optimizer sgd --loss ce  \
                                 --top-k 5 --disp-batches 50 \
                                 --data-train ${DATA_DIR}/train/full_imagenet.rec \
                                 --data-train-idx ${DATA_DIR}/train_meta/full_imagenet.idx \
                                 --data-val ${DATA_DIR}/val/full_imagenet.rec \
                                 --data-val-idx ${DATA_DIR}/val_meta/full_imagenet.idx \
                                 --max-random-area 1 --min-random-area 0.08 \
                                 --max-random-aspect-ratio 1.33 --min-random-aspect-ratio 0.75 --random-resized-crop 1 \
                                 --random-mirror 1
