#!/bin/sh

NUM_BATCH="${num_batch:-50}"
BATCH_SIZE="${batch_size:-128}"
MODEL_NAME="${model_name:-"resnet50"}"
DATA_NAME="${data_name:-"imagenet"}"
DATA_DIR="${data_dir:-"/mnt/newcpfs/imagenet_data"}"
TRAIN_DIR="${train_dir:-"/tmp/ckp-models"}"
OPTIMIZER_NAME="${optimizer_name:-"momentum"}"
MOMENTUM_VAL="${momentum_val:-0.9}"
WEIGHT_DEC="${weight_dec:-0.0001}"
DISPLAY_EVE="${display_eve:-10}"
SUMMARY_VERB="${summary_verb:-0}"
SAVE_SUMMARY_STEP="${save_summary_step:-0}"
TRAIN_DIR=${TRAIN_DIR}-fp16-${MODEL_NAME}-`date +%Y%m%d-%H%M%S`

python ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
    --num_batches=${NUM_BATCH} --display_every=${DISPLAY_EVE} \
    --num_gpus=1 \
    --data_name=${DATA_NAME} \
    --model=${MODEL_NAME} --batch_size=${BATCH_SIZE} \
    --variable_update=horovod --horovod_device=gpu \
    --batch_group_size=4 \
    --optimizer=${OPTIMIZER_NAME} \
    --distortions=False \
    --momentum=0.9 \
    --train_dir=${TRAIN_DIR} \
    --summary_verbosity=${SUMMARY_VERB} \
    --use_datasets=False \
    --use_fp16=True \
    --winograd_nonfused=True \
    --fp16_loss_scale=10.0 \
    --datasets_num_private_threads=5 \
    --xla=False
#--data_dir=${DATA_DIR} \
