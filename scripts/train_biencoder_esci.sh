#!/usr/bin/env bash

set -x
set -e


DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
	  OUTPUT_DIR="${DIR}/checkpoint/biencoder_$(date +%F-%H%M.%S)"
fi
DATA_DIR="/data"

mkdir -p "${OUTPUT_DIR}"

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
# python -u -m torch.distributed.launch --nproc_per_node ${PROC_PER_NODE} src/train_biencoder.py \
deepspeed src/train_biencoder.py --deepspeed ds_config.json \
    --model_name_or_path intfloat/simlm-base-msmarco \
    --per_device_train_batch_size 416 \
    --per_device_eval_batch_size 416 \
    --add_pooler False \
    --t 0.02 \
    --seed 1234 \
    --do_train \
    --fp16 \
    --train_file "${DATA_DIR}/train.jsonl" \
    --validation_file "${DATA_DIR}/test.jsonl" \
    --q_max_len 32 \
    --p_max_len 144 \
    --train_n_passages 8 \
    --dataloader_num_workers 1 \
    --num_train_epochs 40 \
    --learning_rate 2e-5 \
    --use_scaled_loss True \
    --warmup_steps 1000 \
    --share_encoder True \
    --logging_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 1 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --data_name esci \
    --do_kd_biencoder False \
    --full_contrastive_loss False \
    --positive_size 1 \
    --additional_positives 0 \
    --n_hard_neg 0 \
    --n_rand_neg 0 \
    --n_other_neg 0 \
    --fn_kk_threshold 0 \
    --other_neg_sampling_strategy None \
    --add_qq_regularization False \
    --fill_neg_with_random_ids True \
    --embedding_dedup_threshold 0 \
    --use_gradient_checkpointing \
    --report_to none "$@"
