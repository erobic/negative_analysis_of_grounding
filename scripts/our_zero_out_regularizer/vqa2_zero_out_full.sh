#!/bin/bash
set -e
source scripts/common.sh

dataset=vqacp2
split_test=test

expt=${dataset}_zero_out_full
mkdir -p ${SAVE_DIR}/${expt}

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--learning_rate 1e-6 \
--split train \
--split_test ${split_test} \
--max_epochs 12 \
--checkpoint_path ${SAVE_DIR}/${expt} \
--load_checkpoint_path ${SAVE_DIR}/baseline_${dataset}/model-best.pth \
--use_fixed_gt_ans_loss \
--fixed_ans_scores 0 \
--do_not_discard_items_without_hints \
--fixed_gt_ans_loss_weight 2 \
--fixed_random_subset_ratio 1.0 > ${SAVE_DIR}/${expt}/verbose_log.txt