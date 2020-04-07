#!/bin/bash
set -e
source scripts/common.sh

dataset=vqa2
expt=baseline_${dataset}
mkdir -p ${SAVE_DIR}/${expt}

## Train
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--split train \
--split_test test \
--max_epochs 40 \
--do_not_discard_items_without_hints \
--checkpoint_path ${SAVE_DIR}/${expt} > ${SAVE_DIR}/${expt}/verbose_log.txt

## Test
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--test \
--split train \
--split_test test \
--do_not_discard_items_without_hints \
--checkpoint_path ${SAVE_DIR}/${expt} \
--load_checkpoint_path ${SAVE_DIR}/${expt}/model-best.pth > ${SAVE_DIR}/${expt}/test_verbose_log.txt