#!/usr/bin/env bash
set -e
source activate ${ENV_NAME}
source common.sh

expt=baseline_vqa2
#
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--dataset vqa2 \
--split train \
--split_test val \
--max_epochs 40 \
--checkpoint_path ${SAVE_DIR}/${expt} > ${LOG_DIR}/${expt}.log

## To test
#CUDA_VISIBLE_DEVICES=0 python -u main.py \
#--dataset vqa2 \
#--split train \
#--split_test val \
#--max_epochs 40 \
#--checkpoint_path ${SAVE_DIR}/${expt} > ${LOG_DIR}/${expt}.log

#CUDA_VISIBLE_DEVICES=0 python -u main.py \
#--dataset vqa2 \
#--split train \
#--hint_type hat \
#--split_test val \
#--checkpoint_path saved_models/${expt} \
#--load_checkpoint_path saved_models/${expt}/model-best.pth \
#--test