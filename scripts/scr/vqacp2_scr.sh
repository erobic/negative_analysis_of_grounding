#!/bin/bash
set -e
source scripts/common.sh

dataset=vqacp2
split_test=test

scr_loss_weight=3
scr_compare_loss_weight=1000

for hint_type in caption_based_hints one_minus_caption_based_hints random_caption_based_hints; do
    expt=scr_${hint_type}_${dataset}
    mkdir -p ${SAVE_DIR}/${expt}/phase_2

    learning_rate=5e-5
    CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --learning_rate ${learning_rate} \
    --split train \
    --hint_type ${hint_type} \
    --split_test ${split_test} \
    --max_epochs 7 \
    --checkpoint_path ${SAVE_DIR}/${expt}/phase_2 \
    --load_checkpoint_path ${SAVE_DIR}/baseline_${dataset}/model-best.pth \
    --use_scr_loss \
    --scr_hint_loss_weight ${scr_loss_weight} > ${SAVE_DIR}/${expt}/phase_2/verbose_log.txt

    mkdir -p ${SAVE_DIR}/${expt}/phase_3
    learning_rate=1e-4
    CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --data_dir ${DATA_DIR} \
    --dataset ${dataset} \
    --learning_rate ${learning_rate} \
    --split train \
    --hint_type ${hint_type} \
    --split_test ${split_test} \
    --max_epochs 9 \
    --checkpoint_path ${SAVE_DIR}/${expt}/phase_3 \
    --load_checkpoint_path ${SAVE_DIR}/${expt}/phase_2/model-best.pth \
    --scr_hint_loss_weight ${scr_loss_weight} \
    --use_scr_loss \
    --scr_compare_loss_weight ${scr_compare_loss_weight} > ${SAVE_DIR}/${expt}/phase_3/verbose_log.txt
done


expt=scr_variable_${hint_type}_${dataset}
mkdir -p ${SAVE_DIR}/${expt}/phase_2

learning_rate=5e-5
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--learning_rate ${learning_rate} \
--split train \
--hint_type ${hint_type} \
--split_test ${split_test} \
--max_epochs 7 \
--checkpoint_path ${SAVE_DIR}/${expt}/phase_2 \
--load_checkpoint_path ${SAVE_DIR}/baseline_${dataset}/model-best.pth \
--use_scr_loss \
--scr_hint_loss_weight ${scr_loss_weight} \
--change_scores_every_epoch > ${SAVE_DIR}/${expt}/phase2/verbose_log.txt

mkdir -p ${SAVE_DIR}/${expt}/phase_3
learning_rate=1e-4
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--learning_rate ${learning_rate} \
--split train \
--hint_type ${hint_type} \
--split_test ${split_test} \
--max_epochs 9 \
--checkpoint_path ${SAVE_DIR}/${expt}/phase_3 \
--load_checkpoint_path ${SAVE_DIR}/${expt}/phase_2/model-best.pth \
--scr_hint_loss_weight ${scr_loss_weight} \
--use_scr_loss \
--scr_compare_loss_weight ${scr_compare_loss_weight} \
--change_scores_every_epoch > ${SAVE_DIR}/${expt}/phase_3/verbose_log.txt
