#!/bin/bash
set -e
source scripts/common.sh

dataset=vqa2
split_test=val

# Train with:
# Relevant cues (hat)
# Irrelevant cues (one_minus_hat)
# Fixed Random cues (random_hat)
for hint_type in hat one_minus_hat random_hat
do
  expt=HINT_${hint_type}_${dataset}
  mkdir -p ${SAVE_DIR}/${expt}

  CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --data_dir ${DATA_DIR} \
  --dataset ${dataset} \
  --learning_rate 2e-5 \
  --split train \
  --hint_type ${hint_type} \
  --split_test ${split_test} \
  --max_epochs 12 \
  --checkpoint_path ${SAVE_DIR}/${expt} \
  --load_checkpoint_path ${SAVE_DIR}/baseline_${dataset}/model-best.pth \
  --vqa_loss_weight 1 \
  --hint_loss_weight 2 \
  --use_hint_loss > ${SAVE_DIR}/${expt}/verbose_log.txt
done

# Train with variable random cues (i.e., cues vary every epoch)
hint_type=random_hat
expt=HINT_variable_${hint_type}_${dataset}
mkdir -p ${SAVE_DIR}/${expt}

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--data_dir ${DATA_DIR} \
--dataset ${dataset} \
--learning_rate 2e-5 \
--split train \
--hint_type ${hint_type} \
--split_test ${split_test} \
--max_epochs 12 \
--checkpoint_path ${SAVE_DIR}/${expt} \
--load_checkpoint_path ${SAVE_DIR}/baseline_${dataset}/model-best.pth \
--vqa_loss_weight 1 \
--hint_loss_weight 2 \
--use_hint_loss \
--change_scores_every_epoch > ${SAVE_DIR}/${expt}/verbose_log.txt