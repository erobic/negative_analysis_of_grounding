#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqacp2
split_test=test


for seed in 500
do
  for hint_type in hat one_minus_hat random_hat
  do
    expt=HINT_${hint_type}_var_rand_${dataset}_seed${seed}

    CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --dataset ${dataset} \
    --learning_rate 2e-5 \
    --split train \
    --hint_type ${hint_type} \
    --split_test ${split_test} \
    --max_epochs 12 \
    --checkpoint_path saved_models/${expt} \
    --load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
    --vqa_loss_weight 1 \
    --hint_loss_weight 2 \
    --change_scores_every_epoch \
    --seed ${seed} \
    --use_hint_loss > /hdd/robik/scr_experiments/${expt}.log
  done
done