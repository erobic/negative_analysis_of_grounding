#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqacp2
split_test=test
for hint_type in hat
do

  for seed in 1 10 100 10000 100000
  do
    expt=HINT_${hint_type}_${dataset}_seed${seed}

    CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --dataset ${dataset} \
    --learning_rate 2e-5 \
    --seed ${seed} \
    --split train \
    --hint_type ${hint_type} \
    --split_test ${split_test} \
    --max_epochs 12 \
    --checkpoint_path saved_models/${expt} \
    --load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
    --vqa_loss_weight 1 \
    --hint_loss_weight 2 \
    --use_hint_loss > /hdd/robik/scr_experiments/${expt}.log
  done
done
