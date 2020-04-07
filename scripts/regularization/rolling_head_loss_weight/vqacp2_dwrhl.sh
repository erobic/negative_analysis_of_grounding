#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqacp2
split_test=test

for learning_rate in 2e-5; do
  for weight in 25 50 100; do
    for obj in 9 13 17; do
      expt=dwrhl_${dataset}_lr${learning_rate}_weight${weight}_objs${obj}

      CUDA_VISIBLE_DEVICES=0 python -u main.py \
      --dataset ${dataset} \
      --hint_type hat \
      --learning_rate ${learning_rate} \
      --split train \
      --split_test ${split_test} \
      --max_epochs 12 \
      --checkpoint_path saved_models/${expt} \
      --load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
      --use_rolling_head_loss_for_objects \
      --num_most_sensitive_objects ${obj} \
      --dynamically_weight_rolling_head_loss \
      --rolling_head_loss_weight_for_objects ${weight}  > /hdd/robik/scr_experiments/${expt}.log
    done
  done
done

#--load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \

