#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqacp2
split_test=test

scr_compare_loss_weight=4000
seed=1
hint_type=caption_based_hints
  #for seed in 1 10 100 10000 100000
for scr_loss_weight in 1 2 # 3 4
do
  expt=scr_${hint_type}_${dataset}_seed${seed}_wt${scr_loss_weight}

  learning_rate=5e-5
  CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --dataset ${dataset} \
  --learning_rate ${learning_rate} \
  --split train \
  --hint_type ${hint_type} \
  --split_test ${split_test} \
  --max_epochs 7 \
  --checkpoint_path saved_models/${expt}/phase_2 \
  --load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
  --use_scr_loss \
  --seed ${seed} \
  --scr_hint_loss_weight ${scr_loss_weight} > /hdd/robik/scr_experiments/${expt}_phase2.log

  learning_rate=1e-4
  CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --dataset ${dataset} \
  --learning_rate ${learning_rate} \
  --split train \
  --hint_type ${hint_type} \
  --split_test ${split_test} \
  --max_epochs 12 \
  --checkpoint_path saved_models/${expt}/phase_3 \
  --load_checkpoint_path saved_models/${expt}/phase_2/model-best.pth \
  --scr_hint_loss_weight ${scr_loss_weight} \
  --use_scr_loss \
  --seed ${seed} \
  --scr_compare_loss_weight ${scr_compare_loss_weight} > /hdd/robik/scr_experiments/${expt}_phase_3.log
done