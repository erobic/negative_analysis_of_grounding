#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqa2
split_test=val
hint_type=hat

for wt in 1 2; do
      expt=fixed_gt_ans_with_small_noise_wt${wt}_${hint_type}_${dataset}

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
      --use_fixed_gt_ans_loss > /hdd/robik/scr_experiments/${expt}.log
done