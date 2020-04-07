#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqacp2
split_test=test
hint_type=hat

for wt in 1e- 3e-4; do
      expt=random_gt_ans_wt${wt}_${hint_type}_${dataset}

      CUDA_VISIBLE_DEVICES=0 python -u main.py \
      --dataset ${dataset} \
      --learning_rate 2e-5 \
      --split train \
      --hint_type ${hint_type} \
      --split_test ${split_test} \
      --max_epochs 8 \
      --checkpoint_path saved_models/${expt} \
      --load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
      --vqa_loss_weight 1 \
      --use_random_gt_ans_loss \
      --random_gt_ans_loss_weight ${wt}  > /hdd/robik/scr_experiments/${expt}.log
done