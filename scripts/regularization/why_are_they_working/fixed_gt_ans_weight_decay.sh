#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqacp2
split_test=test
hint_type=hat

for wd in 1e-4 1e-5; do
      expt=fixed_gt_ans_wd_${wd}

      CUDA_VISIBLE_DEVICES=0 python -u main.py \
      --dataset ${dataset} \
      --learning_rate 2e-5 \
      --split train \
      --hint_type ${hint_type} \
      --split_test ${split_test} \
      --max_epochs 12 \
      --checkpoint_path saved_models/${expt} \
      --load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
      --use_fixed_gt_ans_loss \
      --log_epochs 20 \
      --weight_decay ${wd} \
      --fixed_gt_ans_loss_weight 2 > /hdd/robik/scr_experiments/${expt}.log
done