#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqacp2
split_test=test
hint_type=hat

for ans_score in 0 0.3 0.5; do
  for wt in 1 5 0.5; do
        expt=fixed_gt_ans_${ans_score}_wt${wt}_${hint_type}_${dataset}

        CUDA_VISIBLE_DEVICES=1 python -u main.py \
        --dataset ${dataset} \
        --learning_rate 2e-5 \
        --split train \
        --hint_type ${hint_type} \
        --split_test ${split_test} \
        --max_epochs 12 \
        --checkpoint_path saved_models/${expt} \
        --load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
        --vqa_loss_weight 1 \
        --use_fixed_gt_ans_loss \
        --lr_gamma 0.5 \
        --fixed_gt_ans_loss_weight ${wt} > /hdd/robik/scr_experiments/${expt}.log
    done
done