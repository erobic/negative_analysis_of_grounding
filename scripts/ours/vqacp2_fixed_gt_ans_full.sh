#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

dataset=vqacp2
split_test=test

expt=${dataset}_fixed_gt_ans_full

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--dataset ${dataset} \
--learning_rate 1e-6 \
--split train \
--split_test ${split_test} \
--max_epochs 12 \
--checkpoint_path saved_models/${expt} \
--load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
--use_fixed_gt_ans_loss \
--fixed_ans_scores 0 \
--do_not_discard_items_without_hints \
--fixed_gt_ans_loss_weight 2 \
--fixed_random_subset_ratio 1.0 > /hdd/robik/scr_experiments/${expt}.log

nohup ./scripts/main_table/ours/vqa2_fixed_gt_ans_full.sh &