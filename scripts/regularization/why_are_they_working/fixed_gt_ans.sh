#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqacp2
split_test=test
hint_type=hat

expt=fixed_gt_ans_0_0.1

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
--log_epochs 12 \
--fixed_ans_scores 0 0.1 \
--fixed_gt_ans_loss_weight 1e-2 > /hdd/robik/scr_experiments/${expt}.log

expt=fixed_gt_ans_0_0.1_0.2

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
--log_epochs 12 \
--fixed_ans_scores 0 0.1 0.2 \
--fixed_gt_ans_loss_weight 1e-2 > /hdd/robik/scr_experiments/${expt}.log