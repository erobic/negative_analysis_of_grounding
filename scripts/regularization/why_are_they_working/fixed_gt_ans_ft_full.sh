#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqacp2
split_test=test
hint_type=hat

expt=fixed_gt_ans_flip_ft_full

CUDA_VISIBLE_DEVICES=1 python -u main.py \
--dataset ${dataset} \
--learning_rate 1e-6 \
--split train \
--hint_type ${hint_type} \
--split_test ${split_test} \
--max_epochs 12 \
--checkpoint_path saved_models/${expt} \
--load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
--use_fixed_gt_ans_loss > /hdd/robik/scr_experiments/${expt}.log