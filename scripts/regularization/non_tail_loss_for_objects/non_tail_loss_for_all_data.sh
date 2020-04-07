#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

expt=non_tail_loss_for_all_data
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--learning_rate 0.00001 \
--split train \
--split_test test \
--max_epochs 6 \
--checkpoint_path saved_models/${expt} \
--load_checkpoint_path saved_models/pretrained_1_default/model-best.pth \
--use_non_tail_loss_for_objects > /hdd/robik/scr_experiments/${expt}.log

expt=non_tail_loss_for_all_data_lr_0.000001
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--learning_rate 0.000001 \
--split train \
--split_test test \
--max_epochs 6 \
--checkpoint_path saved_models/${expt} \
--load_checkpoint_path saved_models/pretrained_1_default/model-best.pth \
--use_non_tail_loss_for_objects > /hdd/robik/scr_experiments/${expt}.log

expt=non_tail_loss_for_all_data_wt_100
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--learning_rate 0.00001 \
--split train \
--split_test test \
--max_epochs 6 \
--checkpoint_path saved_models/${expt} \
--load_checkpoint_path saved_models/pretrained_1_default/model-best.pth \
--use_non_tail_loss_for_objects \
--non_tail_loss_weight_for_objects 100 > /hdd/robik/scr_experiments/${expt}.log
