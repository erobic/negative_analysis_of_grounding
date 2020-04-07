#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

hint_type=caption_based_hints

expt=absolute_non_tail_loss_for_entire_objects

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--learning_rate 0.00001 \
--split train \
--hint_type ${hint_type} \
--split_test test \
--max_epochs 12 \
--checkpoint_path saved_models/${expt} \
--load_checkpoint_path saved_models/pretrained_1_default/model-best.pth \
--use_non_tail_loss_for_objects \
--use_absolute_for_non_tail_loss > /hdd/robik/scr_experiments/${expt}.log
