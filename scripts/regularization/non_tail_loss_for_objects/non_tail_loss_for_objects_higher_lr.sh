#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

hint_type=caption_based_hints

for lr in 5e-5 5e-4 1e-4; do
    expt=non_tail_loss_for_objects_lr_${lr}

    CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --learning_rate ${lr} \
    --split train \
    --hint_type ${hint_type} \
    --split_test test \
    --max_epochs 12 \
    --checkpoint_path saved_models/${expt} \
    --load_checkpoint_path saved_models/pretrained_1_default/model-best.pth \
    --use_non_tail_loss_for_objects > /hdd/robik/scr_experiments/${expt}.log
done