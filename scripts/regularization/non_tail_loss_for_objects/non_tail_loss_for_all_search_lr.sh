#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

lr=1e-7
for lr in 5e-7 1e-7
do
    expt=non_tail_loss_for_all_data_lr_${lr}

    CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --learning_rate ${lr} \
    --split train \
    --split_test test \
    --max_epochs 10 \
    --checkpoint_path saved_models/${expt} \
    --load_checkpoint_path saved_models/pretrained_1_default/model-best.pth \
    --use_non_tail_loss_for_objects > /hdd/robik/scr_experiments/${expt}.log
done

