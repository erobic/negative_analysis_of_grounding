#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

lr=1e-6

for lr in 1e-4 1e-6
do
    for weight in 1 0.1
    do
        expt=non_tail_loss_from_scratch_lr_${lr}_wt_${weight}

        CUDA_VISIBLE_DEVICES=0 python -u main.py \
        --learning_rate ${lr} \
        --split train \
        --split_test test \
        --max_epochs 35 \
        --checkpoint_path saved_models/${expt} \
        --use_non_tail_loss_for_objects \
        --non_tail_loss_weight_for_objects ${weight} > /hdd/robik/scr_experiments/${expt}.log
    done
done

