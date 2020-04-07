#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

for non_tail_loss_weight_for_objects in 0.01 0.1 1.0 10
do
    expt=non_tail_loss_for_all_data_weight_${non_tail_loss_weight_for_objects}
    
    CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --learning_rate 1e-6 \
    --split train \
    --split_test test \
    --max_epochs 6 \
    --checkpoint_path saved_models/${expt} \
    --load_checkpoint_path saved_models/pretrained_1_default/model-best.pth \
    --use_non_tail_loss_for_objects \
    --non_tail_loss_weight_for_objects ${non_tail_loss_weight_for_objects} > /hdd/robik/scr_experiments/${expt}.log
done

