#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

for num_most_sensitive_objects in 1 3 5 7 9
do
    expt=non_tail_loss_for_all_data_head_objs_${num_most_sensitive_objects}
    
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

