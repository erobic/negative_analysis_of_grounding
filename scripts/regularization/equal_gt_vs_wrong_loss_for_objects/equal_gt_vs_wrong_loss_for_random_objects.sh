#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa


for ratio_data in 0.05 0.1 0.2; do
    hint_type=random_${ratio_data}
    expt=equal_gt_vs_wrong_loss_for_random_objects_${ratio_data}

    CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --learning_rate 0.00001 \
    --split train \
    --hint_type ${hint_type} \
    --split_test test \
    --max_epochs 15 \
    --checkpoint_path saved_models/${expt} \
    --load_checkpoint_path saved_models/pretrained_1_default/model-best.pth \
    --use_equal_gt_vs_wrong_loss_for_objects > /hdd/robik/scr_experiments/${expt}.log
done