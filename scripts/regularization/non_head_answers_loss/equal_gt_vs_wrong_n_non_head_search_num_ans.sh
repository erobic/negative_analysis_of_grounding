#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
hint_type=caption_based_hints

for num_non_head_wrong_answers in 0 5 10 50 100; do
    expt=equal_gt_vs_wrong_n_non_head_num_ans_${num_non_head_wrong_answers}

    CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --learning_rate 0.00001 \
    --split train \
    --hint_type ${hint_type} \
    --split_test test \
    --max_epochs 15 \
    --checkpoint_path saved_models/${expt} \
    --load_checkpoint_path saved_models/pretrained_1_default/model-best.pth \
    --use_equal_gt_vs_wrong_loss_for_objects \
    --use_non_head_answers_loss \
    --num_non_head_wrong_answers ${num_non_head_wrong_answers} > /hdd/robik/scr_experiments/${expt}.log
done
