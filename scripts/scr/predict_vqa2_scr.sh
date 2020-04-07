#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqa2
split_test=val


scr_loss_weight=2
scr_compare_loss_weight=2000

for hint_type in caption_based_hints one_minus_caption_based_hints random_caption_based_hints; do

    expt=scr_${hint_type}_${dataset}

    learning_rate=5e-5
    CUDA_VISIBLE_DEVICES=2 python -u main.py \
    --dataset ${dataset} \
    --learning_rate ${learning_rate} \
    --split train \
    --hint_type ${hint_type} \
    --split_test ${split_test} \
    --max_epochs 12 \
    --checkpoint_path saved_models/${expt}/phase_3 \
    --load_checkpoint_path saved_models/${expt}/phase_3/model-best.pth \
    --use_scr_loss \
    --scr_hint_loss_weight ${scr_loss_weight} \
    --test
done


hint_type=random_caption_based_hints
expt=scr_${hint_type}_${dataset}_var_rand

learning_rate=5e-5
CUDA_VISIBLE_DEVICES=2 python -u main.py \
--dataset ${dataset} \
--learning_rate ${learning_rate} \
--split train \
--hint_type ${hint_type} \
--split_test ${split_test} \
--max_epochs 12 \
--checkpoint_path saved_models/${expt}/phase_3 \
--load_checkpoint_path saved_models/${expt}/phase_3/model-best.pth \
--change_scores_every_epoch \
--use_scr_loss \
--scr_hint_loss_weight ${scr_loss_weight} \
--test
