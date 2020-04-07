#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqa2
split_test=val

scr_loss_weight=2
scr_compare_loss_weight=2000

#for hint_type in caption_based_hints one_minus_caption_based_hints random_caption_based_hints; do
#
#    expt=scr_${hint_type}_${dataset}_full
#
#    learning_rate=5e-5
#    CUDA_VISIBLE_DEVICES=0 python -u main.py \
#    --dataset ${dataset} \
#    --learning_rate ${learning_rate} \
#    --split train \
#    --hint_type ${hint_type} \
#    --split_test ${split_test} \
#    --max_epochs 7 \
#    --checkpoint_path saved_models/${expt}/phase_2 \
#    --load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
#    --use_scr_loss \
#    --do_not_discard_items_without_hints \
#    --scr_hint_loss_weight ${scr_loss_weight} > /hdd/robik/scr_experiments/${expt}_phase2.log
#
#    learning_rate=1e-4
#    CUDA_VISIBLE_DEVICES=0 python -u main.py \
#    --dataset ${dataset} \
#    --learning_rate ${learning_rate} \
#    --split train \
#    --hint_type ${hint_type} \
#    --split_test ${split_test} \
#    --max_epochs 7 \
#    --checkpoint_path saved_models/${expt}/phase_3 \
#    --load_checkpoint_path saved_models/${expt}/phase_2/model-best.pth \
#    --scr_hint_loss_weight ${scr_loss_weight} \
#    --use_scr_loss \
#    --do_not_discard_items_without_hints \
#    --scr_compare_loss_weight ${scr_compare_loss_weight} > /hdd/robik/scr_experiments/${expt}_phase_3.log
#done


hint_type=random_caption_based_hints
expt=scr_${hint_type}_${dataset}_var_rand_full

learning_rate=5e-5
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--dataset ${dataset} \
--learning_rate ${learning_rate} \
--split train \
--hint_type ${hint_type} \
--split_test ${split_test} \
--max_epochs 7 \
--checkpoint_path saved_models/${expt}/phase_2 \
--load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
--change_scores_every_epoch \
--use_scr_loss \
--do_not_discard_items_without_hints \
--scr_hint_loss_weight ${scr_loss_weight} > /hdd/robik/scr_experiments/${expt}_phase2.log

learning_rate=1e-4
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--dataset ${dataset} \
--learning_rate ${learning_rate} \
--split train \
--hint_type ${hint_type} \
--split_test ${split_test} \
--max_epochs 7 \
--checkpoint_path saved_models/${expt}/phase_3 \
--load_checkpoint_path saved_models/${expt}/phase_2/model-best.pth \
--change_scores_every_epoch \
--scr_hint_loss_weight ${scr_loss_weight} \
--use_scr_loss \
--do_not_discard_items_without_hints \
--scr_compare_loss_weight ${scr_compare_loss_weight} > /hdd/robik/scr_experiments/${expt}_phase_3.log