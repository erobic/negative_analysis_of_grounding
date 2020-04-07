#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqa2
split_test=val

hint_type=hat

expt=HINT_${hint_type}_var_rand_${dataset}_full

CUDA_VISIBLE_DEVICES=2 python -u main.py \
--dataset ${dataset} \
--learning_rate 2e-5 \
--split train \
--hint_type ${hint_type} \
--split_test ${split_test} \
--max_epochs 10 \
--checkpoint_path saved_models/${expt} \
--load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
--vqa_loss_weight 1 \
--hint_loss_weight 2 \
--change_scores_every_epoch \
--do_not_discard_items_without_hints \
--use_hint_loss > /hdd/robik/scr_experiments/${expt}.log
#
#for hint_type in hat one_minus_hat random_hat; do
#
#    expt=HINT_${hint_type}_${dataset}_full
#
#    CUDA_VISIBLE_DEVICES=2 python -u main.py \
#    --dataset ${dataset} \
#    --learning_rate 2e-5 \
#    --split train \
#    --hint_type ${hint_type} \
#    --split_test ${split_test} \
#    --max_epochs 10 \
#    --checkpoint_path saved_models/${expt} \
#    --load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
#    --vqa_loss_weight 1 \
#    --hint_loss_weight 2 \
#    --do_not_discard_items_without_hints \
#    --use_hint_loss > /hdd/robik/scr_experiments/${expt}.log
#done
#
