#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqacp2
split_test=test


scr_loss_weight=3

hint_type=caption_based_hints
expt=scr_${hint_type}_${dataset}_search_p1${scr_loss_weight}

learning_rate=2e-5
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--dataset ${dataset} \
--learning_rate ${learning_rate} \
--split train \
--hint_type ${hint_type} \
--split_test ${split_test} \
--max_epochs 9 \
--checkpoint_path saved_models/${expt}/phase_2 \
--load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
--use_scr_loss \
--scr_hint_loss_weight ${scr_loss_weight} > /hdd/robik/scr_experiments/${expt}_phase2.log

for scr_loss_weight in 3
do
  for learning_rate in 5e-5
  do
    for scr_compare_loss_weight in 1500
    do
      suffix=_${scr_loss_weight}_${scr_compare_loss_weight}_${learning_rate}
      CUDA_VISIBLE_DEVICES=0 python -u main.py \
      --dataset ${dataset} \
      --learning_rate ${learning_rate} \
      --split train \
      --hint_type ${hint_type} \
      --split_test ${split_test} \
      --max_epochs 9 \
      --checkpoint_path saved_models/${expt}/phase_3_${suffix} \
      --load_checkpoint_path saved_models/${expt}/phase_2/model-best.pth \
      --scr_hint_loss_weight ${scr_loss_weight} \
      --use_scr_loss \
      --scr_compare_loss_weight ${scr_compare_loss_weight} > /hdd/robik/scr_experiments/${expt}_phase_3_${suffix}.log
    done
  done
done