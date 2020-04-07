#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqacp2
split_test=test

scr_loss_weight=3
scr_compare_loss_weight=2000

for seed in 2000 5000 6000 7000 8000
do
  for hint_type in caption_based_hints one_minus_caption_based_hints random_caption_based_hints; do

      expt=scr_${hint_type}_${dataset}_seed${seed}

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
      --use_scr_loss \
      --seed ${seed} \
      --scr_hint_loss_weight ${scr_loss_weight} > /hdd/robik/scr_experiments/${expt}_phase2.log

      learning_rate=1e-4
      CUDA_VISIBLE_DEVICES=0 python -u main.py \
      --dataset ${dataset} \
      --learning_rate ${learning_rate} \
      --split train \
      --hint_type ${hint_type} \
      --split_test ${split_test} \
      --max_epochs 9 \
      --checkpoint_path saved_models/${expt}/phase_3 \
      --load_checkpoint_path saved_models/${expt}/phase_2/model-best.pth \
      --scr_hint_loss_weight ${scr_loss_weight} \
      --use_scr_loss \
      --seed ${seed} \
      --scr_compare_loss_weight ${scr_compare_loss_weight} > /hdd/robik/scr_experiments/${expt}_phase_3.log
  done


  hint_type=random_caption_based_hints
  expt=scr_${hint_type}_var_rand_${dataset}_seed${seed}

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
  --seed ${seed} \
  --scr_hint_loss_weight ${scr_loss_weight} > /hdd/robik/scr_experiments/${expt}_phase2.log

  learning_rate=1e-4
  CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --dataset ${dataset} \
  --learning_rate ${learning_rate} \
  --split train \
  --hint_type ${hint_type} \
  --split_test ${split_test} \
  --max_epochs 9 \
  --checkpoint_path saved_models/${expt}/phase_3 \
  --load_checkpoint_path saved_models/${expt}/phase_2/model-best.pth \
  --change_scores_every_epoch \
  --scr_hint_loss_weight ${scr_loss_weight} \
  --use_scr_loss \
  --seed ${seed} \
  --scr_compare_loss_weight ${scr_compare_loss_weight} > /hdd/robik/scr_experiments/${expt}_phase_3.log
done