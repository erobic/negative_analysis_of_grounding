#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

for expt in HINT_hat_vqa2 HINT_one_minus_hat_vqa2 HINT_random_hat_vqa2 HINT_hat_var_rand_vqa2
do
  CUDA_VISIBLE_DEVICES=1 python -u main.py \
  --dataset vqa2 \
  --split train \
  --hint_type hat \
  --split_test val \
  --checkpoint_path saved_models/${expt} \
  --load_checkpoint_path saved_models/${expt}/model-epoch-5.pth \
  --test
done

for expt in scr_caption_based_hints_vqa2 scr_one_minus_caption_based_hints_vqa2 scr_random_caption_based_hints_vqa2 scr_random_caption_based_hints_vqa2_var_rand
do
  CUDA_VISIBLE_DEVICES=1 python -u main.py \
  --dataset vqa2 \
  --split train \
  --hint_type hat \
  --split_test val \
  --checkpoint_path saved_models/${expt} \
  --load_checkpoint_path saved_models/${expt}/phase_3/model-epoch-6.pth \
  --test
done