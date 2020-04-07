#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

for seed in 1 10 100 1000 10000
do
  for hint_type in hat hat_var_rand one_minus_hat random_hat
  do
    expt=HINT_${hint_type}_vqacp2_seed${seed}
    CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --dataset vqacp2 \
    --split train \
    --split_test test \
    --do_not_discard_items_without_hints \
    --checkpoint_path saved_models/${expt} \
    --load_checkpoint_path saved_models/${expt}/model-best.pth \
    --test_on_train
  done
done