#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

dataset=vqacp2
split_test=test

for ratio in 0.05
do
  expt=${dataset}_fixed_gt_ans_fixed_rand_ratio_${ratio}

  CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --dataset ${dataset} \
  --split train \
  --split_test ${split_test} \
  --checkpoint_path saved_models/${expt} \
  --load_checkpoint_path saved_models/${expt}/model-epoch-6.pth \
  --use_fixed_gt_ans_loss \
  --fixed_ans_scores 0 \
  --do_not_discard_items_without_hints \
  --fixed_gt_ans_loss_weight 2 \
  --var_random_subset_ratio ${ratio} \
  --test
done