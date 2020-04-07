#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

dataset=vqacp2
split_test=test

for ratio in 0.01 0.02 0.03 0.04 0.05 #0.2 0.4 0.6 0.8 1.0
do
  expt=${dataset}_fixed_gt_ans_fixed_rand_ratio_${ratio}

  CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --dataset ${dataset} \
  --split train \
  --split_test ${split_test} \
  --max_epochs 9 \
  --checkpoint_path saved_models/${expt} \
  --load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
  --use_fixed_gt_ans_loss \
  --fixed_ans_scores 0 \
  --do_not_discard_items_without_hints \
  --fixed_gt_ans_loss_weight 2 \
  --fixed_random_subset_ratio ${ratio} > /hdd/robik/scr_experiments/${expt}.log
done