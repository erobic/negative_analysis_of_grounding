#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa

hint_type=caption_based_hints

for num_words in 2 4 6 8 10; do
  expt=equal_gt_vs_wrong_loss_for_words_num_words_${num_words}

  CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --learning_rate 0.00001 \
  --split train \
  --hint_type ${hint_type} \
  --split_test test \
  --max_epochs 12 \
  --checkpoint_path saved_models/${expt} \
  --load_checkpoint_path saved_models/pretrained_1_default/model-best.pth \
  --use_equal_gt_vs_wrong_loss_for_words \
  --num_most_sensitive_words ${num_words} > /hdd/robik/scr_experiments/${expt}.log
done