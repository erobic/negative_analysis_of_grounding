#!/bin/bash
source activate self_critical_vqa
cd /hdd/robik/projects/self_critical_vqa
dataset=vqacp2
split_test=test
hint_type=hat

for seed in 1 100 10000; do
      expt=fixed_gt_ans_seed_${seed}

      CUDA_VISIBLE_DEVICES=0 python -u main.py \
      --dataset ${dataset} \
      --learning_rate 2e-5 \
      --seed ${seed} \
      --split train \
      --hint_type ${hint_type} \
      --split_test ${split_test} \
      --max_epochs 8 \
      --checkpoint_path saved_models/${expt} \
      --load_checkpoint_path saved_models/baseline_${dataset}/model-best.pth \
      --use_fixed_gt_ans_loss \
      --log_epochs 20 \
      --fixed_gt_ans_loss_weight 2 > /hdd/robik/scr_experiments/${expt}.log
done