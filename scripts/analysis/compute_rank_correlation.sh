#!/bin/bash
set -e
source scripts/common.sh

### Sample for HINT
#python -u analysis/compute_rank_correlation_coefficient.py \
#--data_dir ${DATA_DIR} \
#--sensitivity_file ${SAVE_DIR}/HINT_hat_vqacp2/sensitivities/_qid_to_gt_ans_sensitivities_epoch_7.pkl \
#--hint_type hat


### Sample for SCR
#python -u analysis/compute_rank_correlation_coefficient.py \
#--data_dir ${DATA_DIR} \
#--sensitivity_file ${SAVE_DIR}/scr_caption_based_hints_vqacp2/phase_3/sensitivities/_qid_to_gt_ans_sensitivities_epoch_7.pkl \
#--hint_type caption_based_hints

## For our regularizer
python -u analysis/compute_rank_correlation_coefficient.py \
--data_dir ${DATA_DIR} \
--sensitivity_file ${SAVE_DIR}/vqa2_zero_out_full/sensitivities/_qid_to_gt_ans_sensitivities_epoch_6.pkl \
--hint_type hat


