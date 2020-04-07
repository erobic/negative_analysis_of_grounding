from scipy.stats import spearmanr
import pickle
import json
import os
import numpy as np


if __name__ == "__main__":
    expt_configs = {
        # Baselines
        'baseline_vqa2': {
            'sensitivity_file': 'test_qid_to_gt_ans_sensitivities_epoch_35',
            'hint_type': 'hat'
        },
        'baseline_vqacp2': {
            'sensitivity_file': 'test_qid_to_gt_ans_sensitivities_epoch_28',
            'hint_type': 'hat'
        },

        # HINT
        'HINT_hat_vqa2': {
            'sensitivity_file': 'qid_to_gt_ans_sensitivities_epoch_7',
            'hint_type': 'hat'
        },
        'HINT_one_minus_hat_vqa2': {
            'sensitivity_file': 'qid_to_gt_ans_sensitivities_epoch_7',
            'hint_type': 'hat'
        },
        'HINT_random_hat_vqa2': {
            'sensitivity_file': 'qid_to_gt_ans_sensitivities_epoch_7',
            'hint_type': 'hat'
        },
        'HINT_hat_var_rand_vqa2': {
            'sensitivity_file': 'qid_to_gt_ans_sensitivities_epoch_7',
            'hint_type': 'hat'
        },
        'HINT_hat_vqacp2_seed1': {
            'sensitivity_file': 'qid_to_gt_ans_sensitivities_epoch_7',
            'hint_type': 'hat'
        },
        'HINT_one_minus_hat_vqacp2_seed1': {
            'sensitivity_file': 'qid_to_gt_ans_sensitivities_epoch_7',
            'hint_type': 'hat'
        },
        'HINT_random_hat_vqacp2_seed1': {
            'sensitivity_file': 'qid_to_gt_ans_sensitivities_epoch_7',
            'hint_type': 'hat'
        },
        'HINT_hat_var_rand_vqacp2_seed1': {
            'sensitivity_file': 'qid_to_gt_ans_sensitivities_epoch_7',
            'hint_type': 'hat'
        },

        # SCR
        'scr_caption_based_hints_vqacp2_seed4000/phase_3': {
            'sensitivity_file': 'qid_to_gt_ans_sensitivities_epoch_7',
            'hint_type': 'caption_based_hints'
        },
        'scr_one_minus_caption_based_hints_vqacp2_seed4000/phase_3': {
            'sensitivity_file': 'qid_to_gt_ans_sensitivities_epoch_7',
            'hint_type': 'caption_based_hints'
        },
        'scr_random_caption_based_hints_vqacp2_seed4000/phase_3': {
            'sensitivity_file': 'qid_to_gt_ans_sensitivities_epoch_7',
            'hint_type': 'caption_based_hints'
        },
        'scr_random_caption_based_hints_var_rand_vqacp2_seed4000/phase_3': {
            'sensitivity_file': 'qid_to_gt_ans_sensitivities_epoch_7',
            'hint_type': 'caption_based_hints'
        },

        # SCR on VQAv2
        'scr_caption_based_hints_vqa2/phase_3': {
            'sensitivity_file': 'qid_to_all_ans_sensitivities_epoch_7',
            'hint_type': 'caption_based_hints'
        },
        'scr_one_minus_caption_based_hints_vqa2/phase_3': {
            'sensitivity_file': 'qid_to_all_ans_sensitivities_epoch_7',
            'hint_type': 'caption_based_hints'
        },
        'scr_random_caption_based_hints_vqa2/phase_3': {
            'sensitivity_file': 'qid_to_all_ans_sensitivities_epoch_7',
            'hint_type': 'caption_based_hints'
        },
        'scr_random_caption_based_hints_vqa2_var_rand/phase_3': {
            'sensitivity_file': 'qid_to_all_ans_sensitivities_epoch_7',
            'hint_type': 'caption_based_hints'
        },

        # Ours
        'vqacp2_fixed_gt_ans_fixed_rand_ratio_0.01': {
            'sensitivity_file': 'test_qid_to_gt_ans_sensitivities_epoch_6',
            'hint_type': 'hat' # Did not use during training, just checking correlations anyways
        },
        'vqa2_fixed_gt_ans_full': {
            'sensitivity_file': 'test_qid_to_gt_ans_sensitivities_epoch_6',
            'hint_type': 'hat'  # Did not use during training, just checking correlations anyways
        }
    }

    root = '/hdd/robik/projects/self_critical_vqa'
    expt_name = 'vqa2_fixed_gt_ans_full'
    scores_name = expt_configs[expt_name]['sensitivity_file']
    hint_type = expt_configs[expt_name]['hint_type']

    if os.path.exists(os.path.join(root, 'saved_models', expt_name, 'sensitivities', scores_name + '.pkl')):
        pred_scores = pickle.load(
            open(os.path.join(root, 'saved_models', expt_name, 'sensitivities', scores_name + '.pkl'), 'rb'))
    else:
        pred_scores = pickle.load(
            open(os.path.join(root, 'saved_models', expt_name, 'sensitivities', scores_name + '.json'), 'rb'))

    train_gt_scores = pickle.load(open(os.path.join(root, 'data', 'hints', f'train_{hint_type}.pkl'), 'rb'))
    val_gt_scores = pickle.load(open(os.path.join(root, 'data', 'hints', f'val_{hint_type}.pkl'), 'rb'))
    gt_scores = {}
    for qid in train_gt_scores:
        gt_scores[qid] = train_gt_scores[qid]
    for qid in val_gt_scores:
        gt_scores[qid] = val_gt_scores[qid]

    total_coef, total_p, total_num = 0, 0, 0
    not_found = 0

    for qid in pred_scores:
        if qid in gt_scores:
            _gt_scores = gt_scores[qid].tolist()
            _pred_scores = pred_scores[qid]
            coef, p = spearmanr(_gt_scores, _pred_scores)
            if np.isnan(coef):
                continue
            total_coef += coef
            total_p += p
            total_num += 1
        else:
            # print(f"{qid} not found")
            not_found += 1
    # print(f"not found {not_found}")
    coef = total_coef / total_num
    p = total_p / total_num
    print(f"Coefficient {coef} p =  {p} total_num {total_num} not_Found {not_found}")

# HINT
# Baseline: Coefficient 0.01173869140468033 p =  0.29164488316222886 total_num 18746
# HINT_hat_vqacp2: Coefficient 0.10260887017293162 p =  0.2823486704229709 total_num 18746
# HINT_one_minus_hat_vqacp2: Coefficient 0.003560927275527529 p =  0.29724002652052667 total_num 18746
# HINT_random_hat_vqacp2: Coefficient 0.05331055002236303 p =  0.2976860817152182 total_num 18746
# HINT_hat_var_rand_vqacp2: Coefficient 0.057381987267413614 p =  0.2957991544215842 total_num 18746

# SCR
# baseline_vqacp2: Coefficient -0.01876512390000249 p =  0.3060643100374861 total_num 9709
# scr_caption_based_hints_vqacp2: Coefficient 0.04109482185321172 p =  0.3074650061537316 total_num 9709
# scr_one_minus_caption_based_hints_vqacp2: Coefficient -0.028321333883055636 p =  0.3086574984061775 total_num 9709
# scr_random_caption_based_hints_vqacp2: Coefficient -0.023545167733254744 p =  0.30409633607181824 total_num 9709
# scr_random_caption_based_hints_vqacp2_var_rand: Coefficient -0.020078364397889837 p =  0.30521900844249 total_num 9709


# HINT_hat_vqacp2_full Coefficient -0.26659903336279145 p =  0.21458213341556948 total_num 19272 not_Found 40185
# Coefficient -0.23067190987860503 p =  0.23495274127889637 total_num 19272 not_Found 40185