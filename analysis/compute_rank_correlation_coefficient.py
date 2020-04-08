from scipy.stats import spearmanr
import pickle
import json
import os
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--sensitivity_file', type=str, required=True)
    parser.add_argument('--hint_type', type=str, required=True)
    args = parser.parse_args()


    sensitivity_scores = pickle.load(open(args.sensitivity_file, 'rb'))
    train_gt_scores = pickle.load(open(os.path.join(args.data_dir, 'hints', f'train_{args.hint_type}.pkl'), 'rb'))
    val_gt_scores = pickle.load(open(os.path.join(args.data_dir, 'hints', f'val_{args.hint_type}.pkl'), 'rb'))
    gt_scores = {}
    for qid in train_gt_scores:
        gt_scores[qid] = train_gt_scores[qid]
    for qid in val_gt_scores:
        gt_scores[qid] = val_gt_scores[qid]

    total_coef, total_p, total_num = 0, 0, 0
    not_found = 0

    for qid in sensitivity_scores:
        if qid in gt_scores:
            _gt_scores = gt_scores[qid].tolist()
            _qid_sensitivity_scores = sensitivity_scores[qid]
            coef, p = spearmanr(_gt_scores, _qid_sensitivity_scores)
            if np.isnan(coef):
                continue
            total_coef += coef
            total_p += p
            total_num += 1
        else:
            not_found += 1
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