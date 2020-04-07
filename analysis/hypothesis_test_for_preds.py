import os
import pickle
import numpy as np
from scipy.stats import wilcoxon, ttest_ind
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
import csv
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency

hint_vqa2_expts = ['HINT_hat_vqa2',
                   'HINT_one_minus_hat_vqa2',
                   'HINT_random_hat_vqa2',
                   'HINT_hat_var_rand_vqa2']
scr_vqa2_expts = ['scr_caption_based_hints_vqa2',
                  'scr_one_minus_caption_based_hints_vqa2',
                  'scr_random_caption_based_hints_vqa2',
                  'scr_random_caption_based_hints_vqa2_var_rand']
scr_vqacp2_expts = ['baseline_vqacp2',
                    'scr_caption_based_hints_vqacp2',
                    'scr_one_minus_caption_based_hints_vqacp2',
                    'scr_random_caption_based_hints_vqacp2',
                    'scr_random_caption_based_hints_vqacp2_var_rand']

hint_expt_prefixes = ['hat', 'one_minus_hat', 'random_hat', 'hat_var_rand']
hint_vqacp2_expts = ['HINT_' + s + '_vqacp2' for s in hint_expt_prefixes]
# hint_seeds = [1, 10, 100, 1000, 10000]
hint_seeds = [1000]

scr_expt_prefixes = ['caption_based_hints',
                     'one_minus_caption_based_hints',
                     'random_caption_based_hints',
                     'random_caption_based_hints_var_rand']
scr_vqacp2_expts = ['scr_' + s + '_vqacp2' for s in scr_expt_prefixes]
scr_seeds = [1000, 2000, 3000, 4000, 5000]


def load_single_scores(root, expt_name):
    qid_to_scores = pickle.load(open(
        os.path.join(root, 'saved_models', expt_name, 'prediction_scores', f'final_qid_to_prediction_scores.pkl'),
        'rb'))
    qids = sorted(list(qid_to_scores.keys()))
    data = np.asarray([qid_to_scores[qid] for qid in qids])
    return data


def load_agreements(root, expt_name):
    qid_to_scores = pickle.load(open(
        os.path.join(root, 'saved_models', expt_name, 'prediction_scores', f'final_qid_to_human_agreement.pkl'), 'rb'))
    _qid_to_scores = {}
    qids = sorted(list(qid_to_scores.keys()))
    agreements = []
    for qid in qids:
        agreements += qid_to_scores[qid]
    return qid_to_scores, np.asarray(agreements)


def perform_mcnemar_test(qid_to_agreements1, qid_to_agreements2):
    table = np.zeros((2, 2))
    qids = list(qid_to_agreements1.keys())
    for qid in qids:
        for _agree1, _agree2 in zip(qid_to_agreements1[qid], qid_to_agreements2[qid]):
            if _agree1 > 0 and _agree2 > 0:
                table[0][0] += 1
            elif _agree1 > _agree2:
                table[0][1] += 1
            elif _agree2 > _agree1:
                table[1][0] += 1
            else:
                table[1][1] += 1
    results = mcnemar(table, exact=False, correction=True)
    return table, results.pvalue


def compare_agreements(root, expt_list):
    for ix1, expt1 in enumerate(expt_list):
        for ix2, expt2 in enumerate(expt_list):
            if ix1 >= ix2:
                continue
            qid_to_agreements1, agreements1 = load_agreements(root, expt1)
            qid_to_agreements2, agreements2 = load_agreements(root, expt2)
            msg = f"{expt1} vs {expt2}: "
            table, p = perform_mcnemar_test(qid_to_agreements1, qid_to_agreements2)
            msg += " mcnemar p: %.4f " % (p)
            msg += f" mcnemar table: {table} "
            msg += " percent_intersecting_predictions: %.2f " % (
                    percent_intersecting_predictions(agreements1, agreements2) * 100)
            msg += " percent_intersecting_correct_predictions: %.2f " % (
                    percent_intersecting_correct_predictions(agreements1, agreements2) * 100)
            print(msg)


def percent_intersecting_predictions(preds1, preds2):
    binary_pred1, binary_pred2 = (preds1 > 0), (preds2 > 0)
    intersect = binary_pred1 == binary_pred2
    num_intersect = len(np.nonzero(intersect)[0])
    num_union = len(preds1)
    return num_intersect / num_union


def percent_intersecting_predictions_multiple_experiments(preds1_multi_expts, preds2_multi_expts):
    total_intersect, total_union = 0, 0
    for preds1, preds2 in zip(preds1_multi_expts, preds2_multi_expts):
        binary_pred1, binary_pred2 = (preds1 > 0), (preds2 > 0)
        intersect = binary_pred1 == binary_pred2
        total_intersect += len(np.nonzero(intersect)[0])
        total_union += len(preds1)

    return total_intersect / total_union


def percent_intersecting_correct_predictions(preds1, preds2):
    binary_pred1, binary_pred2 = (preds1 > 0), (preds2 > 0)
    correct1, correct2 = np.nonzero(binary_pred1)[0], np.nonzero(binary_pred2)[0]
    intersection = np.intersect1d(correct1, correct2)
    union = np.union1d(correct1, correct2)
    return len(intersection) / len(union)


def percent_intersecting_correct_predictions_multiple_experiments(preds1_multi_expts, preds2_multi_expts):
    total_intersect, total_union = 0, 0
    for preds1, preds2 in zip(preds1_multi_expts, preds2_multi_expts):
        binary_pred1, binary_pred2 = (preds1 > 0), (preds2 > 0)
        correct1, correct2 = np.nonzero(binary_pred1)[0], np.nonzero(binary_pred2)[0]
        total_intersect += np.intersect1d(correct1, correct2)
        total_union += np.union1d(correct1, correct2)
    return len(total_intersect) / len(total_union)


def subsample_accuracies(preds1, preds2, subsets, rand_ixs):
    subset_ixs = np.array_split(rand_ixs, indices_or_sections=subsets)
    accuracies1, accuracies2 = [], []
    for curr_subset_ixs in subset_ixs:
        preds1_subset, preds2_subset = preds1[curr_subset_ixs], preds2[curr_subset_ixs]
        # preds1_subset = np.where(preds1_subset > 0, np.ones_like(preds1_subset), np.zeros_like(preds1_subset))
        # preds2_subset = np.where(preds2_subset > 0, np.ones_like(preds2_subset), np.zeros_like(preds2_subset))
        acc1, acc2 = np.sum(preds1_subset) / len(preds1_subset), np.sum(preds2_subset) / len(preds2_subset)
        accuracies1.append(acc1)
        accuracies2.append(acc2)
    return accuracies1, accuracies2


def perform_tests_with_single_experiments(preds1, preds2, subsets, rand_ixs):
    accuracies1, accuracies2 = subsample_accuracies(preds1, preds2, subsets, rand_ixs)
    _, ttest_p = ttest_ind(accuracies1, accuracies2, equal_var=False)
    _, wilcoxon_p = wilcoxon(accuracies1, accuracies2)
    acc1, acc2 = np.sum(preds1) / len(preds1), np.sum(preds2) / len(preds2)
    return ttest_p, wilcoxon_p, acc1, acc2


def compare_with_single_experiments(root, expt_list, subsets, rand_ixs):
    for ix1, expt1 in enumerate(expt_list):
        for ix2, expt2 in enumerate(expt_list):
            if ix1 >= ix2:
                continue
            preds1 = load_single_scores(root, expt1)
            preds2 = load_single_scores(root, expt2)
            msg = f"{expt1} vs {expt2}: "
            ttest_p, wilcoxon_p, acc1, acc2 = perform_tests_with_single_experiments(preds1, preds2, subsets=subsets,
                                                                                    rand_ixs=rand_ixs)
            msg += f" t-test p: {ttest_p} wilcoxon p: {wilcoxon_p}"

            msg += " percent_intersecting_predictions: %.2f " % (percent_intersecting_predictions(preds1, preds2) * 100)

            msg += " percent_intersecting_correct_predictions: %.2f " % (
                    percent_intersecting_correct_predictions(preds1, preds2) * 100)
            msg += " acc1: %.4f acc2: %.4f " % (acc1, acc2)
            print(msg)


def perform_tests_with_multiple_experiments(preds1_multi_seeds, preds2_multi_seeds, subsets, rand_ixs):
    all_accuracies1, all_accuracies2 = [], []
    for preds1, preds2 in zip(preds1_multi_seeds, preds2_multi_seeds):
        accuracies1, accuracies2 = subsample_accuracies(preds1, preds2, subsets, rand_ixs)
        all_accuracies1.append(accuracies1), all_accuracies2.append(accuracies2)
    mean_accuracies1, mean_accuracies2 = np.mean(np.array(all_accuracies1), axis=0), \
                                         np.mean(np.array(all_accuracies2), axis=0)
    _, ttest_p = ttest_ind(mean_accuracies1, mean_accuracies2, equal_var=False)
    _, wilcoxon_p = wilcoxon(mean_accuracies1, mean_accuracies2)
    acc1, acc2 = np.mean(all_accuracies1), np.mean(all_accuracies2)
    return ttest_p, wilcoxon_p, acc1, acc2


def compare_with_multiple_experiments(root, expt_list, seed_list, subsets, rand_ixs):
    for ix1, expt1 in enumerate(expt_list):
        for ix2, expt2 in enumerate(expt_list):
            if ix1 >= ix2:
                continue

            # Load all of the results from different seeds
            preds1_multi_expts, preds2_multi_expts = [], []
            for seed in seed_list:
                expt_seed1 = expt1 + f"_seed{seed}"
                expt_seed2 = expt2 + f"_seed{seed}"
                preds1_multi_expts.append(load_single_scores(root, expt_seed1))
                preds2_multi_expts.append(load_single_scores(root, expt_seed2))

            msg = f"{expt1} vs {expt2}:"
            ttest_p, wilcoxon_p, acc1, acc2 = perform_tests_with_multiple_experiments(preds1_multi_expts,
                                                                                      preds2_multi_expts,
                                                                                      subsets=subsets,
                                                                                      rand_ixs=rand_ixs)
            msg += f" t-test p: {ttest_p} wilcoxon p: {wilcoxon_p}"

            msg += " percent_intersecting_predictions: %.2f " % (
                    percent_intersecting_predictions_multiple_experiments(preds1_multi_expts,
                                                                          preds2_multi_expts) * 100)
            msg += " acc1: %.4f acc2: %.4f " % (acc1, acc2)
            print(msg)


def compare_baseline_against_multiple_experiments(root, baseline_expt, expt_list, seed_list, subsets, rand_ixs):
    baseline_expts = []
    for seed in seed_list:
        baseline_expts.append(load_single_scores(root, baseline_expt))

    for ix1, expt1 in enumerate(expt_list):

        # Load all of the results from different seeds
        preds1_multi_expts = []
        for seed in seed_list:
            expt_seed1 = expt1 + f"_seed{seed}"
            preds1_multi_expts.append(load_single_scores(root, expt_seed1))

        msg = f"{expt1} vs {baseline_expt}:"
        ttest_p, wilcoxon_p, acc1, acc2 = perform_tests_with_multiple_experiments(preds1_multi_expts,
                                                                                  baseline_expts,
                                                                                  subsets=subsets,
                                                                                  rand_ixs=rand_ixs)
        msg += f" t-test p: {ttest_p} wilcoxon p: {wilcoxon_p}"

        msg += " percent_intersecting_predictions: %.2f " % (
                percent_intersecting_predictions_multiple_experiments(preds1_multi_expts,
                                                                      baseline_expts) * 100)
        msg += " acc1: %.4f acc2: %.4f " % (acc1, acc2)
        print(msg)


def plot_distribution(preds1, preds2, title1, title2, subsets, bins=100):
    accuracies1, accuracies2 = subsample_accuracies(preds1, preds2, subsets)
    accuracies1, accuracies2 = np.asarray(accuracies1) * 100, np.asarray(accuracies2) * 100
    plt.figure()
    fig, axes = plt.subplots(1, 2, sharey=True)
    axes[0].hist(accuracies1, bins=bins)
    mean1, std1 = np.mean(accuracies1), np.std(accuracies1)
    axes[0].set_title(title1 + f"\n $\mu$=%.2f $\sigma$=%.2f " % (mean1, std1))

    axes[1].hist(accuracies2, bins=bins)
    mean2, std2 = np.mean(accuracies2), np.std(accuracies2)
    axes[1].set_title(title2 + f"\n $\mu$=%.2f $\sigma$=%.2f " % (mean2, std2))

    if not os.path.exists('histograms'):
        os.makedirs('histograms')
    plt.savefig(os.path.join('histograms', title1 + ' vs. ' + title2 + '.png'))
    plt.tight_layout()


def convert_to_csv(root, expt_name):
    qid_to_scores = pickle.load(open(
        os.path.join(root, 'saved_models', expt_name, 'prediction_scores', f'final_qid_to_prediction_scores.pkl'),
        'rb'))
    qid_to_scores_list = sorted(list(qid_to_scores.keys()))

    with open(os.path.join(root, 'saved_models', expt_name, 'prediction_scores', f'{expt_name}.csv'),
              'w') as f:
        writer = csv.writer(f)
        for qid in qid_to_scores_list:
            writer.writerow([qid, qid_to_scores[qid]])


if __name__ == "__main__":
    root = '/hdd/robik/projects/self_critical_vqa'
    subsets = 5000
    np.random.seed(0)
    rand_ixs = np.arange(0, 219928)
    np.random.shuffle(rand_ixs)

    # Get variance within multiple runs of HINT
    # compare_with_single_experiments(root, ['HINT_hat_vqacp2_seed' + str(s) for s in hint_seeds], subsets=subsets,
    #                                 rand_ixs=rand_ixs)

    # Compare among multiple runs of different variants of HINT
    # compare_with_multiple_experiments(root, hint_vqacp2_expts, hint_seeds, subsets, rand_ixs)

    # Compare HINT variants against the baseline
    compare_baseline_against_multiple_experiments(root,
                                                  'baseline_vqacp2',
                                                  ['HINT_hat_vqacp2',
                                                   'HINT_one_minus_hat_vqacp2',
                                                   'HINT_random_hat_vqacp2',
                                                   'HINT_hat_var_rand_vqacp2'],
                                                  hint_seeds,
                                                  subsets,
                                                  rand_ixs)

    # Compare SCR variants against the baseline
    # compare_baseline_against_multiple_experiments(root,
    #                                               'baseline_vqacp2',
    #                                               ['scr_caption_based_hints_vqacp2',
    #                                                'scr_one_minus_caption_based_hints_vqacp2',
    #                                                'scr_random_caption_based_hints_vqacp2',
    #                                                'scr_random_caption_based_hints_var_rand_vqacp2'],
    #                                               scr_seeds,
    #                                               subsets,
    #                                               rand_ixs)

    # # Compare the predictions from our approach against SCR
    # compare_baseline_against_multiple_experiments(root,
    #                                               'vqacp2_fixed_gt_ans_fixed_rand_ratio_0.01',
    #                                               ['scr_caption_based_hints_vqacp2'],
    #                                               scr_seeds,
    #                                               subsets,
    #                                               rand_ixs)
    # scr_caption_based_hints_vqacp2 vs vqacp2_fixed_gt_ans_fixed_rand_ratio_0.01: t-test p: 0.1544743129705349 wilcoxon p: 0.010471045914657758 percent_intersecting_predictions: 87.34  acc1: 0.4910 acc2: 0.4890

    # Compare the predictions from our approach against HINT
    # compare_baseline_against_multiple_experiments(root,
    #                                               'vqacp2_fixed_gt_ans_fixed_rand_ratio_0.01',
    #                                               ['HINT_hat_vqacp2'],
    #                                               hint_seeds,
    #                                               subsets,
    #                                               rand_ixs)
