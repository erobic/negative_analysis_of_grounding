import argparse
import json
import os
import pickle


def get_per_answer_type_metrics(qid_to_ann, qid_to_preds):
    per_type_pred_scores = {}
    per_type_total = {}
    for qid in qid_to_preds:
        ann = qid_to_ann[qid]
        ans_type = ann['answer_type']
        if ans_type not in per_type_pred_scores:
            per_type_pred_scores[ans_type] = 0
            per_type_total[ans_type] = 0
        per_type_pred_scores[ans_type] += qid_to_preds[qid]
        per_type_total[ans_type] += 1
    per_type_final_scores = {}
    for at in per_type_pred_scores:
        per_type_final_scores[at] = per_type_pred_scores[at] / per_type_total[at]
    print(json.dumps(per_type_final_scores, sort_keys=True, indent=4))
    print(f"Total")


def compute_metrics(data_dir, save_dir, qid_to_ann):
    ##### Gather stats about baseline
    baseline_preds = pickle.load(open(
        os.path.join(save_dir, 'baseline_vqacp2', 'prediction_scores', 'test_qid_to_prediction_scores_epoch_28.pkl'),
        'rb'))
    print("Metrics for baseline")
    get_per_answer_type_metrics(qid_to_ann, baseline_preds)

    def load_preds(expt_name):
        return pickle.load(
            open(os.path.join(save_dir, expt_name, 'prediction_scores', '_qid_to_prediction_scores_epoch_7.pkl'),
                 'rb'))

    ###### Gather stats about ours
    our_expts = ['vqacp2_zero_out_full', 'vqacp2_zero_out_subset0.01']
    for expt in our_expts:
        print(f"Experiment: {expt}")
        qid_to_preds = load_preds(expt)
        get_per_answer_type_metrics(qid_to_ann, qid_to_preds)

    # ###### Gather stats about HINT
    hint_expts = ['HINT_hat_vqacp2', 'HINT_one_minus_hat_vqacp2', 'HINT_random_hat_vqacp2',
                  'HINT_variable_random_hat_vqacp2']

    for expt in hint_expts:
        print(f"Experiment: {expt}")
        get_per_answer_type_metrics(qid_to_ann, load_preds(expt))

    #
    ########### Gather stats about SCR
    def load_scr_preds(expt_name):
        return pickle.load(open(os.path.join(save_dir, expt_name, 'phase_3', 'prediction_scores',
                                             '_qid_to_prediction_scores_epoch_7.pkl'), 'rb'))

    scr_expts = ['scr_caption_based_hints_vqacp2', 'scr_one_minus_caption_based_hints_vqacp2',
                 'scr_random_caption_based_hints_vqacp2',
                 'scr_variable_random_caption_based_hints_vqacp2']
    for expt in scr_expts:
        print(f"Experiment: {expt}")
        get_per_answer_type_metrics(qid_to_ann, load_scr_preds(expt))


def get_qid_to_qn_and_ann(data_dir, split='test'):
    qns = json.load(open(os.path.join(data_dir, f'vqacp_v2_{split}_questions.json')))
    anns = json.load(open(os.path.join(data_dir, f'vqacp_v2_{split}_annotations.json')))

    qid_to_qn, qid_to_ann = {}, {}

    for qn in qns:
        qid_to_qn[qn['question_id']] = qn
    for ann in anns:
        qid_to_ann[ann['question_id']] = ann
    print(f"len(qid_to_qn) {len(qid_to_qn)}")
    print(f"len(qid_to_ann) {len(qid_to_ann)}")
    return qid_to_qn, qid_to_ann


def print_yes_no_counts(data_dir):
    train_qid_to_qn, train_qid_to_ann = get_qid_to_qn_and_ann(data_dir, split='train')
    test_qid_to_qn, test_qid_to_ann = get_qid_to_qn_and_ann(data_dir, split='test')

    def count(qid_to_ann):
        counts = {}
        for qid in qid_to_ann:
            ann = qid_to_ann[qid]
            if ann['answer_type'] == 'yes/no':
                for ans_holder in ann['answers']:
                    ans = ans_holder['answer']
                    if ans not in ['yes', 'no']:
                        continue
                    if ans not in counts:
                        counts[ans] = 0
                    counts[ans] += 1
        print(counts)

    print("Train yes/no:")
    count(train_qid_to_ann)
    print("Test yes/no:")
    count(test_qid_to_ann)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()
    analysis_dir = os.path.join(args.save_dir, 'analysis_by_answer_type')
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    # qid_to_qn, qid_to_ann = get_qid_to_qn_and_ann(args.data_dir, split='test')

    # compute_metrics(args.data_dir, args.save_dir, qid_to_ann)
    print_yes_no_counts(args.data_dir)
