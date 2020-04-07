import os
import json
import numpy as np

np.random.seed(1)


def load_qid_to_qns(split):
    qns = json.load(open(f'data/vqacp_v2_{split}_questions.json'))
    qid_to_qn = {}
    for qn in qns:
        qid_to_qn[qn['question_id']] = qn
    return qid_to_qn


def load_qid_to_anns(split):
    anns = json.load(open(f'data/vqacp_v2_{split}_annotations.json'))
    qid_to_ann = {}
    for ann in anns:
        qid_to_ann[ann['question_id']] = ann
    return qid_to_ann


def load_qtype_ans_to_qids(split):
    anns = json.load(open(f'data/vqacp_v2_{split}_annotations.json'))
    qtype_ans_to_qids = {}

    for ann in anns:
        qtype = ann['question_type']
        curr_answers = {}
        for ans_holder in ann['answers']:
            answer = ans_holder['answer']
            if answer in curr_answers:
                continue
            curr_answers[answer] = 1
            qtype_ans = qtype + '_' + answer

            if qtype_ans not in qtype_ans_to_qids:
                qtype_ans_to_qids[qtype_ans] = []
            qtype_ans_to_qids[qtype_ans].append(ann['question_id'])
    return qtype_ans_to_qids


def choose_some_qtype_ans_pairs(qtype_ans_to_qids, max_ratio=0.05):
    num_filtered = 0
    num_total = np.sum([len(qtype_ans_to_qids[qtype_ans]) for qtype_ans in qtype_ans_to_qids])
    qtype_ans_keys = list(qtype_ans_to_qids.keys())
    filtered_qtype_ans_to_qids = {}
    while True:
        if num_filtered > max_ratio * num_total:
            break
        rand_ix = np.random.randint(0, len(qtype_ans_keys))
        qtype_ans = qtype_ans_keys[rand_ix]
        if len(qtype_ans_to_qids[qtype_ans]) < 5:
            continue
        if qtype_ans not in filtered_qtype_ans_to_qids:
            filtered_qtype_ans_to_qids[qtype_ans] = qtype_ans_to_qids[qtype_ans]
            num_filtered += len(qtype_ans_to_qids[qtype_ans])
    return filtered_qtype_ans_to_qids


def choose_qtype_ans(qtype_ans_filter, qtype_ans_to_qids):
    filtered = {}
    for qtype_ans in qtype_ans_filter:
        if qtype_ans in qtype_ans_to_qids:
            filtered[qtype_ans] = qtype_ans_to_qids[qtype_ans]
    return filtered


def len_of_filtered(filtered):
    return np.sum([len(filtered[key]) for key in filtered])


def main():
    # train_qid_to_qns = load_qid_to_qns('train')
    # train_qid_to_anns = load_qid_to_anns('train')
    train_qtype_ans_to_qids = load_qtype_ans_to_qids('train')
    train_filtered_qtype_ans_to_qids = choose_some_qtype_ans_pairs(train_qtype_ans_to_qids)
    print(
        f"Selected {len(train_filtered_qtype_ans_to_qids)} training qtype-ans combinations with "
        f"{len_of_filtered(train_filtered_qtype_ans_to_qids)} instances")
    json.dump(train_filtered_qtype_ans_to_qids, open('data/hints/train_filtered_qtype_ans.json', 'w'))
    test_filtered_qtype_ans_to_qids = choose_qtype_ans(train_filtered_qtype_ans_to_qids, load_qtype_ans_to_qids('test'))
    print(
        f"Selected {len(test_filtered_qtype_ans_to_qids)} test qtype-ans combinations with "
        f"{len_of_filtered(test_filtered_qtype_ans_to_qids)} instances")
    json.dump(test_filtered_qtype_ans_to_qids, open('data/hints/test_filtered_qtype_ans.json', 'w'))


if __name__ == "__main__":
    main()

# Selected 2088 training qtype-ans combinations with 63137 instances
# Selected 1769 test qtype-ans combinations with 50128 instances