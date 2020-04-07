import os
import pickle
import json


def count_overlaps(hints, qids):
    count = 0
    for h_qid in hints:
        if h_qid in qids:
            count += 1
    return count


if __name__ == "__main__":
    root = '/hdd/robik/projects/self_critical_vqa'
    hint_type = 'caption_based_hints'
    train_hints = pickle.load(open(os.path.join(root, 'data', 'hints', f'train_{hint_type}.pkl'), 'rb'))
    test_hints = pickle.load(open(os.path.join(root, 'data', 'hints', f'val_{hint_type}.pkl'), 'rb'))
    all_hint_qids = train_hints
    for qid in test_hints:
        all_hint_qids[qid] = test_hints[qid]

    train_qns = json.load(open(os.path.join(root, 'data', 'vqacp_v2_train_questions.json')))
    test_qns = json.load(open(os.path.join(root, 'data', 'vqacp_v2_test_questions.json')))
    train_qids = {q['question_id']: q for q in train_qns}
    test_qids = {q['question_id']: q for q in test_qns}

    train_overlap = count_overlaps(all_hint_qids, train_qids)
    test_overlap = count_overlaps(all_hint_qids, test_qids)

    print(f"Train qids {len(train_qids)} qids with hints {train_overlap}")
    print(f"Test qids {len(test_qids)} qids with hints {test_overlap}")

# HAT:
# Train qids 438183 qids with hints 40185 - 9%
# Test qids 219928 qids with hints 19272 - 8.8%

# Textual explanation:
# Train qids 438183 qids with hints 20417 - 4.6%
# Test qids 219928 qids with hints 12469 - 5.6% 
