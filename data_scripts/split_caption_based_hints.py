import pickle
import os
import numpy as np

np.random.seed(1)


def main(hint_type, num_splits, save_dir='data/hints'):
    hints = pickle.load(open(f'data/hints/train_{hint_type}.pkl', 'rb'))
    qids = list(hints.keys())
    np.random.shuffle(qids)
    start_ix = 0
    split_len = len(hints) // num_splits
    end_ix = split_len

    for split_num in range(num_splits):
        split_qids = qids[start_ix:end_ix]
        split = {qid: hints[qid] for qid in split_qids}
        pickle.dump(split, open(f'data/hints/train_{hint_type}_{split_num + 1}.pkl', 'wb'))
        start_ix = end_ix
        end_ix = end_ix + split_len


if __name__ == "__main__":
    hint_type = 'caption_based_hints'
    num_splits = 2
    main(hint_type, num_splits)
