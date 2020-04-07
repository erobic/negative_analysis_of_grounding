import pickle
import os
import numpy as np

np.random.seed(1)


def main(hint_type, split):
    hints = pickle.load(open(f'data/hints/{split}_{hint_type}.pkl', 'rb'))

    one_minus_hints = {}
    for qid in hints:
        one_minus_hints[qid] = 1 - hints[qid]
    pickle.dump(one_minus_hints, open(f'data/hints/{split}_one_minus_{hint_type}.pkl', 'wb'))


if __name__ == "__main__":
    hint_type = 'hat'
    main(hint_type, 'train')
    main(hint_type, 'val')
