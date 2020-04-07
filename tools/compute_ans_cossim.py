import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def read_glove(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    idx2emb = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        # vals = map(float, vals[1:])
        vals = [float(v) for v in vals[1:]]
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            print(word)
            continue
        idx2emb[idx] = word2emb[word]
    return idx2emb, word2emb


def compute_ans_cossim(idx2emb):
    cossim = cosine_similarity(idx2emb, idx2emb)
    return cossim


if __name__ == "__main__":
    root = '/hdd/robik/projects/self_critical_vqa'
    label2ans = pickle.load(open(os.path.join(root, 'data', 'processed', 'trainval_label2ans.pkl'), 'rb'))
    glove_file = os.path.join(root, 'data/glove/glove.6B.300d.txt')
    idx2emb, word2emb = read_glove(label2ans, glove_file)
    ans_cossim = compute_ans_cossim(idx2emb)
    pickle.dump(ans_cossim, open(os.path.join(root, 'ans_cossim.pkl'), 'wb'))
