import cv2, pickle, h5py, os, spacy, json
import numpy as np
import argparse
import pickle


def get_qid_to_random_hint(qids):
    qid_to_hint = {}
    for qid in qids:
        qid_to_hint[qid] = np.random.rand(36)
    return qid_to_hint


def main(args):
    train_qids = pickle.load(open(f'{args.data_dir}/hints/train_{args.load_hint_type}.pkl', 'rb'))
    val_qids = pickle.load(open(f'{args.data_dir}/hints/val_{args.load_hint_type}.pkl', 'rb'))

    train_qid_to_random_hint = get_qid_to_random_hint(train_qids)
    val_qid_to_random_hint = get_qid_to_random_hint(val_qids)
    pickle.dump(train_qid_to_random_hint, open(f'{args.data_dir}/hints/train_{args.save_hint_type}.pkl', 'wb'))
    pickle.dump(val_qid_to_random_hint, open(f'{args.data_dir}/hints/val_{args.save_hint_type}.pkl', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_hint_type')
    parser.add_argument('--save_hint_type')
    args = parser.parse_args()
    main(args)
