import os
import pickle

if __name__ == "__main__":
    root = '/hdd/robik/projects/self_critical_vqa'
    hat = pickle.load(open(os.path.join(root, 'data', 'hints', 'train_hat.pkl'), 'rb'))
    one_minus_hat = pickle.load(open(os.path.join(root, 'data', 'hints', 'train_one_minus_hat.pkl'), 'rb'))
    random_hat = pickle.load(open(os.path.join(root, 'data', 'hints', 'train_random_hat.pkl'), 'rb'))

    qid = list(hat.keys())[0]
    print(f"hat {hat[qid]}")
    print(f"one_minus_hat {one_minus_hat[qid]}")
    print(f"random_hat {random_hat[qid]}")
