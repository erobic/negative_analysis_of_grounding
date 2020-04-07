import cv2, pickle, h5py, os, spacy, json
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pickle as cPickle

names = ['train', 'val']

image_id_to_question_ids = {}
train_anns = json.load(open('/hdd/robik/VQACP2/self_critical_features/vqacp_v2_train_annotations.json'))
test_anns = json.load(open('/hdd/robik/VQACP2/self_critical_features/vqacp_v2_test_annotations.json'))

for ann in train_anns:
    image_id = ann['image_id']
    if image_id not in image_id_to_question_ids:
        image_id_to_question_ids[image_id] = []
    qid = ann['question_id']
    image_id_to_question_ids[image_id].append(qid)

for name in names:
    qid2hint = {}
    img_id2idx = pickle.load(open(os.path.join('%s36_imgid2img.pkl' % name), 'rb'))
    img_idx2id = {int(v): k for (k, v) in zip(img_id2idx.keys(), img_id2idx.values())}
    h5_path = os.path.join('%s36.hdf5' % name)
    hf = h5py.File(h5_path, 'r')
    features = hf.get('image_features')
    spatials = hf.get('spatial_features')

    for idx in range(len(img_idx2id)):
        img_id = img_idx2id[idx]

        hint_score = np.zeros((36))
        hint_score_attr = np.zeros((36))

        # Assign random scores
        hint_score = np.random.rand(36)
        if img_id not in image_id_to_question_ids:
            print(f'image {img_id} not found')
            continue
        for qid in image_id_to_question_ids[img_id]:
            qid2hint[qid] = hint_score
    print(f"Dumping to {name + '_vqx_random_hint.pkl'}")
    pickle.dump(qid2hint, open(name + '_vqx_random_hint_for_all.pkl', 'wb'))
