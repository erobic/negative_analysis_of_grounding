import os
import json
import h5py
import pickle
import numpy as np

# What color are the towels? COCO_train2014_000000559449.jpg
# Is there something to cut the vegetables with? COCO_train2014_000000460833.jpg
#  What is the color of the man's clothes? COCO_train2014_000000273898.jpg

def format_image_id(img_id, split):
    return f'COCO_{split}2014_' + str(img_id).rjust(12, '0') + '.jpg'


# Search for questions with hint and 'color' in it
def filter_qns(root):
    train_img_id_to_ix = pickle.load(open(os.path.join(root, 'data', 'train36_imgid2img.pkl'), 'rb'))
    # val_img_id_to_ix = pickle.load(open(os.path.join(root, 'data', 'val36_imgid2img.pkl'), 'rb'))
    train_h5 = h5py.File(os.path.join(root, 'data', 'train36.hdf5'), 'r')
    # val_h5 = os.path.join(root, 'data', 'train36.hdf5')

    qid_to_hat = pickle.load(open(os.path.join(root, 'data', 'hints', 'train_hat.pkl'), 'rb'))
    qns = json.load(open(os.path.join(root, 'data', 'vqacp_v2_train_questions.json')))
    qid_to_qns = {
        q['question_id']: q for q in qns
    }
    for qid in qid_to_hat:
        if qid in qid_to_qns:
            qn = qid_to_qns[qid]
            #COCO_train2014_000000283912.jpg, COCO_train2014_000000060982.jpg,COCO_train2014_000000340226.jpg
            #COCO_train2014_000000474846.jpg

            if 'color' in qn['question']: #and 'man' in qn['question'] and 'many' not in qn['question']:
                if qn['image_id'] in train_img_id_to_ix:
                    qn['image_name'] = format_image_id(qn['image_id'], 'train')
                    if qn['image_id'] not in [60982]:
                        continue
                    image_ix = int(train_img_id_to_ix[qn['image_id']])
                    bbox = train_h5['spatial_features'][image_ix]
                    hints = qid_to_hat[qid]
                    qn['image_ix'] = image_ix
                    # qn['bbox'] = bbox.tolist()
                    # qn['hints'] = hints.tolist()
                    top5 = np.argsort(hints, )[31:]
                    top5_bboxes = bbox[top5]
                    qn['top5bboxes'] = top5_bboxes.tolist()


                    print(json.dumps(qn, indent=4))


if __name__ == "__main__":
    root = '/hdd/robik/projects/self_critical_vqa'
    filter_qns(root)
