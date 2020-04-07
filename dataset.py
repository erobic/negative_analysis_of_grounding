import os
import json
import pickle as cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle
from torch.utils.data.sampler import Sampler


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-',
                                                                                             ' ').replace('.',
                                                                                                          '').replace(
            '"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                if '-' in w:
                    print(w)
                tokens.append(self.add_word(w))
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(len(self.word2idx))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry


class SelfCriticalDataset(Dataset):

    def __init__(self, split,
                 hint_type,
                 dictionary,
                 opt,
                 load_img=True,
                 discard_items_without_hints=False,
                 discard_items_with_hints=False,
                 ignore_counting_questions=False):
        super(SelfCriticalDataset, self).__init__()
        self.split = split
        self.hint_type = hint_type
        self.dictionary = dictionary  # questions' dictionary
        self.load_img = load_img
        self.opt = opt
        self.data_dir = opt.data_dir
        self.discard_items_without_hints = discard_items_without_hints
        self.discard_items_with_hints = discard_items_with_hints
        if hint_type is None and self.discard_items_without_hints:
            raise Exception("Cannot discard items without hints because hint_type is not specified")
        if hint_type is None and self.discard_items_with_hints:
            raise Exception("Cannot discard items with hints because hint_type is not specified")
        self.ignore_counting_questions = ignore_counting_questions

        # Load data
        self.qid_to_target = self.get_qid_to_target()
        ans2label_path = os.path.join(self.data_dir, 'processed', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(self.data_dir, 'processed', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.image_id2ix = {}
        self.datalen = self.get_datalen()
        print(f"split {self.split} len {self.datalen}")

    def get_qid_to_target(self):
        train_target = cPickle.load(open(os.path.join(self.data_dir, 'processed', f'train_target.pkl'), 'rb'))
        val_target = cPickle.load(open(os.path.join(self.data_dir, 'processed', f'val_target.pkl'), 'rb'))
        target = train_target + val_target
        qid_to_target = {}
        for t in target:
            question_id = t['question_id']
            assert question_id not in qid_to_target
            qid_to_target[question_id] = t
        return qid_to_target

    def init_data(self):
        self.hf = {}
        self.features = {}
        self.spatials = {}
        self.cls_scores = {}
        self.attr_scores = {}

        print('loading features from h5 file')
        self.load_data('train')
        self.load_data('val')
        count = self.init_vqx()
        print(f"{self.split} count {count}")
        self.tokenize()
        self.tensorize()
        self.v_dim = 2048  # self.features.size(2)
        self.s_dim = 36  # self.spatials.size(2)

    def load_data(self, split):
        self.image_id2ix[split] = cPickle.load(open(os.path.join(self.data_dir, f'{split}36_imgid2img.pkl'), 'rb'))
        h5_path = os.path.join(self.data_dir, '%s36.hdf5' % split)
        self.hf[split] = h5py.File(h5_path, 'r')
        self.features[split] = self.hf[split].get('image_features')
        self.spatials[split] = self.hf[split].get('spatial_features')
        self.cls_scores[split] = np.array(self.hf[split].get('cls_score'))
        self.attr_scores[split] = np.array(self.hf[split].get('attr_score'))

    def get_hint_fname(self):

        if self.hint_type is None or len(self.hint_type) == 0:
            return None
        else:
            hint_fname = f'hints/{self.split}_{self.hint_type}.pkl'
            print(f"loading hints from hint_fname")
            return hint_fname

    def get_questions(self):
        if self.opt.dataset == 'vqacp2':
            f = os.path.join(self.data_dir, f'vqacp_v2_{self.split}_questions.json')
            return json.load(open(f))
        elif self.opt.dataset == 'vqa2':
            f = os.path.join(self.data_dir, f'v2_OpenEnded_mscoco_{self.split}2014_questions.json')
            return json.load(open(f))['questions']

    def get_annotations(self):
        if self.opt.dataset == 'vqacp2':
            f = os.path.join(self.data_dir, f'vqacp_v2_{self.split}_annotations.json')
            return json.load(open(f))
        elif self.opt.dataset == 'vqa2':
            f = os.path.join(self.data_dir, f'v2_mscoco_{self.split}2014_annotations.json')
            return json.load(open(f))['annotations']

    def get_datalen(self):
        hint_fname = self.get_hint_fname()
        count = 0
        questions = self.get_questions()

        if hint_fname is not None:
            self.hint = cPickle.load(open(os.path.join(self.data_dir, hint_fname), 'rb'))

        for i in tqdm(range(len(questions))):
            question = questions[i]
            question_id = question['question_id']

            if self.ignore_counting_questions:
                if ('how many' in question['question']) or ('how much' in question['question']):
                    continue

            #  if hint type is not specified, or if the dataset is being asked to return entire dataset, then
            if self.discard_items_without_hints and question_id not in self.hint.keys():
                continue
            elif self.discard_items_with_hints and question_id in self.hint.keys():
                continue
            count += 1

        return count

    def init_vqx(self):
        hint_fname = self.get_hint_fname()
        count = 0
        # self.entriess = cPickle.load(open(self.dataroot + '/VQAcp_caption_' + self.split + 'dataset.pkl', 'rb'))
        self.questions = self.get_questions()
        self.entries = {}
        print(f"split {self.split} questions {len(self.questions)}")

        if hint_fname is not None:
            self.hint = cPickle.load(open(os.path.join(self.data_dir, hint_fname), 'rb'))
        for i in tqdm(range(len(self.questions))):
            question = self.questions[i]
            image_id = question['image_id']
            if image_id in self.image_id2ix['train']:
                split = 'train'
            else:
                split = 'val'
            question_id = question['question_id']

            if self.ignore_counting_questions:
                if ('how many' in question['question']) or ('how much' in question['question']):
                    continue

            if self.discard_items_without_hints and question_id not in self.hint.keys():
                continue
            elif self.discard_items_with_hints and question_id in self.hint.keys():
                continue
            elif self.hint_type is not None and question_id in self.hint.keys():
                hint = self.hint[question_id]
                hint_a = np.zeros((36))
                obj_cls = np.array(self.cls_scores[split][self.image_id2ix[split][image_id]][:, 0])
                hint_o = obj_cls.astype('float')
                hint_flag = 1
            else:
                hint_a = np.zeros((36))
                hint_o = np.zeros((36))
                hint = np.zeros((36))
                hint_flag = 0

            new_entry = {'image': self.image_id2ix[split][image_id],
                         'image_id': image_id,
                         'question_id': question_id,
                         'question': question['question'],
                         'answer': self.qid_to_target[question_id],
                         'hint': hint,
                         'hint_a': hint_a,
                         'hint_o': hint_o,
                         'hint_flag': hint_flag}
            self.entries[count] = new_entry
            count += 1
        print(f"split {self.split} init_vqx count {count}")
        return count

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for e_id in range(len(self.entries)):
            entry = self.entries[e_id]
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for e_id in range(len(self.entries)):
            entry = self.entries[e_id]
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if labels is None:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
            elif len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        # features
        if not hasattr(self, 'hf'):
            self.init_data()
        entry = self.entries[index]
        imgid = entry['image_id']
        if imgid in self.image_id2ix['train']:
            split = 'train'
        else:
            split = 'val'

        qid = entry['question_id']
        if self.load_img:
            obj_nodes = torch.from_numpy(np.array(self.features[split][entry['image']]))
        else:
            obj_nodes = torch.zeros(36, 2048)
        hint_score = torch.from_numpy(entry['hint'])
        hint_flag = entry['hint_flag']
        if self.load_img:
            hint_o = torch.from_numpy(entry['hint_o'])
        else:
            hint_o = torch.zeros_like(hint_score)

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)

        hint_score = hint_score.float().unsqueeze(1)

        if labels is not None:
            target.scatter_(0, labels, scores)

        return obj_nodes, question, target, hint_score, hint_o, qid, imgid, hint_flag

    def __len__(self):
        return self.datalen


class RandomSubsetSampler(Sampler):
    def __init__(self, data_source, subset_size):
        print(f"Creating RandomSubsetSampler of subset size {subset_size} and total size {len(data_source)}")
        self.data_source = data_source
        self.subset_size = subset_size
        self.subset = torch.randperm(len(self.data_source))[:self.subset_size]

    def __iter__(self):
        return iter(self.subset.tolist())

    def __len__(self):
        return self.subset_size
