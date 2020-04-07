import nltk
from concurrent.futures import ProcessPoolExecutor
import os
import json
from tqdm import tqdm
import numpy as np
import sys

sys.setrecursionlimit(5000)

good_pos = ['JJ', 'JJS', 'NN', 'NNS', 'RB', 'RBR', 'VB', 'VBD', 'VBG']
# bad_words = ['many', 'much', 'color', 'man', 'people', 'picture', 'wearing', 'person', 'there', 'kind', 'woman',
#              'photo', 'doing', 'room', 'type', 'be', 'holding', 'animal', 'see', 'shirt', 'animals']


def filter_words(question_holder):
    qn = question_holder['question']
    tokens = nltk.word_tokenize(qn)
    tags = nltk.pos_tag(tokens)
    filtered_words = []
    for tag in tags:
        for gtp in good_pos:
            if tag[1] == gtp and tag[0] not in bad_words:
                filtered_words.append(tag)
    question_holder['filtered_words'] = filtered_words
    return question_holder


def demo():
    tokens = nltk.word_tokenize("How many people are wearing red shirts?")
    print(filter_words(tokens))


def create_batches(items, batch_size):
    items = np.asarray(items)
    return np.array_split(items, len(items) // batch_size)


def parse(questions):
    pool = ProcessPoolExecutor(max_workers=8)
    all_questions = []
    question_holder_batches = create_batches(questions, batch_size=10000)
    for question_holder_batch in tqdm(question_holder_batches):
        question_holder_batch = list(pool.map(filter_words, question_holder_batch))
        all_questions += question_holder_batch
    pool.shutdown()
    all_questions_map = {}
    for q in all_questions:
        all_questions_map[q['question_id']] = q
    return all_questions_map


def parse_and_save(save_dir, fname):
    questions = json.load(open(f'data/{fname}.json'))
    questions = parse(questions)
    json.dump(questions, open(os.path.join(save_dir, f'{fname}_with_filtered_words.json'), 'w'))


def get_words_to_qids(save_dir, split):
    questions = json.load(open(os.path.join(save_dir, f'vqacp_v2_{split}_questions_with_filtered_words.json')))
    words_to_qids = {}
    for qid in questions:
        q = questions[qid]
        for w in q['filtered_words']:
            w = w[0]
            if w not in words_to_qids:
                words_to_qids[w] = []
            words_to_qids[w].append(q['question_id'])
    json.dump(words_to_qids, open(os.path.join(save_dir, f'{split}_words_to_qids.json'), 'w'))


def sample(save_dir, ratio=0.05):
    train_questions = json.load(open(os.path.join(save_dir, 'vqacp_v2_train_questions_with_filtered_words.json')))
    selected_words = {}
    words_to_qids = json.load(open(os.path.join(save_dir, 'train_words_to_qids.json')))
    qids = list(train_questions.keys())
    selected_qids = {}

    def sample_recursively(qid):
        qid = str(qid)
        if qid not in selected_qids:
            selected_qids[qid] = qid

        print(f"len = {len(selected_qids)}")
        if len(selected_qids) > ratio * len(qids):
            return 'FULLY DONE'
        q = train_questions[qid]
        for w in q['filtered_words']:
            w = w[0]
            if w not in selected_words:
                selected_words[w] = 1
                print(f"words_to_qids for {w}: {len(words_to_qids[w])}")
                for _qid in words_to_qids[w]:
                    ret = sample_recursively(_qid)
                    if ret == 'FULLY DONE':
                        return 'FULLY DONE'
            else:
                return
        return 'NOT FULLY DONE'

    while True:
        rand_ix = np.random.randint(0, len(train_questions))
        qid = qids[rand_ix]
        ret = sample_recursively(qid)
        if ret == 'FULLY DONE':
            break

    json.dump(selected_words, open(os.path.join(save_dir, 'selected_words.json'), 'w'))
    json.dump(selected_qids, open(os.path.join(save_dir, 'selected_train_qids.json'), 'w'))


def sample_test(save_dir):
    selected_words = json.load(open(os.path.join(save_dir, 'selected_words.json')))
    words_to_test_qids = json.load(open(os.path.join(save_dir, 'test_words_to_qids.json')))
    filtered_test_qids = {}
    max_freq = 0
    max_w = 'none'
    w_freq = []
    for sw in selected_words:
        if sw not in words_to_test_qids:
            continue
        _test_qids = words_to_test_qids[sw]
        w_freq.append((sw, len(_test_qids)))
        for tqid in _test_qids:
            filtered_test_qids[tqid] = tqid
    print(f"Filtered test qids {len(filtered_test_qids)}")
    w_freq = sorted(w_freq, key=lambda x: x[1], reverse=True)
    print(w_freq)
    json.dump(filtered_test_qids, open(os.path.join(save_dir, 'selected_test_qids.json'), 'w'))


if __name__ == "__main__":
    save_dir = 'data/concepts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # print("Parsing concepts ...")
    # parse_and_save(save_dir, 'vqacp_v2_train_questions')
    # parse_and_save(save_dir, 'vqacp_v2_test_questions')
    #
    # print("Creating word to qid maps...")
    # get_words_to_qids(save_dir, 'train')
    # get_words_to_qids(save_dir, 'test')
    #
    # print("Sampling some words...")
    # sample(save_dir)

    print("Sampling test questions having those concepts")
    sample_test(save_dir)

    print("Done!")

# Parse all of the questions
# Create word to qids
# Select a word at random and consolidate all of the questions and their concepts recursively until we reach 5%
# Those question ids are our training set
# Search test questions having those words, those are our test set
# Test default model on both
# Test phase-2 model on both
# Test our model on both
