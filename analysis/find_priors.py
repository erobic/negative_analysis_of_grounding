import os
import json


def map_by_qid(data):
    qid_to_item = {}
    for d in data:
        qid_to_item[d['question_id']] = d
    return qid_to_item


def filter_qns_and_count_by_ans(qns, anns, filters):
    cnt_by_ans = {}
    qid_to_qn = map_by_qid(qns)
    qid_to_ann = map_by_qid(anns)
    for qid in qid_to_ann:
        qn = qid_to_qn[qid]['question']
        add = True
        for filter in filters:
            if filter not in qn:
                add = False
        if add:
            answers = qid_to_ann[qid]['answers']
            for ans_holder in answers:
                ans = ans_holder['answer']
                if ans not in cnt_by_ans:
                    cnt_by_ans[ans] = 0
                cnt_by_ans[ans] += 1

    ans_cnts = []

    for ans in cnt_by_ans:
        if cnt_by_ans[ans] > 100:
            ans_cnts.append({'answer': ans, 'cnt': cnt_by_ans[ans]})
    print(json.dumps(sorted(ans_cnts, key=lambda x: x['cnt'], reverse=True), indent=4))
    # print(f"cnt_by_ans {cnt_by_ans}")


if __name__ == "__main__":
    qfilters = ["color", "couch"]

    train_qns = json.load(open('data/vqacp_v2_train_questions.json'))
    train_anns = json.load(open('data/vqacp_v2_train_annotations.json'))

    test_qns = json.load(open('data/vqacp_v2_test_questions.json'))
    test_anns = json.load(open('data/vqacp_v2_test_annotations.json'))

    print("#### Training")
    filter_qns_and_count_by_ans(train_qns, train_anns, qfilters)

    print("#### Testing")
    filter_qns_and_count_by_ans(test_qns, test_anns, qfilters)

