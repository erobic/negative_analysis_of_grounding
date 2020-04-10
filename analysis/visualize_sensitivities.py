import os
import json
import h5py
import pickle
import numpy as np
import argparse
from PIL import Image, ImageDraw


# What color are the towels? COCO_train2014_000000559449.jpg
# Is there something to cut the vegetables with? COCO_train2014_000000460833.jpg
#  What is the color of the man's clothes? COCO_train2014_000000273898.jpg

def format_image_id(img_id, split):
    return f'COCO_{split}2014_' + str(img_id).rjust(12, '0') + '.jpg'


def gather_gt_hints(data_dir, hint_type):
    """
    Collect GT hint scores for FasterRCNN object computed using equation 1 of HINT paper
    :param data_dir:
    :param hint_type:
    :return:
    """
    train_hints = pickle.load(open(os.path.join(data_dir, 'hints', f'train_{hint_type}.pkl'), 'rb'))
    val_hints = pickle.load(open(os.path.join(data_dir, 'hints', f'val_{hint_type}.pkl'), 'rb'))
    all_hints = train_hints
    for qid in val_hints:
        all_hints[qid] = val_hints[qid]
    return all_hints


def gather_experiment_sensitivities(save_dir, expt_name, sensitivity_file):
    return pickle.load(open(os.path.join(save_dir, expt_name, 'sensitivities', sensitivity_file)))


def filter(hat_gt_sensitivities, text_gt_sensitivities, expt_sensitivities, filter, limit=2):
    def is_good(curr_expt_qid_sens, gt_sens, top=1):
        sorted_sens = np.argsort(curr_expt_qid_sens)
        sorted_gt_sens = np.argsort(gt_sens)

        ret = True
        for t in range(len(curr_expt_qid_sens) - 1, len(curr_expt_qid_sens) - top - 1, -1):
            if sorted_sens[t] != sorted_gt_sens[t]:
                ret = False

        return ret

    filtered_entries = {}
    expts = list(expt_sensitivities.keys())
    qids = list(expt_sensitivities[expts[0]].keys())
    for qid in qids:
        curr_entry = {}
        should_add = True
        if qid not in hat_gt_sensitivities or qid not in text_gt_sensitivities:
            continue

        for expt in expts:
            curr_expt_sens = expt_sensitivities[expt][qid]
            curr_entry[expt] = curr_expt_sens
            if filter[expt] is None:
                continue
            curr_hat_gt_sens = hat_gt_sensitivities[qid]
            curr_text_gt_sens = text_gt_sensitivities[qid]
            if filter[expt] == 'good':
                if not is_good(curr_expt_sens, curr_hat_gt_sens) and not is_good(curr_expt_sens, curr_text_gt_sens):
                    should_add = False

            if filter[expt] == 'bad':
                if is_good(curr_expt_sens, curr_hat_gt_sens) or is_good(curr_expt_sens, curr_text_gt_sens):
                    should_add = False

        if should_add:
            filtered_entries[qid] = curr_entry
        if len(filtered_entries) >= limit:
            return filtered_entries
    return filtered_entries


def is_correctly_grounded(gt_hints, pred_sensitivities, topK=3):
    # If any object in
    gt_obj_ixs = np.argsort(gt_hints)
    pred_obj_ixs = np.argsort(np.asarray(pred_sensitivities))
    topK_ixs = -1 * np.arange(1, topK + 1)
    if pred_obj_ixs[-1] in gt_obj_ixs[topK_ixs]:
        return True
    else:
        return False


def filter_by_sensitivity_and_answer(qid_to_gt_ans, qid_to_gt_hints, qid_to_pred_sens, qid_to_pred_ans):
    correct_ans_and_grounding, total = 0, 0
    correct_ans_incorrect_grounding = 0
    correct_ans = 0
    for qid in qid_to_gt_ans:
        if int(qid) not in qid_to_gt_hints or int(qid) not in qid_to_pred_ans:
            continue
        gt_ans = qid_to_gt_ans[qid]
        gt_hints = qid_to_gt_hints[int(qid)]
        pred_ans = qid_to_pred_ans[int(qid)]
        pred_sens = qid_to_pred_sens[int(qid)]

        # If it was correctly predicted, and if grounding was also correct then increase count
        if pred_ans > 0:
            if is_correctly_grounded(gt_hints, pred_sens):
                correct_ans_and_grounding += 1
            else:
                correct_ans_incorrect_grounding += 1
            correct_ans += 1
        total += 1

    print(
        f"Correct ans+grounding: {correct_ans_and_grounding} "
        f"Correct ans but incorrect grounding: {correct_ans_incorrect_grounding}  "
        f"Correct ans: {correct_ans} "
        f"Total: {total}\n")


def gather_data_for_visualization(data_dir, save_dir, qid_to_gt_ans, qid_to_hat_hints, qid_to_textual_hints):
    data = {}

    ##### Gather stats about baseline
    baseline_sens = pickle.load(open(
        os.path.join(save_dir, 'baseline_vqacp2', 'sensitivities', 'test_qid_to_gt_ans_sensitivities_epoch_28.pkl'),
        'rb'))
    baseline_preds = pickle.load(open(
        os.path.join(save_dir, 'baseline_vqacp2', 'prediction_scores', 'test_qid_to_prediction_scores_epoch_28.pkl'),
        'rb'))
    print("baseline wrt HAT hints")
    filter_by_sensitivity_and_answer(qid_to_gt_ans, qid_to_hat_hints, baseline_sens, baseline_preds)

    print("baseline wrt textual hints")
    filter_by_sensitivity_and_answer(qid_to_gt_ans, qid_to_textual_hints, baseline_sens, baseline_preds)

    def load_sensitivities(expt_name):
        return pickle.load(
            open(os.path.join(save_dir, expt_name, 'sensitivities', '_qid_to_gt_ans_sensitivities_epoch_7.pkl'),
                 'rb'))

    def load_preds(expt_name):
        return pickle.load(
            open(os.path.join(save_dir, expt_name, 'prediction_scores', '_qid_to_prediction_scores_epoch_7.pkl'),
                 'rb'))

    ###### Gather stats about ours
    our_expts = ['vqacp2_zero_out_full', 'vqacp2_zero_out_subset0.01']
    for expt in our_expts:
        print(f"Experiment: {expt} vs HAT")
        filter_by_sensitivity_and_answer(qid_to_gt_ans, qid_to_hat_hints, load_sensitivities(expt),
                                         load_preds(expt))

        print(f"Experiment: {expt} vs txt")
        filter_by_sensitivity_and_answer(qid_to_gt_ans, qid_to_textual_hints, load_sensitivities(expt),
                                         load_preds(expt))

    ###### Gather stats about HINT
    hint_expts = ['HINT_hat_vqacp2', 'HINT_one_minus_hat_vqacp2', 'HINT_random_hat_vqacp2',
                  'HINT_variable_random_hat_vqacp2']

    for expt in hint_expts:
        print(f"Experiment: {expt}")
        filter_by_sensitivity_and_answer(qid_to_gt_ans, qid_to_hat_hints, load_sensitivities(expt),
                                         load_preds(expt))

    ########### Gather stats about SCR
    def load_scr_sensitivities(expt_name):
        return pickle.load(
            open(os.path.join(save_dir, expt_name, 'phase_3', 'sensitivities',
                              '_qid_to_gt_ans_sensitivities_epoch_7.pkl'),
                 'rb'))

    def load_scr_preds(expt_name):
        return pickle.load(open(os.path.join(save_dir, expt_name, 'phase_3', 'prediction_scores',
                                             '_qid_to_prediction_scores_epoch_7.pkl'), 'rb'))

    scr_expts = ['scr_caption_based_hints_vqacp2', 'scr_one_minus_caption_based_hints_vqacp2',
                 'scr_random_caption_based_hints_vqacp2',
                 'scr_variable_random_caption_based_hints_vqacp2']
    for expt in scr_expts:
        print(f"Experiment: {expt}")
        filter_by_sensitivity_and_answer(qid_to_gt_ans, qid_to_textual_hints, load_scr_sensitivities(expt),
                                         load_scr_preds(expt))

    return data


def filter_for_visualization(save_dir, qid_to_hat_hints):
    def load_sensitivities(expt_name):
        return pickle.load(
            open(os.path.join(save_dir, expt_name, 'sensitivities', '_qid_to_gt_ans_sensitivities_epoch_7.pkl'),
                 'rb'))

    def load_preds(expt_name):
        return pickle.load(
            open(os.path.join(save_dir, expt_name, 'prediction_scores', '_qid_to_prediction_scores_epoch_7.pkl'),
                 'rb'))

    ###### Gather stats about HINT
    hint_expts = {'HINT_hat_vqacp2': {'correct_ans': True, 'correct_grounding': True},
                  'HINT_one_minus_hat_vqacp2': {'correct_ans': True, 'correct_grounding': False},
                  'HINT_random_hat_vqacp2': {'correct_ans': True, 'correct_grounding': False}
                  }
    for expt_name in hint_expts:
        hint_expts[expt_name]['sensitivities'] = load_sensitivities(expt_name)
        hint_expts[expt_name]['preds'] = load_preds(expt_name)

    filtered_qids = []
    for qid in qid_to_hat_hints:
        add = False
        for expt_name in hint_expts:
            expt_details = hint_expts[expt_name]
            if qid not in expt_details['preds']:
                break
            if expt_details['preds'][qid] > 0:
                add = hint_expts[expt_name]['correct_grounding'] == is_correctly_grounded(qid_to_hat_hints[qid],
                                                                                          expt_details['sensitivities'][
                                                                                              qid])
        if add:
            filtered_qids.append(qid)
    return filtered_qids, hint_expts

    ########### Gather stats about SCR
    def load_scr_sensitivities(expt_name):
        return pickle.load(
            open(os.path.join(save_dir, expt_name, 'phase_3', 'sensitivities',
                              '_qid_to_gt_ans_sensitivities_epoch_7.pkl'),
                 'rb'))

    def load_scr_preds(expt_name):
        return pickle.load(open(os.path.join(save_dir, expt_name, 'phase_3', 'prediction_scores',
                                             '_qid_to_prediction_scores_epoch_7.pkl'), 'rb'))

    scr_expts = ['scr_caption_based_hints_vqacp2', 'scr_one_minus_caption_based_hints_vqacp2',
                 'scr_random_caption_based_hints_vqacp2',
                 'scr_variable_random_caption_based_hints_vqacp2']
    for expt_name in scr_expts:
        print(f"Experiment: {expt_name}")
        filter_by_sensitivity_and_answer(qid_to_gt_ans, qid_to_textual_hints, load_scr_sensitivities(expt_name),
                                         load_scr_preds(expt_name))

    return data


# def load_qid_to_qa(data_dir):
#     train_qns = json.load(open(os.path.join(data_dir, 'vqacp_v2_train_questions.json')))
#     val_qns = json.load(open(os.path.join(data_dir, 'vqacp_v2_test_questions.json')))
#     qid_to_qns = {
#         q['question_id']: q for q in train_qns
#     }


def visualize(data_dir, save_dir):
    train_img_id_to_ix = pickle.load(open(os.path.join(data_dir, 'train36_imgid2img.pkl'), 'rb'))
    val_img_id_to_ix = pickle.load(open(os.path.join(data_dir, 'val36_imgid2img.pkl'), 'rb'))
    train_h5 = h5py.File(os.path.join(data_dir, 'train36.hdf5'), 'r')
    val_h5 = h5py.File(os.path.join(data_dir, 'val36.hdf5'), 'r')

    viz_data = gather_data_for_visualization(data_dir, save_dir)
    train_qns = json.load(open(os.path.join(data_dir, 'vqacp_v2_train_questions.json')))
    val_qns = json.load(open(os.path.join(data_dir, 'vqacp_v2_test_questions.json')))


def create_qid_to_qn_and_ann(data_dir, analysis_dir, filter_qids):
    # train_qns = json.load(open(os.path.join(data_dir, 'vqacp_v2_train_questions.json')))
    test_qns = json.load(open(os.path.join(data_dir, 'vqacp_v2_test_questions.json')))
    # train_anns = json.load(open(os.path.join(data_dir, 'vqacp_v2_train_annotations.json')))
    test_anns = json.load(open(os.path.join(data_dir, 'vqacp_v2_test_annotations.json')))

    qid_to_qn, qid_to_ann = {}, {}

    # for qn in train_qns:
    #     if qn['question_id'] in filter_qids:
    #         qid_to_qn[qn['question_id']] = qn
    for qn in test_qns:
        if qn['question_id'] in filter_qids:
            qid_to_qn[qn['question_id']] = qn
    # for ann in train_anns:
    #     if ann['question_id'] in filter_qids:
    #         qid_to_ann[ann['question_id']] = ann
    for ann in test_anns:
        if ann['question_id'] in filter_qids:
            qid_to_ann[ann['question_id']] = ann
    print(f"len(qid_to_qn) {len(qid_to_qn)}")
    print(f"len(qid_to_ann) {len(qid_to_ann)}")
    json.dump(qid_to_qn, open(os.path.join(analysis_dir, 'qid_to_qn.json'), 'w'))
    json.dump(qid_to_ann, open(os.path.join(analysis_dir, 'qid_to_ann.json'), 'w'))


def create_qid_to_image_details(data_dir, qid_to_ann, analysis_dir, filter_qids):
    # image_id, bounding boxes, image name
    train_img_id_to_ix = pickle.load(open(os.path.join(data_dir, 'train36_imgid2img.pkl'), 'rb'))
    val_img_id_to_ix = pickle.load(open(os.path.join(data_dir, 'val36_imgid2img.pkl'), 'rb'))
    train_h5 = h5py.File(os.path.join(data_dir, 'train36.hdf5'), 'r')
    val_h5 = h5py.File(os.path.join(data_dir, 'val36.hdf5'), 'r')

    qid_to_image_details = {}
    for qid in filter_qids:
        qid = str(qid)
        if qid not in qid_to_ann:
            continue
        ann = qid_to_ann[qid]
        image_id = ann['image_id']
        if image_id in train_img_id_to_ix:
            image_ix = train_img_id_to_ix[image_id]
            boxes = train_h5['spatial_features'][int(image_ix)]
        elif image_id in val_img_id_to_ix:
            image_ix = val_img_id_to_ix[image_id]
            boxes = val_h5['spatial_features'][int(image_ix)]
        else:
            continue
        image_name = str(image_id).rjust(12, '0') + '.jpg'
        qid_to_image_details[qid] = {
            'image_name': image_name,
            'image_ix': image_ix,
            'boxes': boxes.tolist()
        }
    print(f"len(qid_to_image_details) {len(qid_to_image_details)}")
    json.dump(qid_to_image_details, open(os.path.join(analysis_dir, 'qid_to_image_details.json'), 'w'))


def qid_to_gt_sensitivity():
    pass


def create_qid_to_gt_ans(data_dir, analysis_dir, filter_qids):
    train_target = pickle.load(open(os.path.join(data_dir, 'processed', 'train_target.pkl'), 'rb'))
    val_target = pickle.load(open(os.path.join(data_dir, 'processed', f'val_target.pkl'), 'rb'))
    target = train_target + val_target
    qid_to_gt_ans = {}
    for t in target:
        question_id = t['question_id']
        if question_id in filter_qids:
            assert question_id not in qid_to_gt_ans
            qid_to_gt_ans[question_id] = t
    print(f"len(qid_to_gt_ans) {len(qid_to_gt_ans)} ")
    json.dump(qid_to_gt_ans, open(os.path.join(analysis_dir, 'qid_to_gt_ans.json'), 'w'))


def qid_to_predictions():
    pass


def visualize_one_image(data_dir, image_name, boxes, sensitivities, viz_fname, save_orig=False):
    if os.path.exists(os.path.join(data_dir, 'COCO', 'train2017', image_name)):
        dir = 'train2017'
    else:
        dir = 'val2017'
    img = Image.open(os.path.join(data_dir, 'COCO', dir, image_name)).convert('RGB')
    if save_orig:
        img.save(viz_fname + '_orig.png', 'png')

    alpha = 0
    alpha_step_size = 0.025  # reduced by 0.25 every time
    alpha_max_limit = 0.2
    sorted_box_ids = np.argsort(-np.asarray(sensitivities))[:len(sensitivities)]  # sort in desc order
    # new_img = Image.new('RGB', (np_img.shape[0], np_img.shape[1]))

    draw = ImageDraw.Draw(img, 'RGBA')
    cnt = 0
    border_width = 9
    border_width_step_size = 5
    min_border_width = 0
    if save_orig:
        print(f'{image_name}: box: {boxes[0]}')
    # for box_id in range(len(sorted_box_ids) - 1, -1, -1):
    for box_id in sorted_box_ids:
        box = boxes[box_id]
        (x1, y1, x2, y2) = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # draw.rectangle(((x1, y1), (x2, y2)), fill=(0, 0, 0, int(alpha * 255)))
        if border_width > 0:
            draw.rectangle(((x1, y1), (x2, y2)), width=border_width, outline=(255, 0, 0))
        # region = img.crop((x1, y1, x2, y2))
        # region.putalpha(int(255 * alpha))
        # new_img.paste(region, (x1, y1, x2, y2))
        if cnt > 3:
            alpha += alpha_step_size
        border_width -= border_width_step_size
        cnt += 1
        if alpha > alpha_max_limit:
            alpha = alpha_max_limit
        if border_width <= min_border_width:
            border_width = min_border_width

    img.save(viz_fname + '.png', 'png')


def visualize_many(data_dir, filtered_qids, qid_to_image_details, qid_to_gt_hint, qid_to_qn, qid_to_ann, expt_details,
                   viz_dir,
                   limit=50):
    expt0 = list(expt_details.keys())[0]
    cnt = 0
    for qid in filtered_qids:
        cnt += 1
        qid_viz_dir = os.path.join(viz_dir, f'{qid}')
        if not os.path.exists(qid_viz_dir):
            os.makedirs(qid_viz_dir)
        img_details = qid_to_image_details[str(qid)]
        gt_hint = qid_to_gt_hint[qid]
        visualize_one_image(data_dir, img_details['image_name'], img_details['boxes'], gt_hint,
                            os.path.join(qid_viz_dir, 'gt'), save_orig=True)
        json.dump(qid_to_qn[str(qid)], open(os.path.join(qid_viz_dir, 'qn.json'), 'w'))
        json.dump(qid_to_ann[str(qid)], open(os.path.join(qid_viz_dir, 'ann.json'), 'w'))

        for expt in expt_details:
            pred_hint = expt_details[expt]['sensitivities'][qid]
            visualize_one_image(data_dir, img_details['image_name'], img_details['boxes'], pred_hint,
                                os.path.join(qid_viz_dir, expt))

        if cnt >= limit:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()
    analysis_dir = os.path.join(args.save_dir, 'analysis')
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    qid_to_hat_hints = gather_gt_hints(args.data_dir, 'hat')
    qid_to_textual_hints = gather_gt_hints(args.data_dir, 'caption_based_hints')
    filter_qids = {qid: qid for qid in
                   list(set(list(qid_to_hat_hints.keys()) + list(qid_to_textual_hints.keys())))}

    # create_qid_to_qn_and_ann(args.data_dir, analysis_dir, filter_qids)
    qid_to_qn = json.load(open(os.path.join(analysis_dir, 'qid_to_qn.json')))
    qid_to_ann = json.load(open(os.path.join(analysis_dir, 'qid_to_ann.json')))

    # create_qid_to_image_details(args.data_dir, qid_to_ann, analysis_dir, filter_qids)
    qid_to_image_details = json.load(open(os.path.join(analysis_dir, 'qid_to_image_details.json')))

    # create_qid_to_gt_ans(args.data_dir, analysis_dir, filter_qids)
    qid_to_gt_ans = json.load(open(os.path.join(analysis_dir, 'qid_to_gt_ans.json')))

    # gather_data_for_visualization(args.data_dir, args.save_dir, qid_to_gt_ans, qid_to_hat_hints, qid_to_textual_hints)
    filtered_qids, expt_details = filter_for_visualization(args.save_dir, qid_to_hat_hints)
    viz_dir = os.path.join(analysis_dir, 'viz')
    visualize_many(args.data_dir, filtered_qids, qid_to_image_details, qid_to_hat_hints, qid_to_qn, qid_to_ann,
                   expt_details,
                   viz_dir)
