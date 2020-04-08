from __future__ import print_function

import json
import os
import pickle
import pickle as cPickle

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from tools.mixup import mixup_tensors
import time
import numpy as np
from tools.preprocess_answer import preprocess_answer
from dataset import RandomSubsetSampler
from torch.utils.data.dataloader import DataLoader


def set_lr(optimizer, frac):
    for group in optimizer.param_groups:
        group['lr'] *= frac


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def get_prediction_scores(logits, labels):
    logits = torch.max(logits, 1)[1]
    return labels.gather(1, logits.unsqueeze(1)).data


def compute_score_with_k_logits(logits, labels, k=5):
    logits = torch.sort(logits, 1)[1].data  # argmax
    scores = torch.zeros((labels.size(0), k))

    for i in range(k):
        one_hots = torch.zeros(*labels.size()).cuda()
        one_hots.scatter_(1, logits[:, -i - 1].view(-1, 1), 1)
        scores[:, i] = (one_hots * labels).squeeze().sum(1)
    scores = scores.max(1)[0]
    return scores


def compute_scr_loss(opt, objs, answers, ans_idxs, logits, hint_flags, hint_scores, ans_cossim):
    """Self-Critical Loss copied from https://github.com/jialinwu17/self_critical_vqa"""
    eps = 0.0000001
    sigmoid = nn.Sigmoid()

    bucket = opt.bucket
    vqa_grad = torch.autograd.grad((logits * (answers > 0).float()).sum(), objs, create_graph=True)[
        0]  # B x num_objs x 2048
    vqa_grad_cam = vqa_grad.sum(2)
    aidx = answers.argmax(1).detach().cpu().numpy().reshape((-1))

    loss_hint = torch.zeros(
        (vqa_grad_cam.size(0), opt.num_sub, 36)).cuda()  # B x 5 x 36 # num_sub = size of proposal object set
    hint_scores = hint_scores.squeeze()  # B x num_objs
    hint_sort, hint_ind = hint_scores.sort(1, descending=True)

    thresh = hint_sort[:, opt.num_sub:opt.num_sub + 1] - 0.00001
    thresh += ((thresh < 0.2).float() * 0.1)
    hint_scores = (hint_scores > thresh).float()

    for j in range(opt.num_sub):
        for k in range(36):
            if j == k:
                continue
            hint1 = hint_scores.gather(1, hint_ind[:, j:j + 1]).squeeze()  # j-th hint score
            hint2 = hint_scores.gather(1, hint_ind[:, k:k + 1]).squeeze()

            vqa1 = vqa_grad_cam.gather(1, hint_ind[:, j:j + 1]).squeeze()  # j-th grad
            vqa2 = vqa_grad_cam.gather(1, hint_ind[:, k:k + 1]).squeeze()
            if j < k:
                mask = ((hint1 - hint2) * (vqa1 - vqa2 - 0.0001) < 0).float()
                loss_hint[:, j, k] = torch.abs(vqa1 - vqa2 - 0.0001) * mask
            else:
                mask = ((hint2 - hint1) * (vqa2 - vqa1 - 0.0001) < 0).float()
                loss_hint[:, j, k] = torch.abs(vqa2 - vqa1 - 0.0001) * mask

    hint_flag1 = hint_flags.unsqueeze(1).unsqueeze(2).repeat(1, loss_hint.shape[1], loss_hint.shape[2]) \
        .detach_().cuda().float()
    loss_hint *= opt.scr_hint_loss_weight
    loss_hint *= hint_flag1
    loss_hint = loss_hint.sum(2)  # b num_sub
    loss_hint += (((loss_hint.sum(1).unsqueeze(1) > eps).float() * (loss_hint < eps).float()) * 10000)

    loss_hint, loss_hint_ind = loss_hint.min(1)  # loss_hint_ind b
    loss_hint_mask = (loss_hint > eps).float()
    loss_hint = (loss_hint * loss_hint_mask).sum() / (loss_hint_mask.sum() + eps)
    gt_logits = logits.gather(1, answers.argmax(1).view((-1, 1)))
    prob = sigmoid(gt_logits).view(-1)

    loss_compare = torch.zeros((logits.size(0), bucket)).cuda()
    loss_reg = torch.zeros((logits.size(0), bucket)).cuda()
    comp_mask = torch.zeros((logits.size(0), bucket)).cuda()
    for j in range(bucket):
        logits_pred = logits.gather(1, ans_idxs[:, j:j + 1])
        prob_pred = sigmoid(logits_pred).squeeze()
        vqa_grad_pred = torch.autograd.grad(logits.gather(1, ans_idxs[:, j:j + 1]).sum(), objs, create_graph=True)[
            0]
        vqa_grad_pred_cam = vqa_grad_pred.sum(2)  # b 36
        gradcam_diff = vqa_grad_pred_cam - vqa_grad_cam
        pred_aidx = ans_idxs[:, j].detach().cpu().numpy().reshape((-1))
        if opt.apply_answer_weight:
            ans_diff = torch.from_numpy(1 - ans_cossim[aidx, pred_aidx].reshape((-1))).cuda().float()
        prob_diff = prob_pred - prob
        prob_diff_relu = prob_diff * (prob_diff > 0).float()

        if opt.apply_answer_weight:
            loss_comp1 = prob_diff_relu.unsqueeze(1) * gradcam_diff * ans_diff.unsqueeze(1) * hint_scores
        else:
            loss_comp1 = prob_diff_relu.unsqueeze(1) * gradcam_diff * hint_scores
        loss_comp1 = loss_comp1.gather(1, loss_hint_ind.view(-1, 1)).squeeze()  # sum(1)
        loss_comp1 *= opt.scr_compare_loss_weight
        loss_compare[:, j] = loss_comp1
        comp_mask[:, j] = (prob_diff > 0).float().squeeze()

        if opt.apply_answer_weight:
            loss_reg[:, j] = (torch.abs(vqa_grad_pred_cam * ans_diff.unsqueeze(1) * (1 - hint_scores))).sum(1)
        else:
            loss_reg[:, j] = (torch.abs(vqa_grad_pred_cam * (1 - hint_scores))).sum(1)

    hint_flag2 = hint_flags.unsqueeze(1).repeat(1, loss_reg.shape[1]).detach_().cuda().float()
    loss_compare *= hint_flag2
    loss_reg *= hint_flag2
    loss_reg = loss_reg.mean() * opt.reg_loss_weight
    # loss_compare = loss_compare.mean()
    loss_compare = (loss_compare * comp_mask).sum() / (comp_mask.sum() + 0.0001)
    return loss_hint, loss_compare, loss_reg


def compute_hint_loss(opt, objs, gt_answers, logits, gt_hint_scores, hint_flags):
    """Implementation for the HINT paper (Selvaraju, Ramprasaath R., et al.)"""
    pred_hint_scores = torch.autograd.grad((logits * (gt_answers > 0).float()).sum(), objs, create_graph=True)[0]
    pred_hint_scores = pred_hint_scores.sum(2)  # B x num_objs

    # Subtract hint of every object from other objects
    gt_hint_scores, gt_hintscore_ixs = torch.sort(gt_hint_scores, 1, descending=True)
    gt_hint_scores = gt_hint_scores.squeeze()
    gt_hint_score_diff = gt_hint_scores.unsqueeze(2) - gt_hint_scores.unsqueeze(1)

    # Sort the predicted hint scores in the same order as GT hint scores
    pred_hint_scores_sorted_as_gt = pred_hint_scores.gather(1, gt_hintscore_ixs.squeeze())
    pred_hint_scores_sorted_as_gt_diff = pred_hint_scores_sorted_as_gt.unsqueeze(
        2) - pred_hint_scores_sorted_as_gt.unsqueeze(1)

    # Mask off the hint differences that are negative in GT, as we don't need to consider them for the loss
    # This should basically produce an upper triangular matrix
    gt_mask = torch.where(gt_hint_score_diff < 0, torch.zeros_like(gt_hint_score_diff),
                          torch.ones_like(gt_hint_score_diff))
    pred_hint_scores_sorted_as_gt_diff = pred_hint_scores_sorted_as_gt_diff * gt_mask

    # Mask off prediction hint differences which have negative signs
    # i.e., only keep the object pairs which do not match the order defined by GT
    pred_mask = torch.where(pred_hint_scores_sorted_as_gt_diff < 0,
                            -1 * torch.ones_like(pred_hint_scores_sorted_as_gt_diff),
                            torch.zeros_like(pred_hint_scores_sorted_as_gt_diff))
    pred_hint_scores_sorted_as_gt_diff = pred_hint_scores_sorted_as_gt_diff * pred_mask
    pred_hint_scores_sorted_as_gt_diff = pred_hint_scores_sorted_as_gt_diff * hint_flags.unsqueeze(1).unsqueeze(
        2).float().cuda()
    hint_loss = pred_hint_scores_sorted_as_gt_diff.sum(dim=1).mean()
    return opt.hint_loss_weight * hint_loss


def compute_non_tail_loss_for_objects(opt, objs, gt_answers, logits, use_absolute=False, limit_to_gt_answers=True):
    if limit_to_gt_answers:
        sensitivity_ans = torch.autograd.grad((logits * (gt_answers > 0).float()).sum(), objs, create_graph=True)[0]
    else:
        sensitivity_ans = torch.autograd.grad(logits.sum(), objs, create_graph=True)[0]

    sensitivity_ans = sensitivity_ans.sum(2)
    if use_absolute:
        sensitivity_ans = torch.abs(sensitivity_ans)

    # Get the top-K objects and bottom-k objects responsible for GT prediction
    # sensitivities_sorted, obj_ixs_all_ans_sorted = torch.sort(sensitivity_gt_ans, dim=1, descending=True)
    sensitivities_sorted, obj_ixs_gt_ans_sorted = torch.sort(sensitivity_ans, dim=1)
    num_bottom_objs = int(objs.shape[1]) - opt.num_most_sensitive_objects
    bottom_sensitivities = sensitivities_sorted[:, :num_bottom_objs]
    top_sensitivities = sensitivities_sorted[:, num_bottom_objs:]
    top_sensitivities_sum, bottom_sensitivities_sum = top_sensitivities.sum(1), bottom_sensitivities.sum(1)
    # diff = top_sensitivities_sum - bottom_sensitivities_sum
    diff = top_sensitivities_sum - bottom_sensitivities_sum

    diff += opt.non_tail_loss_margin_for_objects
    loss = torch.where(diff > 0, diff, torch.zeros(bottom_sensitivities.shape[0]).cuda())
    loss = opt.non_tail_loss_weight_for_objects * loss.mean()
    return loss, top_sensitivities_sum.sum(), bottom_sensitivities_sum.sum()


def compute_rolling_head_loss_for_objects(opt, objs, gt_answers, logits, use_absolute=False,
                                          dynamically_weight_rolling_head_loss=False):
    """Penalize the model if there is a 'head' object that is more sensitive towards GT answer(s) and
    top-K incorrect answers as compared to the summation of the remaining tail objects."""

    # Gather logits for gt answers
    gt_answer_logits = logits * (gt_answers > 0).float()

    # Gather logits for wrong answers
    wrong_ans_logits = logits * (gt_answers <= 0).float()
    wrong_ans_logits, _ = torch.sort(wrong_ans_logits, dim=1, descending=True)

    # Gather the object sensitivities towards predicting the GT and the wrong answers
    sensitivity_for_gt_and_wrong_ans = torch.autograd.grad(gt_answer_logits.sum()
                                                           + wrong_ans_logits[:, :opt.num_wrong_answers].sum(), objs,
                                                           create_graph=True)[0]
    # + wrong_ans_logits[:, :opt.num_wrong_answers].sum()
    # sensitivity_for_gt_and_wrong_ans = torch.autograd.grad(gt_answer_logits.sum(), objs,
    #                                                        create_graph=True)[0]
    sensitivity_for_gt_and_wrong_ans = sensitivity_for_gt_and_wrong_ans.sum(2)

    if use_absolute:
        sensitivity_for_gt_and_wrong_ans = torch.abs(sensitivity_for_gt_and_wrong_ans)

    # Starting from the object with lowest sensitivity, penalize the model if the sensitivity is larger than rest of the objects # Once a violating head object is found, then do not use rest of the head objects
    sensitivity_for_gt_and_wrong_ans, _ = torch.sort(sensitivity_for_gt_and_wrong_ans, dim=1)
    num_tail_objs = int(objs.shape[1]) - opt.num_most_sensitive_objects
    tail_sensitivities = sensitivity_for_gt_and_wrong_ans[:, :num_tail_objs]
    diff = None
    for head_ix in range(num_tail_objs, objs.shape[1]):
        current_head_sensitivities = sensitivity_for_gt_and_wrong_ans[:, head_ix].squeeze()
        curr_diff = current_head_sensitivities - tail_sensitivities.sum(1).squeeze() + \
                    opt.rolling_head_loss_margin_for_objects
        if diff is None:
            diff = curr_diff
        curr_diff = torch.where(curr_diff > 0, curr_diff, torch.zeros_like(curr_diff).cuda())
        diff = torch.where(diff > 0, diff, curr_diff)
    loss = opt.rolling_head_loss_weight_for_objects * diff.mean()
    return loss


def compute_dynamic_head_loss_for_objects(opt, objs, gt_answers, logits, use_absolute=False):
    sensitivity_gt_ans = torch.autograd.grad((logits * (gt_answers > 0).float()).sum(), objs, create_graph=True)[0]
    sensitivity_gt_ans = sensitivity_gt_ans.sum(2)
    if use_absolute:
        sensitivity_gt_ans = torch.abs(sensitivity_gt_ans)

    # Get the top-K objects and bottom-k objects responsible for GT prediction
    # sensitivities_sorted, obj_ixs_all_ans_sorted = torch.sort(sensitivity_gt_ans, dim=1, descending=True)
    sensitivities_sorted, obj_ixs_gt_ans_sorted = torch.sort(sensitivity_gt_ans, dim=1)
    num_bottom_objs = int(objs.shape[1]) - opt.num_most_sensitive_objects
    bottom_sensitivities = sensitivities_sorted[:, :num_bottom_objs]
    top_sensitivities = sensitivities_sorted[:, num_bottom_objs:]
    top_sensitivities_sum, bottom_sensitivities_sum = top_sensitivities.sum(1), bottom_sensitivities.sum(1)
    # diff = top_sensitivities_sum - bottom_sensitivities_sum
    diff = top_sensitivities_sum - bottom_sensitivities_sum

    diff += opt.non_tail_loss_margin_for_objects
    loss = torch.where(diff > 0, diff, torch.zeros(bottom_sensitivities.shape[0]).cuda())
    loss = opt.non_tail_loss_weight_for_objects * loss.mean()
    return loss, top_sensitivities_sum.sum(), bottom_sensitivities_sum.sum()


def compute_make_wrong_higher_than_gt_ans_loss(opt, objs, gt_answers, logits, use_absolute=False):
    """
    Tries to make the GT answer be more sensitive towards most sensitive objects for overall predictions.
    """

    # Compute the objects' sensitivities towards overall prediction
    num_bottom_objs = int(objs.shape[1]) - opt.num_most_sensitive_objects
    sensitivity_ans = torch.autograd.grad(logits.float().sum(), objs, create_graph=True)[0]
    sensitivity_ans = sensitivity_ans.sum(2)
    if use_absolute:
        sensitivity_ans = torch.abs(sensitivity_ans)

    sensitivity_ans_sorted, obj_ixs_all_ans_sorted = torch.sort(sensitivity_ans, dim=1)
    top_obj_ixs_for_all_ans = obj_ixs_all_ans_sorted[:, num_bottom_objs:]

    # Compute the objects' sensitivities towards GT answer prediction
    gt_ans_preds = logits * (gt_answers > 0).float()
    sensitivity_gt_ans = torch.autograd.grad(gt_ans_preds.sum(), objs, create_graph=True)[0]
    sensitivity_gt_ans = sensitivity_gt_ans.sum(2)
    gt_ans_wrt_most_sens_objs = sensitivity_gt_ans.gather(dim=1, index=top_obj_ixs_for_all_ans)

    # Compute the objects' sensitivities towards wrong answer prediction
    wrong_ans_preds = logits * (gt_answers <= 0).float()
    wrong_ans_preds, _ = torch.sort(wrong_ans_preds, dim=1, descending=True)
    sensitivity_wrong_ans = torch.autograd.grad(wrong_ans_preds[:, :opt.num_wrong_answers].sum(),
                                                objs, create_graph=True)[0]
    sensitivity_wrong_ans = sensitivity_wrong_ans.sum(2)
    wrong_ans_wrt_most_sens_objs = sensitivity_wrong_ans.gather(dim=1, index=top_obj_ixs_for_all_ans)

    # Compute the loss
    diff = opt.num_wrong_answers * gt_ans_wrt_most_sens_objs.sum(1) - wrong_ans_wrt_most_sens_objs.sum(1)
    loss = torch.where(diff > 0, diff, torch.zeros_like(diff)).cuda()
    loss = opt.make_wrong_higher_than_gt_ans_loss_weight * loss.mean()
    return loss


def compute_equal_gt_vs_wrong_loss_for_objects(opt, objs, gt_answers, logits, use_absolute=False):
    num_bottom_objs = int(objs.shape[1]) - opt.num_most_sensitive_objects

    sensitivity_gt_ans = torch.autograd.grad((logits * (gt_answers > 0).float()).sum(), objs, create_graph=True)[0]
    sensitivity_gt_ans = sensitivity_gt_ans.sum(2)
    if use_absolute:
        sensitivity_gt_ans = torch.abs(sensitivity_gt_ans)

    sensitivity_gt_ans_sorted, obj_ixs_gt_ans_sorted = torch.sort(sensitivity_gt_ans, dim=1)
    top_sensitivities_wrt_gt_ans = sensitivity_gt_ans_sorted[:, num_bottom_objs:]
    top_obj_ixs_for_gt = obj_ixs_gt_ans_sorted[:, num_bottom_objs:]

    wrong_ans_preds = logits * (gt_answers <= 0).float()
    wrong_ans_preds, _ = torch.sort(wrong_ans_preds, dim=1, descending=True)
    sensitivity_wrong_ans = torch.autograd.grad(wrong_ans_preds[:, :opt.num_wrong_answers].sum(),
                                                objs, create_graph=True)[0]
    sensitivity_wrong_ans = sensitivity_wrong_ans.sum(2)
    wrong_ans_wrt_most_sens_objs = sensitivity_wrong_ans.gather(dim=1, index=top_obj_ixs_for_gt)
    # Loss punishes the model if the objects most sensitive towards GT were less sensitive to wrong answers
    diff = top_sensitivities_wrt_gt_ans.sum(1) - wrong_ans_wrt_most_sens_objs.sum(1)
    loss = torch.where(diff > 0, diff, torch.zeros_like(diff)).cuda()
    loss = opt.equal_gt_vs_wrong_loss_weight_for_objects * loss.mean()
    return loss


def compute_non_head_answers_loss(opt, objs, gt_answers, logits):
    """Penalize the model if the top objects responsible for GT answers are also sensitive to incorrect answers
    (except for the top-K incorrect answers). The main intuition is that we want the objects to be responsible for
    distinguishing between confusing answers, but not have an impact on determining other responses. For instance,
    if it is a 'what color' question, we want the objects to be used to distinguish between red, pink vs green etc.,
    but not have any effect on predicting non-sensible answers such as 0, yes, etc."""
    num_bottom_objs = int(objs.shape[1]) - opt.num_most_sensitive_objects

    sensitivity_gt_ans = torch.autograd.grad((logits * (gt_answers > 0).float()).sum(), objs, create_graph=True)[0]
    sensitivity_gt_ans = sensitivity_gt_ans.sum(2)
    sensitivity_gt_ans_sorted, obj_ixs_gt_ans_sorted = torch.sort(sensitivity_gt_ans, dim=1)
    top_obj_ixs_for_gt = obj_ixs_gt_ans_sorted[:, num_bottom_objs:]

    wrong_ans_preds = logits * (gt_answers <= 0).float()
    wrong_ans_preds, _ = torch.sort(wrong_ans_preds, dim=1, descending=True)
    sensitivity_non_head_wrong_ans = torch.autograd.grad(wrong_ans_preds[:, opt.num_non_head_wrong_answers:].sum(),
                                                         objs, create_graph=True)[0]
    sensitivity_non_head_wrong_ans = sensitivity_non_head_wrong_ans.sum(2)
    sensitivity_wrong_ans_wrt_head_objs = sensitivity_non_head_wrong_ans.gather(dim=1, index=top_obj_ixs_for_gt)

    # Punish the wrong answers for being sensitive with head objects (head w.r.t. GT answer)
    loss = opt.non_head_answers_loss_weight * torch.abs(sensitivity_wrong_ans_wrt_head_objs.mean(dim=1)).mean()
    return loss


def compute_fixed_gt_ans_loss(opt, logits, answers, fixed_gt_ans_perturbation=0):
    if len(opt.fixed_ans_scores) == 2:
        fixed_gt1 = torch.ones_like(logits) * opt.fixed_ans_scores[0]
        fixed_gt2 = torch.ones_like(logits) * opt.fixed_ans_scores[1]
        fixed_gt = torch.where(torch.rand(fixed_gt1.shape).cuda() > 0.5, fixed_gt1, fixed_gt2)
    else:
        fixed_gt = torch.ones_like(logits) * opt.fixed_ans_scores[0]

    if fixed_gt_ans_perturbation > 0:
        fixed_gt += torch.rand(fixed_gt.shape).cuda() * fixed_gt_ans_perturbation

    if opt.fixed_gt_ans_loss_function == 'mse':
        loss = F.mse_loss(F.sigmoid(logits), fixed_gt)
    elif opt.fixed_gt_ans_loss_function == 'l1':
        loss = F.l1_loss(F.sigmoid(logits), fixed_gt)
    else:
        loss = nn.functional.binary_cross_entropy_with_logits(logits, fixed_gt)
    loss *= fixed_gt.size(1)

    return opt.fixed_gt_ans_loss_weight * loss


def compute_random_gt_ans_loss(opt, logits):
    random_gt = torch.rand(logits.shape).cuda()
    loss = instance_bce_with_logits(logits, random_gt)
    return opt.random_gt_ans_loss_weight * loss


def eval_extra(model, epoch, log_file,
               train_loader_all, train_loader_for_regularization, train_loader_except_regularization,
               eval_loader_for_regularization, eval_loader_except_regularization, opt):
    if train_loader_all is not None:
        evaluate_and_log("Training set", model, train_loader_all, opt, epoch, log_file)

    if train_loader_for_regularization is not None:
        evaluate_and_log("Training set used for reg", model, train_loader_for_regularization, opt, epoch,
                         log_file)
    if train_loader_except_regularization is not None:
        evaluate_and_log("Training set not used for reg", model, train_loader_except_regularization, opt,
                         epoch,
                         log_file)
    if eval_loader_for_regularization is not None:
        evaluate_and_log("Eval set used for reg", model, eval_loader_for_regularization, opt, epoch,
                         log_file)
    if eval_loader_except_regularization is not None:
        evaluate_and_log("Eval set not used for reg", model, eval_loader_except_regularization, opt, epoch,
                         log_file)


def create_optim(opt, model):
    if opt.optimizer == 'adadelta':
        optim = torch.optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'RMSprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08,
                                    weight_decay=opt.weight_decay, momentum=0, centered=False)
    elif opt.optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=opt.weight_decay)
    return optim


def compute_loss(opt, train_loader, epoch, iter_num, objs, answers, logits, ans_idxs, hint_flags, hint_scores,
                 ans_cossim):
    loss_vqa = opt.vqa_loss_weight * instance_bce_with_logits(logits, answers)  # floating point number

    msg = f"iter {iter_num} / {len(train_loader)} (epoch {epoch}) vqa = %.4f " % (loss_vqa.item())

    loss = loss_vqa
    if opt.use_scr_loss:
        loss_scr_hint, loss_scr_compare, loss_scr_reg = compute_scr_loss(opt, objs, answers, ans_idxs, logits,
                                                                         hint_flags,
                                                                         hint_scores,
                                                                         ans_cossim)
        loss += loss_scr_hint + loss_scr_compare + loss_scr_reg
        msg += " , scr_hint = %.3f, scr_compare = %.3f, scr_reg = %.3f" % (
            loss_scr_hint.item(), loss_scr_compare.item(),
            loss_scr_reg.item())

    if opt.use_hint_loss:
        loss_hint = compute_hint_loss(opt, objs, answers, logits, hint_scores, hint_flags)
        loss += loss_hint
        msg += " , hint = %.3f " % (loss_hint.item())

    if opt.use_non_tail_loss_for_objects:
        limit_to_gt_answers = opt.answers_for_non_tail_loss == 'gt'
        non_tail_loss_for_objects, _, _ = compute_non_tail_loss_for_objects(opt, objs, answers, logits,
                                                                            use_absolute=opt.use_absolute_for_non_tail_loss,
                                                                            limit_to_gt_answers=limit_to_gt_answers)
        if opt.auto_reweight_nte_loss:
            reweight = float(non_tail_loss_for_objects / loss_vqa)
            non_tail_loss_for_objects = non_tail_loss_for_objects / reweight

        msg += ", non_tail_loss_for_objects = %.4f" % (non_tail_loss_for_objects.item())

        loss += non_tail_loss_for_objects

    if opt.use_make_wrong_higher_than_gt_ans_loss:
        make_wrong_higher_than_gt_ans_loss = compute_make_wrong_higher_than_gt_ans_loss(opt, objs, answers, logits)
        msg += ', make_wrong_higher_than_gt_ans_loss = %.4f' % (make_wrong_higher_than_gt_ans_loss.item())
        loss += make_wrong_higher_than_gt_ans_loss

    if opt.use_rolling_head_loss_for_objects:
        rolling_head_loss_for_objects = compute_rolling_head_loss_for_objects(opt, objs, answers, logits,
                                                                              use_absolute=opt.use_absolute_for_rolling_head_loss,
                                                                              dynamically_weight_rolling_head_loss=opt.dynamically_weight_rolling_head_loss)
        msg += ", rolling_head_loss_for_objects = %.4f" % (rolling_head_loss_for_objects.item())

        loss += rolling_head_loss_for_objects

    if opt.use_equal_gt_vs_wrong_loss_for_objects:
        gt_vs_wrong_loss_for_objects = compute_equal_gt_vs_wrong_loss_for_objects(opt, objs, answers, logits,
                                                                                  use_absolute=opt.use_absolute_for_equal_gt_vs_wrong_loss)
        msg += " gt_vs_wrong_loss_for_objects = %.4f" % gt_vs_wrong_loss_for_objects
        loss += gt_vs_wrong_loss_for_objects

    if opt.use_non_head_answers_loss:
        non_head_answers_loss = compute_non_head_answers_loss(opt, objs, answers, logits)
        msg += " non_head_answers_loss = %.4f" % non_head_answers_loss
        loss += non_head_answers_loss

    if opt.use_fixed_gt_ans_loss:
        fixed_gt_ans_loss = compute_fixed_gt_ans_loss(opt, logits,
                                                      answers=answers,
                                                      fixed_gt_ans_perturbation=opt.fixed_gt_ans_perturbation)
        msg += " fixed_gt_ans_loss = %.4f " % fixed_gt_ans_loss
        loss += fixed_gt_ans_loss

    if opt.use_random_gt_ans_loss:
        random_gt_ans_loss = compute_random_gt_ans_loss(opt, logits)
        msg += " random_gt_ans_loss = %.4f " % random_gt_ans_loss
        loss += random_gt_ans_loss

    print(msg)
    return loss


def run(model,
        train_loader,
        eval_loader_all,
        opt,
        train_loader_all,
        train_loader_for_regularization=None,
        train_loader_except_regularization=None,
        eval_loader_for_regularization=None,
        eval_loader_except_regularization=None):
    """Contains the main training loop and test logic.
    Also, handles saving/loading of checkpoints
    """

    def _eval_extra():
        eval_extra(model, epoch, log_file,
                   train_loader_all, train_loader_for_regularization, train_loader_except_regularization,
                   eval_loader_for_regularization, eval_loader_except_regularization, opt)

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    # Preliminary setup of optimizer, lr scheduler, logs
    optim = create_optim(opt, model)
    lr_scheduler = ExponentialLR(optim, gamma=opt.lr_gamma)
    best_eval_score = 0
    ans_cossim = pickle.load(open('ans_cossim.pkl', 'rb'))
    log_file = open(opt.checkpoint_path + '/log.txt', 'a')
    print(json.dumps(vars(opt), indent=4, sort_keys=True), file=log_file)
    log_file.flush()

    ## If load_checkpoint_path flag is specified, then we need to load model from that state
    if opt.load_checkpoint_path is not None and len(opt.load_checkpoint_path) > 0:
        ckpt = torch.load(os.path.join(opt.load_checkpoint_path))
        if 'epoch' in ckpt:
            states_ = ckpt['model_state_dict']
        else:
            states_ = ckpt

        model.load_state_dict(states_)

    ## Handle test logic
    if opt.test:
        print("Evaluating ...")
        if 'epoch' in ckpt:
            epoch = ckpt['epoch']
        else:
            epoch = 'unknown'
        evaluate_and_log("Eval on Test", model, eval_loader_all, opt, epoch, log_file,
                         save_sensitivities=True,
                         save_prediction_scores=True,
                         prefix='test')
        return

    if opt.test_on_train:
        print("Evaluating ...")
        if 'epoch' in ckpt:
            epoch = ckpt['epoch']
        else:
            epoch = 'unknown'
        evaluate_and_log("Eval on Train", model, train_loader_all, opt, epoch, log_file,
                         save_sensitivities=False,
                         save_prediction_scores=True,
                         prefix='train')
        return

    ## The main training loop
    for epoch in range(opt.max_epochs):
        print(f"Training epoch {epoch}...")
        iter_num = 0
        if opt.var_random_subset_ratio is not None:
            ## Gather a new training subset every epoch. The subset is selected randomly.
            train_dset = train_loader.dataset
            subset_sampler = RandomSubsetSampler(torch.LongTensor(range(0, len(train_dset))),
                                                 int(len(train_dset) * opt.var_random_subset_ratio))
            train_loader = DataLoader(train_dset,
                                      opt.batch_size,
                                      shuffle=False,
                                      num_workers=opt.num_workers,
                                      sampler=subset_sampler)

        for objs, qns, answers, hint_scores, _, question_ids, image_ids, hint_flags in iter(train_loader):
            if opt.change_scores_every_epoch:
                ## Assign random scores every epoch, if the flag says to do so.
                hint_scores = torch.rand(hint_scores.shape).cuda()

            objs = objs.cuda().float().requires_grad_()  # B x num_objs x emb
            qns = qns.cuda().long()  # B x len
            answers = answers.cuda()  # B x num classes
            hint_scores = hint_scores.cuda().float()  # B x num_objs x 1
            words, logits, attended_objs, ans_idxs = model(qns, objs)  # pred: B x num classes
            loss = compute_loss(opt, train_loader, epoch, iter_num, objs, answers, logits, ans_idxs, hint_flags,
                                hint_scores,
                                ans_cossim)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip)
            optim.step()
            optim.zero_grad()
            log_file.flush()
            iter_num += 1

        print("##\n")
        lr_scheduler.step()
        print(f"lr {lr_scheduler.get_lr()}")

        if epoch in opt.log_epochs:
            eval_score = evaluate_and_log("Eval", model, eval_loader_all, opt, epoch, log_file,
                                          save_sensitivities=True,
                                          save_prediction_scores=True)
        else:
            eval_score = evaluate_and_log("Eval", model, eval_loader_all, opt, epoch, log_file,
                                          save_sensitivities=False,
                                          save_prediction_scores=True)
        log_file.flush()
        # Save model
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'lr_state_dict': lr_scheduler.state_dict()
        }
        if epoch in opt.log_epochs:
            torch.save(state, os.path.join(opt.checkpoint_path, f'model-epoch-{epoch}.pth'))
        if eval_score > best_eval_score:
            torch.save(state, os.path.join(opt.checkpoint_path, 'model-best.pth'))
            best_eval_score = eval_score
        if epoch in opt.log_epochs:
            _eval_extra()

    print("Evaluating extra stuff on the best model ...")
    best_model_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    _eval_extra()


def predict(model, dataloader, opt):
    dataroot = 'data'
    label2ans_path = os.path.join(dataroot, 'processed', 'trainval_label2ans.pkl')
    label2ans = cPickle.load(open(label2ans_path, 'rb'))
    results = []
    for objs, qns, gt_answers, hintscore, _, qid, image_id, hint_flag in tqdm(iter(dataloader)):
        for _a, _qid in zip(gt_answers, qid):
            _a = int(torch.argmax(_a))
            _qid = int(_qid)
            results.append({
                "question_id": _qid,
                "answer": label2ans[_a]
            })
    json.dump(results, open(os.path.join(opt.predict_checkpoint, 'scr.json'), 'w'))


def compute_gt_ans_sensitivities(objs, gt_answers, logits):
    sensitivities = torch.autograd.grad((logits * (gt_answers > 0).float()).sum(), objs, create_graph=True)[0]
    sensitivities = sensitivities.sum(2)
    return sensitivities


def compute_all_ans_sensitivities(objs, logits):
    sensitivities = torch.autograd.grad(logits.sum(), objs, create_graph=True)[0]
    sensitivities = sensitivities.sum(2)
    return sensitivities


def evaluate(model, dataloader, opt, epoch=None, save_sensitivities=False, save_prediction_scores=False,
             save_logits=False, prefix=''):
    model.eval()
    score = 0
    scorek = 0
    V_loss = 0

    upper_bound = 0
    num_data = 0
    qid_to_logits = {}
    qid_to_prediction_scores = {}  # 0 if prediction is incorrect and the GT softscore if it is correct
    qid_to_human_agreement = {}
    qid_to_gt_ans_sensitivities = {}
    qid_to_all_ans_sensitivities = {}

    if save_prediction_scores:
        annotations = dataloader.dataset.get_annotations()
        qid_to_annotations = {ann['question_id']: ann for ann in annotations}
        label2ans = dataloader.dataset.label2ans

    for objs, qns, answers, hint_scores, _, question_ids, image_ids, hint_flags in tqdm(iter(dataloader)):
        objs = objs.cuda().float().requires_grad_()
        qns = qns.cuda().long()
        answers = answers.cuda()  # true labels
        _, logits, attended_objs, ansidx = model(qns, objs)
        loss = instance_bce_with_logits(logits, answers)
        V_loss += loss.item() * objs.size(0)
        batch_score = compute_score_with_logits(logits, answers.data).sum()
        batch_scorek = compute_score_with_k_logits(logits, answers.data).sum()
        score += batch_score
        scorek += batch_scorek

        if save_prediction_scores:
            prediction_scores = get_prediction_scores(logits, answers.data)
            answer_ids = torch.argmax(logits, dim=1).detach().cpu().numpy()

        upper_bound += (answers.max(1)[0]).sum()
        num_data += logits.size(0)

        if save_sensitivities:
            gt_ans_sensitivities = compute_gt_ans_sensitivities(objs, answers, logits)
            all_ans_sensitivities = compute_all_ans_sensitivities(objs, logits)
            for qid, gt_sens, all_sens in zip(question_ids, gt_ans_sensitivities, all_ans_sensitivities):
                qid_to_gt_ans_sensitivities[int(qid)] = gt_sens.detach().cpu().numpy().tolist()
                qid_to_all_ans_sensitivities[int(qid)] = all_sens.detach().cpu().numpy().tolist()

        if save_prediction_scores:
            for qid, _prediction_scores in zip(question_ids, prediction_scores):
                qid_to_prediction_scores[int(qid)] = float(_prediction_scores)

        if save_prediction_scores:
            for qid, pred_ans_id in zip(question_ids, answer_ids):
                pred_ans = label2ans[pred_ans_id]
                _agreement = []
                for gt_ans_holder in qid_to_annotations[int(qid)]['answers']:
                    gt_ans = preprocess_answer(gt_ans_holder['answer'])
                    if pred_ans == gt_ans:
                        _agreement.append(1)
                    else:
                        _agreement.append(0)

                qid_to_human_agreement[int(qid)] = _agreement

        if save_logits:
            for qid, qid_logits in zip(question_ids, logits):
                qid_to_logits[int(qid)] = qid_logits.detach().cpu().numpy().tolist()

    if save_sensitivities:
        if not os.path.exists(os.path.join(opt.checkpoint_path, 'sensitivities')):
            os.makedirs(os.path.join(opt.checkpoint_path, 'sensitivities'))
        cPickle.dump(qid_to_gt_ans_sensitivities,
                     open(os.path.join(opt.checkpoint_path, 'sensitivities',
                                       f'{prefix}_qid_to_gt_ans_sensitivities_epoch_{epoch}.pkl'), 'wb'))
        print(f'Saved {prefix}_qid_to_gt_ans_sensitivities_epoch_{epoch}.pkl')
        cPickle.dump(qid_to_all_ans_sensitivities,
                     open(os.path.join(opt.checkpoint_path, 'sensitivities',
                                       f'{prefix}_qid_to_all_ans_sensitivities_epoch_{epoch}.pkl'), 'wb'))
        print(f'Saved {prefix}_qid_to_all_ans_sensitivities_epoch_{epoch}.pkl')

    if save_prediction_scores:
        if not os.path.exists(os.path.join(opt.checkpoint_path, 'prediction_scores')):
            os.makedirs(os.path.join(opt.checkpoint_path, 'prediction_scores'))
        cPickle.dump(qid_to_prediction_scores, open(
            os.path.join(opt.checkpoint_path, 'prediction_scores',
                         f'{prefix}_qid_to_prediction_scores_epoch_{epoch}.pkl'),
            'wb'))
        cPickle.dump(qid_to_prediction_scores, open(
            os.path.join(opt.checkpoint_path, 'prediction_scores', f'{prefix}_qid_to_prediction_scores.pkl'), 'wb'))
        cPickle.dump(qid_to_human_agreement,
                     open(
                         os.path.join(opt.checkpoint_path, 'prediction_scores', f'{prefix}_qid_to_human_agreement.pkl'),
                         'wb'))
        print(f'Saved {prefix}_qid_to_prediction_scores_epoch_{epoch}.pkl')

    if save_logits:
        st = time.time()
        if not os.path.exists(os.path.join(opt.checkpoint_path, 'qid_to_logits')):
            os.makedirs(os.path.join(opt.checkpoint_path, 'qid_to_logits'))
            cPickle.dump(qid_to_logits,
                         open(os.path.join(opt.checkpoint_path, 'qid_to_logits',
                                           f'{prefix}_qid_to_logits_epoch_{epoch}.pkl'),
                              'wb'))
        print(f"Saved logits in {time.time() - st} secs")

    score = score / len(dataloader.dataset)
    scorek = scorek / len(dataloader.dataset)
    V_loss /= len(dataloader.dataset)

    upper_bound = upper_bound / len(dataloader.dataset)
    model.train()
    return score, upper_bound, V_loss, scorek


def evaluate_and_log(key, model, dataloader, opt, epoch, log_file, save_sensitivities=False,
                     save_prediction_scores=False, save_logits=False, prefix=""):
    print(f"Evaluating {key} ...")
    score, _, loss, scorek = evaluate(model, dataloader, opt=opt, epoch=epoch,
                                      save_sensitivities=save_sensitivities,
                                      save_prediction_scores=save_prediction_scores,
                                      save_logits=save_logits,
                                      prefix=prefix)
    print(f"{key} (epoch {epoch}), score = %.3f, score_k = %.3f" % (score, scorek))
    print(f"{key} (epoch {epoch}), score = %.3f, score_k = %.3f" % (score, scorek),
          file=log_file)
    return score
