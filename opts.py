import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--data_dir')
    parser.add_argument('--dataset', type=str, default='vqacp2', choices=['vqa2', 'vqacp2'])
    parser.add_argument('--do_not_discard_items_without_hints', action='store_true')
    parser.add_argument('--split', type=str, default='v2cp_train',
                        help='training split')
    parser.add_argument('--hint_type', type=str)
    parser.add_argument('--fixed_random_subset_ratio', type=float, default=None)
    parser.add_argument('--var_random_subset_ratio', type=float, default=None)
    parser.add_argument('--split_test', type=str, default='v2cp_test',
                        help='test split')

    parser.add_argument('--rnn_size', type=int, default=1280,
                        help='size of the rnn in number of hidden nodes in question gru')
    parser.add_argument('--num_hid', type=int, default=1280,
                        help='size of the rnn in number of hidden nodes in question gru')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GCN layers')
    parser.add_argument('--rnn_type', type=str, default='gru',
                        help='rnn, gru, or lstm')
    parser.add_argument('--v_dim', type=int, default=2048,
                        help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--logit_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--activation', type=str, default='ReLU',
                        help='number of layers in the RNN')
    parser.add_argument('--norm', type=str, default='weight',
                        help='number of layers in the RNN')
    parser.add_argument('--initializer', type=str, default='kaiming_normal',
                        help='number of layers in the RNN')
    parser.add_argument('--num_objects', type=int, default=36)
    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=40,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=384,
                        help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.25,
                        help='clip gradients at this value')
    parser.add_argument('--dropC', type=float, default=0.5,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropG', type=float, default=0.2,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropL', type=float, default=0.1,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropW', type=float, default=0.4,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='strength of dropout in the Language Model RNN')

    # Optimization: for the Language Model

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--optim_alpha', type=float, default=0.9,
                        help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                        help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay')
    parser.add_argument('--seed', type=int, default=1000,
                        help='seed')
    parser.add_argument('--ntokens', type=int, default=777,
                        help='ntokens')
    parser.add_argument('--load_checkpoint_path')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='directory to store checkpointed models')


    # Other params that probably don't need to be changed
    parser.add_argument('--load_model_states', type=str, default=0,
                        help='which model to load')

    parser.add_argument('--evaluate_every', type=int, default=300,
                        help='which model to load')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_on_train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--predict_checkpoint', type=str)
    parser.add_argument('--change_scores_every_epoch', action='store_true')
    parser.add_argument('--test_has_regularization_split', action='store_true')
    parser.add_argument('--apply_answer_weight', action='store_true')
    parser.add_argument('--ignore_counting_questions', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[5, 10])
    parser.add_argument('--lr_gamma', type=float, default=1)

    # parser.add_argument('--loss_type', default=None)

    # SCR loss parameters
    parser.add_argument('--use_scr_loss', action='store_true')

    parser.add_argument('--num_sub', type=int, default=5,
                        help='size of the proposal object set')

    parser.add_argument('--bucket', type=int, default=4,
                        help='bucket of predicted answers')

    parser.add_argument('--scr_hint_loss_weight', type=float, default=0,
                        help='Influence strength loss weights')

    parser.add_argument('--scr_compare_loss_weight', type=float, default=0,
                        help='self-critical loss weights')

    parser.add_argument('--reg_loss_weight', type=float, default=0.0,
                        help='regularization loss weights, set to zero in our paper ')

    # Parameters for the main VQA loss
    parser.add_argument('--vqa_loss_weight', type=float, default=1)

    # HINT loss parameters
    parser.add_argument('--use_hint_loss', action='store_true')
    parser.add_argument('--hint_loss_weight', type=float, default=1)

    # Uniform KL loss
    parser.add_argument('--uniform_kl_loss_weight', type=float, default=100)
    parser.add_argument('--sensitivity_to_probability_using', default='softmax')

    # Parameters for our losses
    parser.add_argument('--num_most_sensitive_objects', default=1, type=int)
    parser.add_argument('--num_most_sensitive_words', default=7, type=int)

    # Non tail ended loss for objects
    parser.add_argument('--use_non_tail_loss_for_objects', action='store_true')
    parser.add_argument('--answers_for_non_tail_loss', type=str, default='')
    parser.add_argument('--non_tail_loss_margin_for_objects', default=0, type=float)
    parser.add_argument('--non_tail_loss_weight_for_objects', default=1, type=float)
    parser.add_argument('--use_absolute_for_non_tail_loss', action='store_true')

    # Loss to make sensitivity towards GT higher than that towards wrong answers
    parser.add_argument('--use_make_wrong_higher_than_gt_ans_loss', action='store_true')
    parser.add_argument('--make_wrong_higher_than_gt_ans_loss_weight', type=float, default=1)

    # Rolling head loss for objects
    parser.add_argument('--use_rolling_head_loss_for_objects', action='store_true')
    parser.add_argument('--rolling_head_loss_margin_for_objects', default=0, type=float)
    parser.add_argument('--rolling_head_loss_weight_for_objects', default=1, type=float)
    parser.add_argument('--use_absolute_for_rolling_head_loss', action='store_true')
    parser.add_argument('--dynamically_weight_rolling_head_loss', action='store_true')

    parser.add_argument('--auto_reweight_nte_loss', action='store_true')
    parser.add_argument('--use_equal_gt_vs_wrong_loss_for_objects', action='store_true')
    parser.add_argument('--equal_gt_vs_wrong_loss_weight_for_objects', type=float, default=1)

    # Equal gt vs wrong loss
    parser.add_argument('--num_wrong_answers', type=int, default=1)
    parser.add_argument('--use_absolute_for_equal_gt_vs_wrong_loss', action='store_true')

    parser.add_argument('--use_non_head_answers_loss', action='store_true')
    parser.add_argument('--non_head_answers_loss_weight', type=float, default=1e-2)
    parser.add_argument('--num_non_head_wrong_answers', type=int, default=10)
    parser.add_argument('--log_epochs', type=int, nargs='+', default=[5, 6, 7])

    # Fixed GT ans loss
    parser.add_argument('--use_fixed_gt_ans_loss', action='store_true')
    parser.add_argument('--fixed_ans_scores', type=float, default=[0], nargs='+')
    parser.add_argument('--fixed_gt_ans_loss_weight', type=float, default=1)
    parser.add_argument('--fixed_gt_ans_perturbation', type=float, default=0)
    parser.add_argument('--fixed_gt_ans_zeros_weight', type=float, default=1)
    parser.add_argument('--fixed_gt_ans_ones_weight', type=float, default=1)
    parser.add_argument('--fixed_gt_ans_loss_function', type=str, default='bce')

    # Random GT ans loss
    parser.add_argument('--use_random_gt_ans_loss', action='store_true')
    parser.add_argument('--random_gt_ans_loss_weight', type=float, default=1)

    args = parser.parse_args()

    return args