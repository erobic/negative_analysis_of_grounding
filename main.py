import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dictionary, SelfCriticalDataset

from models.updn import UpDn
import opts
from train import run
from torch.optim.lr_scheduler import MultiStepLR
from dataset import RandomSubsetSampler


def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data, a=0.01)


if __name__ == '__main__':
    opt = opts.parse_opt()
    seed = 0

    ## Set random seeds for reproducibility
    if opt.seed == 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        seed = opt.seed
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = True  # For reproducibility

    # If we are using fixed target (e.g, zero-vector for answers), then we need to adjust learning rate based upon the size of the dataset we want to do this on
    if opt.use_fixed_gt_ans_loss:
        if opt.var_random_subset_ratio is not None:
            opt.learning_rate = (2e-6) / opt.var_random_subset_ratio
        else:
            opt.learning_rate = (2e-6) / opt.fixed_random_subset_ratio

    dictionary = Dictionary.load_from_file(f'{opt.data_dir}/dictionary.pkl')
    opt.ntokens = dictionary.ntoken

    if opt.use_scr_loss:
        opt.apply_answer_weight = True

    ## Create model
    model = UpDn(opt)
    model = model.cuda()
    model.apply(weights_init_kn)
    model = nn.DataParallel(model).cuda()
    model.train()

    # The dataset used for training
    train_dset = SelfCriticalDataset(opt.split, opt.hint_type, dictionary, opt,
                                     discard_items_without_hints=not opt.do_not_discard_items_without_hints,
                                     ignore_counting_questions=opt.ignore_counting_questions)

    if opt.fixed_random_subset_ratio is not None:
        ## If fixed_random_subset_ratio flag is set to True, then load only a subset of data at the beginning of the experiment
        # And use it through that experiment
        shuffle = False
        subset_sampler = RandomSubsetSampler(torch.LongTensor(range(0, len(train_dset))),
                                             int(len(train_dset) * opt.fixed_random_subset_ratio))
    else:
        shuffle = True
        subset_sampler = None
    train_loader = DataLoader(train_dset,
                              opt.batch_size,
                              shuffle=shuffle,
                              num_workers=opt.num_workers,
                              sampler=subset_sampler)

    # Entire training set
    train_dset_all = SelfCriticalDataset(opt.split, None, dictionary, opt)
    train_loader_all = DataLoader(train_dset_all, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # # Dataset used to evaluate performance on the training instances used for regularization
    # train_dset_for_reg = SelfCriticalDataset(opt.split, opt.hint_type, dictionary, opt)
    # train_dset_for_reg_loader = DataLoader(train_dset_for_reg, opt.batch_size, shuffle=False,
    #                                        num_workers=opt.num_workers)
    #
    # train_dset_except_reg = SelfCriticalDataset(opt.split, opt.hint_type, dictionary, opt, exclude_items_with_hints=True)
    # train_dset_except_reg_loader = DataLoader(train_dset_except_reg, opt.batch_size, shuffle=False,
    #                                           num_workers=opt.num_workers)
    train_dset_for_reg_loader = None
    train_dset_except_reg_loader = None
    eval_dset = SelfCriticalDataset(opt.split_test, None, dictionary, opt)
    eval_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    if opt.test_has_regularization_split:
        ## regularization split refers to the subset of dataset with hints (e.g., HAT subset or subset with textual explanations)
        test_dset_for_reg = SelfCriticalDataset(opt.split_test, opt.hint_type, dictionary, opt)
        test_dset_for_reg_loader = DataLoader(test_dset_for_reg, opt.batch_size, shuffle=False,
                                              num_workers=opt.num_workers)

        test_dset_except_reg = SelfCriticalDataset(opt.split_test, opt.hint_type, dictionary, opt,
                                                   discard_items_with_hints=True)
        test_dset_except_reg_loader = DataLoader(test_dset_except_reg, opt.batch_size, shuffle=False,
                                                 num_workers=opt.num_workers)
    else:
        test_dset_for_reg_loader = None
        test_dset_except_reg_loader = None

    run(model,
        train_loader,
        eval_loader,
        opt,
        train_loader_all=train_loader_all,
        train_loader_for_regularization=train_dset_for_reg_loader,
        train_loader_except_regularization=train_dset_except_reg_loader,
        eval_loader_for_regularization=test_dset_for_reg_loader,
        eval_loader_except_regularization=test_dset_except_reg_loader)
