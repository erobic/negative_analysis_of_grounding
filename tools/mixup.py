import torch
import numpy as np


def mixup_tensors(x, alpha=1.0, use_cuda=True, dim=0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    dim_size = x.shape[dim]
    if use_cuda:
        index = torch.randperm(dim_size).cuda()
    else:
        index = torch.randperm(dim_size)

    if dim == 0:
        mixed_x = lam * x + (1 - lam) * x[index]
    elif dim == 1:
        mixed_x = lam * x + (1 - lam) * x[:, index]
    else:
        raise Exception("We don't yet support mixup for dimensions other than 0 or 1.")
    return mixed_x
