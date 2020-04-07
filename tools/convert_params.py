import torch
import os
from models.updn import UpDn
import torch.nn as nn
from copy import deepcopy


def convert_params():
    root = '/hdd/robik/projects/self_critical_vqa'
    ckpt = torch.load(os.path.join(root, 'saved_models', 'pretrained_1_default', 'model-best-scr-compatible.pth'))

    class Opt():
        def __init__(self):
            self.num_hid = 1280
            self.activation = 'ReLU'
            self.dropG = 0
            self.dropW = 0
            self.dropout = 0
            self.dropL = 0
            self.norm = 'weight'
            self.dropC = 0
            self.ntokens = 18455

    opt = Opt()
    model = nn.DataParallel(UpDn(opt))

    key_map = {
        "module.q_emb.rnn.weight_ih_l0": "module.q_emb.rnn.gru_cell.weight_ih.weight",
        "module.q_emb.rnn.weight_hh_l0": "module.q_emb.rnn.gru_cell.weight_hh.weight",
        "module.q_emb.rnn.bias_ih_l0": "module.q_emb.rnn.gru_cell.weight_ih.bias",
        "module.q_emb.rnn.bias_hh_l0": "module.q_emb.rnn.gru_cell.weight_hh.bias"
    }
    for k in key_map:
        new_k = key_map[k]
        ckpt[new_k] = deepcopy(ckpt[k])
        del ckpt[k]

    model.load_state_dict(ckpt)
    torch.save(ckpt, os.path.join(root, 'saved_models', 'pretrained_1_default', 'model-best.pth'))


if __name__ == "__main__":
    convert_params()
