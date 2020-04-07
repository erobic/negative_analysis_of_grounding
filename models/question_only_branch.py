import torch.nn as nn

import models.mlp as mlp
import utils


class QuestionOnlyBranch(nn.Module):
    def __init__(self, q_emb_dim, hid_dim, num_classes):
        super().__init__()
        self.mlp = mlp.MLP(input_dim=q_emb_dim,
                           dimensions=[hid_dim, hid_dim, num_classes])

    def forward(self, q_emb):
        # We want q-only branch to change based on q_emb but not the other way round
        q_emb = utils.grad_mul_const(q_emb, 0)
        q_pred = self.mlp(q_emb)
        return q_pred
