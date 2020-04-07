import torch.nn as nn

from models.question_only_branch import QuestionOnlyBranch
import utils


class QuestionOnlyWrapper(nn.Module):
    def __init__(self, main_model, q_emb_dim, q_only_hid_dim, num_classes):
        super().__init__()
        self.main_model = main_model
        self.question_only_branch = QuestionOnlyBranch(q_emb_dim, q_only_hid_dim, num_classes)

    def forward(self, ):
        out = {}
        main_out = self.main_model()