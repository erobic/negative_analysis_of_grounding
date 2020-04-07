import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """

    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        # assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:weight_init.shape[0]] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        self.ndirections = 1
        self.rnn = GRUModel(in_dim, num_hid)
        self.in_dim = in_dim
        self.num_hid = num_hid

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (1, batch, self.num_hid)
        return weight.new(*hid_shape).zero_()

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size).squeeze()
        output, hidden = self.rnn(x, hidden.squeeze())
        return output
# class QuestionEmbedding(nn.Module):
#     def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
#         """Module for question embedding
#         """
#         super(QuestionEmbedding, self).__init__()
#         assert rnn_type == 'LSTM' or rnn_type == 'GRU'
#         rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
#
#         # self.rnn = rnn_cls(
#         #     in_dim, num_hid, nlayers,
#         #     bidirectional=bidirect,
#         #     dropout=dropout,
#         #     batch_first=True)
#         self.rnn = GRUModel(in_dim, num_hid)
#
#         self.in_dim = in_dim
#         self.num_hid = num_hid
#         self.nlayers = nlayers
#         self.rnn_type = rnn_type
#         self.ndirections = 1 + int(bidirect)
#
#     def init_hidden(self, batch):
#         # just to get the type of tensor
#         weight = next(self.parameters()).data
#         hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
#         if self.rnn_type == 'LSTM':
#             return (weight.new(*hid_shape).zero_(),
#                     weight.new(*hid_shape).zero_())
#         else:
#             return weight.new(*hid_shape).zero_()
#
#     def forward(self, x):
#         # x: [batch, sequence, in_dim]
#         batch = x.size(0)
#         hidden = self.init_hidden(batch)
#         # self.rnn.flatten_parameters()
#         output, hidden = self.rnn(x, hidden)
#
#         if self.ndirections == 1:
#             return output[:, -1]
#
#         forward_ = output[:, -1, :self.num_hid]
#         backward = output[:, 0, self.num_hid:]
#         return torch.cat((forward_, backward), dim=1)
#
#     def forward_all(self, x):
#         # x: [batch, sequence, in_dim]
#         batch = x.size(0)
#         hidden = self.init_hidden(batch)
#         self.rnn.flatten_parameters()
#         output, hidden = self.rnn(x, hidden)
#         return output


class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.weight_ih(x)
        gate_h = self.weight_hh(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.gru_cell = GRUCell(input_dim, hidden_dim)

    def forward(self, x, hidden):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        outs = []
        for seq in range(x.size(1)):
            hidden = self.gru_cell(x[:, seq, :], hidden)
            outs.append(hidden)

        out = outs[-1].squeeze()
        return out, hidden
