# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ipdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

import egg.core as core


class Receiver(nn.Module):
    def __init__(self, n_bits, n_hidden):
        super(Receiver, self).__init__()
        self.emb_column = nn.Linear(n_bits, n_hidden)

        self.fc1 = nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.fc2 = nn.Linear(2 * n_hidden, n_bits)

    def forward(self, embedded_message, bits):
        embedded_bits = self.emb_column(bits.float())

        # if len(embedded_bits.size()) != len(embedded_message.size()):
        #     ipdb.set_trace()
        x = torch.cat([embedded_bits, embedded_message], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        return x.sigmoid()


class ReinforcedReceiver(nn.Module):
    def __init__(self, n_bits, n_hidden):
        super(ReinforcedReceiver, self).__init__()
        self.emb_column = nn.Linear(n_bits, n_hidden)

        self.fc1 = nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.fc2 = nn.Linear(2 * n_hidden, n_bits)

    def forward(self, embedded_message, bits):
        embedded_bits = self.emb_column(bits.float())

        x = torch.cat([embedded_bits, embedded_message], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        probs = x.sigmoid()

        distr = Bernoulli(probs=probs)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = (probs > 0.5).float()
        log_prob = distr.log_prob(sample).sum(dim=1)
        return sample, log_prob, entropy


class Sender(nn.Module):
    def __init__(self, vocab_size, n_bits, n_hidden):
        super(Sender, self).__init__()
        self.emb = nn.Linear(n_bits, n_hidden)
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, bits):
        x = self.emb(bits.float())
        x = F.leaky_relu(x)
        message = self.fc(x)

        return message


class BSChannel(nn.Module):
    def __init__(self, pe):
        super().__init__()
        self.pe = pe

    def ohe2bit(self, ohe):
        L = np.ceil(np.log2(ohe.size()[-1]))
        bit_mask = 2**torch.arange(L-1, -1, -1).to(ohe.device, torch.long)
        idx = ohe.argmax(-1)
        bits = idx.unsqueeze(-1).bitwise_and(bit_mask).ne(0).byte()
        return bits

    def bit2ohe(self, bits):
        L = bits.size()[-1]
        bit_mask = 2**torch.arange(L-1, -1, -1).to(bits.device)
        idx = torch.sum(bit_mask * bits, -1, dtype=torch.long)
        message = torch.nn.functional.one_hot(idx, 2**L)
        return message

    def forward(self, message_ohe):
        in_bits = self.ohe2bit(message_ohe)
        p = torch.tensor([self.pe]).to(in_bits.device).expand_as(in_bits)
        noise = torch.bernoulli(p)
        out_bits = (in_bits + noise) % 2
        out_ohe = self.bit2ohe(out_bits).to(message_ohe.dtype)
        return out_ohe
