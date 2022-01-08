"""
My Transformer implementations
"""


import torch
from torch import nn
import math


class DotProdAttention(nn.Module):
    """calculating dot product self attention"""

    def __init__(self, in_dim, out_dim):
        super(DotProdAttention, self).__init__()
        self.key = nn.Linear(in_dim, out_dim)
        self.query = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)

    def forward(self, inputs):
        # q,k,v = (batch,seq,features)
        k = self.key(inputs)
        q = self.query(inputs)
        v = self.value(inputs)

        depth = q.shape[2]
        scores = torch.matmul(k, q.permute([0, 2, 1])) / math.sqrt(
            depth
        )  # (batch,seq,seq)

        softmax_scores = torch.softmax(scores, dim=2)
        return torch.matmul(softmax_scores, v)  # (batch,seq,features)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, feat_dim):
        super(MultiHeadAttention, self).__init__()
        assert feat_dim % num_heads == 0
        embed_dim = int(feat_dim / num_heads)

        self.num_heads = num_heads
        self.feat_dim = feat_dim

        self.attention_heads = nn.ModuleList(
            [DotProdAttention(feat_dim, embed_dim) for i in range(num_heads)]
        )

    def forward(self, inputs):

        l = []
        for layer in self.attention_heads:  ###err bad parralelism
            l.append(layer(inputs))

        out = torch.concat(l, dim=2)
        return out


class LayerNorm(nn.Module):
    """LayerNorm from https://arxiv.org/pdf/1607.06450.pdf"""

    def __init__(self, feat_dim):
        super(LayerNorm, self).__init__()

        self.bias = nn.parameter.Parameter(
            data=torch.zeros(feat_dim), requires_grad=True
        )
        self.gain = nn.parameter.Parameter(
            data=torch.ones(feat_dim), requires_grad=True
        )

        self.input_shape = feat_dim

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): tensor of shape (batch,seq,feat_dim)

        Returns:
            torch.tensor : layer normalized output
        """

        mean = torch.mean(inputs, dim=(1, 2), keepdim=True)
        var = torch.mean(torch.square(inputs - mean), dim=(1, 2), keepdim=True)
        std = torch.sqrt(var)

        norm = (inputs - mean) / std
        af_norm = self.gain * norm + self.bias
        return af_norm


class TransformerEncoder(nn.Module):
    # (batch, seq, features)
    def __init__(self, size, num_heads):
        super(TransformerEncoder, self).__init__()

        self.feedForward = nn.Sequential(
            nn.Linear(size, size), nn.ReLU(), nn.Linear(size, size)
        )

        self.selfAttention = MultiHeadAttention(num_heads, size)

        self.layerNorm1 = LayerNorm(size)
        self.layerNorm2 = LayerNorm(size)

    def forward(self, inputs):
        # (batch, seq, features)
        shape = inputs.shape

        attention = self.selfAttention(inputs)
        res = inputs + attention
        res = self.layerNorm1(res)

        x = self.feedForward(res)

        out = res + x
        out = self.layerNorm2(out)
        return out
