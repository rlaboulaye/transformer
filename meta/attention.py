import math

import torch
from torch import nn


class Attention(nn.Module):

    def __init__(self, embed_dim, seq_dim, num_head, attn_pdrop, resid_pdrop, scale=True):
        super(Attention, self).__init__()
        assert embed_dim % num_head == 0
        self.num_head = num_head
        self.split_size = embed_dim
        self.scale = scale
        self.W_attn = nn.Linear(embed_dim, embed_dim * 3)
        self.W_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, key=False):
        new_x_shape = x.size()[:-1] + (self.num_head, x.size(-1) // self.num_head)
        x = x.view(*new_x_shape)
        if key:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        qkv = self.W_attn(x)
        query, key, value = qkv.split(self.split_size, dim=2)
        query = query.mean(dim=-2, keepdim=True)
        query = self.split_heads(query)
        key = self.split_heads(key, key=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.W_proj(a)
        a = self.resid_dropout(a)
        return a.squeeze(-2)