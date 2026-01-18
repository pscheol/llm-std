import torch.nn as nn
import torch.nn.functional as F
import torch
from math import sqrt


def compute_attention(query, key, value, is_causal=False):
    dim_k = query.size(-1)
    score = query @ key.transpose(-2, -1) / sqrt(dim_k)


    if is_causal:
        query_length = query.size(-2)
        key_length = key.size(-2)
        temp_mask = torch.ones(query_length, key_length, dtype=torch.bool).tril(diagonal=0)
        score = score.masked_fill(temp_mask == False, float("-inf"))
    weight = F.softmax(score, dim=-1)

    return weight @ value



class AttentionHead(nn.Module):
    def __init__(self, token_embed_dim, head_dim):
        super().__init__()
        self.weight_q = nn.Linear(token_embed_dim, head_dim)
        self.weight_k = nn.Linear(token_embed_dim, head_dim)
        self.weight_v = nn.Linear(token_embed_dim, head_dim)

    def forward(self, queries, keys, values):
        return compute_attention(
            self.weight_q(queries)
            , self.weight_k(keys)
            , self.weight_v(values))

