import torch.nn as nn
from attention import compute_attention

class MultiheadAttention(nn.Module):
    def __init__(self, token_embed_dim, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.weight_q = nn.Linear(token_embed_dim, d_model)
        self.weight_k = nn.Linear(token_embed_dim, d_model)
        self.weight_v = nn.Linear(token_embed_dim, d_model)
        self.concat_linear = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        B,T,C = queries.size()
        queries = (self.weight_q(queries)
                   .view(B, T, self.n_heads, C // self.n_heads).transpose(1,2))

        keys = (self.weight_k(keys)
                   .view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2))

        values = (self.weight_v(values)
                .view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2))

        attention = compute_attention(queries, keys, values)
        output = attention.transpose(1,2).contiguous().view(B, T, C)
        output = self.concat_linear(output)
        return output