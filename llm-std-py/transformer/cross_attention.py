import torch.nn as nn
from feedforward import PreLayerNormFeedForward
from multihead_attention import MultiheadAttention


class TransformerDecoderLayer(nn.Module):
    def __init__(self, token_embed_dim, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, d_model, n_heads)
        self.multihead_attn = MultiheadAttention(d_model, d_model, n_heads)
        self.feedForward = PreLayerNormFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output, is_causal=True):
        # 셀프 어텐션 계산
        x = self.norm1(tgt)
        x = x + self.dropout1(self.self_attn(x, x, x,
        is_causal = True))
        # 크로스 어텐션 연산
        x = self.norm2(x)
        x = x + self.dropout2(self.multihead_attn(x, encoder_output, encoder_output))
        # 피드 포워드 연산
        x = self.feed_forward(x)
        return x
