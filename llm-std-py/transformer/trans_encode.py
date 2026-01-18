import torch.nn as nn

from feedforward import PreLayerNormFeedForward
from multihead_attention import MultiheadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        super().__init__()
        self.attn = MultiheadAttention(d_model, d_model, n_head) ## 멀티 헤드 어텐션
        self.norm1 = nn.LayerNorm(d_model)                       ## 층 정규화
        self.dropout1 = nn.Dropout(dropout)                       ## 드롭아웃
        self.feedForward = PreLayerNormFeedForward(d_model, dim_feedforward, dropout) ## 피드 포워드


    def forward(self, src):
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


