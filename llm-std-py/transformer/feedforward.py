import torch.nn as nn

class PreLayerNormFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward) ## 선형 층 1
        self.linear2 = nn.Linear(dim_feedforward, d_model) ## 선형 층 2
        self.dropout1 = nn.Dropout(dropout) ## 드롭아웃 층 1
        self.dropout2 = nn.Dropout(dropout) ## 드롭아웃 층 2
        self.activation = nn.GELU() ## 활성 함수
        self.norm = nn.LayerNorm(d_model) ## 층 정규화

    def forward(self, x):
        x = self.norm(x)
        x = x + self.linear2(self.dropout1(self.activation(self.linear1(x))))
        x = self.dropout2(x)
        return x
