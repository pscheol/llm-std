import torch
import torch.nn as nn

from attention import AttentionHead

# input_text =('나는 최근에 파리 여행을 다녀왔다.')
# input_text_list = input_text.split()
#
# str2idx = {word:idx for idx, word in enumerate(input_text_list)}
#
# # 텍스트를 숫자 ID 리스트로 변환
# input_ids = [str2idx[word] for word in input_text_list]
#
# embedding_dim = 16
# embed_layer = nn.Embedding(len(str2idx), embedding_dim)
#
# # input_ids를 LongTensor로 변환하여 임베딩 레이어에 전달
# input_tensor = torch.tensor(input_ids)
# input_embeddings = embed_layer(input_tensor)
#
# # 배치 차원 추가
# input_embeddings = input_embeddings.unsqueeze(0)
#
# # 결과 텐서의 모양 출력
# print(input_embeddings.shape)

#
# ####################################################
# # 위치인코딩
# max_position = 12
# embed_layer = nn.Embedding(len(str2idx), embedding_dim)
# position_embed_layer = nn.Embedding(max_position, embedding_dim)
#
# position_ids = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0)
# position_encodings = position_embed_layer(position_ids)
#
# token_embeddings = embed_layer(torch.tensor(input_ids))
# token_embeddings = token_embeddings.unsqueeze(0)
# input_embeddings = token_embeddings + position_encodings
# print(input_embeddings.shape)
#
# ####################################################
# ### 어텐션
# head_dim = 16
# # 쿼리, 키 값을 계산하기 위한 변환
# weight_q = nn.Linear(embedding_dim, head_dim)
# weight_k = nn.Linear(embedding_dim, head_dim)
# weight_v = nn.Linear(embedding_dim, head_dim)
#
# ## 변환 수행
# querys = weight_q(input_embeddings)
# keys = weight_k(input_embeddings)
# values = weight_v(input_embeddings)
#
# print(f"querys={querys}, keys={keys}, values={values}")
#
# from math import sqrt
# import torch.nn.functional as F
#
# def compute_attention(query, key, value):
#     dim_k = query.size(-1)
#     score = query @ key.transpose(-2, -1) / sqrt(dim_k)
#     weight = F.softmax(score, dim=-1)
#     return weight @ value
#
# print(f"attention={compute_attention(querys, keys, values).shape}")
#

######################
import attention
from multihead_attention import MultiheadAttention

input_text =('나는 최근에 파리 여행을 다녀왔다.')
input_text_list = input_text.split()

embedding_dim = 16

str2idx = {word:idx for idx, word in enumerate(input_text_list)}
input_ids = [str2idx[word] for word in input_text_list]

input_tensor = torch.tensor(input_ids)
embed_layer = nn.Embedding(len(str2idx), embedding_dim)
input_embeddings = embed_layer(input_tensor)
input_embeddings = input_embeddings.unsqueeze(0)
####################################################################

attention_head = AttentionHead(embedding_dim, embedding_dim)
after_attention_embeddings = attention_head(input_embeddings, input_embeddings, input_embeddings)

print(f"after_attention_embeddings={after_attention_embeddings.shape}")


n_head = 4
mh_attention = MultiheadAttention(embedding_dim, embedding_dim, n_head)
after_attention_embeddings = mh_attention(input_embeddings, input_embeddings, input_embeddings)
print(f"after_attention_embeddings={after_attention_embeddings.shape}")