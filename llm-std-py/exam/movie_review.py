import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

data = load_dataset("rotten_tomatoes")
###############################################
## 모델 설정 및 파이프라인 구축
# 허깅페이스 모델 경로
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
pipe = pipeline(
    model=model_path,
    tokenizer=model_path,
    top_k=None,
    device="mps"
)
#########################
# all-mpnet-base-v2 모델 로드
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# 텍스트를 임베딩으로 변환
train_embeddings = model.encode(list(data["train"]["text"]), show_progress_bar=True)
test_embeddings = model.encode(list(data["test"]["text"]), show_progress_bar=True)
print(train_embeddings.shape)

#########################
## 추론 예측을 수행
## KeyDataset: 대량의 데이터를 효율적으로 파이프라인에 전달하기 위해 사용
## 결과 매핑: 사용된 모델은 원래 **[부정, 중립, 긍정]**의 3가지 결과(Output)를 내놓는다
y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
    negative_score = output[0]["score"] # Negative 점수
    positive_score = output[2]["score"] # Positive 점수
    assignment = np.argmax([negative_score, positive_score]) # 둘 중 높은 쪽 선택
    y_pred.append(assignment)
#####################
## 평가 수행
def evaluate_performance(y_true, y_pred):
    """분류 리포트를 만들어 출력"""
    # classification_report: 정밀도(Precision), 재현율(Recall), F1-Score를 요약해서 보여준다.
    # 이를 통해 모델이 긍정 리뷰와 부정 리뷰를 얼마나 정확하게 분류했는지 한눈에 알 수 있다.
    performance = classification_report(
            y_true, y_pred,
            target_names=["Negative Review","Positive Review"]
        )
    print(performance)

###########################
## 분류 리포트 수행
print("=============== Before ==================")
evaluate_performance(data["test"]["label"], y_pred)

###########################
### 훈련세트 임베딩으로 로지스틱 회기 모델을 훈련
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])

## 모델 평가
print("=============== Evaluation ====================")
y_pred = clf.predict(test_embeddings)
evaluate_performance(data["test"]["label"], y_pred)

print("=============== Cosign Similarity ====================")

label_embeddings = model.encode(["A negative review", "A positive review"])
#각 문서와 가장 잘 맞는 레이블을 찾아낸다.
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)

evaluate_performance(data["test"]["label"], y_pred)