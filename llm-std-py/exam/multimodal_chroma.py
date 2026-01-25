import os

from PIL import Image
from datasets import load_dataset
from langchain_ollama import ChatOllama
from langchain_teddynote.models import MultiModal
from matplotlib import pyplot as plt
import open_clip
import pandas as pd
import numpy as np
from langchain_experimental.open_clip import OpenCLIPEmbeddings

from config.load_sys import gemma_model

# COCO 데이터셋 로드
dataset = load_dataset(
    path="detection-datasets/coco", name="default", split="train", streaming=True
)

# 이미지 저장 폴더와 이미지 개수 설정
IMAGE_FOLDER = "tmp"
N_IMAGES = 20

# 그래프 플로팅을 위한 설정
plot_cols = 5
plot_rows = N_IMAGES // plot_cols
fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(plot_rows * 2, plot_cols * 2))
axes = axes.flatten()

# 이미지를 폴더에 저장하고 그래프에 표시
dataset_iter = iter(dataset)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
for i in range(N_IMAGES):
    # 데이터셋에서 이미지와 레이블 추출
    data = next(dataset_iter)
    image = data["image"]
    label = data["objects"]["category"][0]  # 첫 번째 객체의 카테고리를 레이블로 사용

    # 그래프에 이미지 표시 및 레이블 추가
    axes[i].imshow(image)
    axes[i].set_title(label, fontsize=8)
    axes[i].axis("off")

    # 이미지 파일로 저장
    image.save(f"{IMAGE_FOLDER}/{i}.jpg")

# 그래프 레이아웃 조정 및 표시
plt.tight_layout()
plt.show()


# 사용 가능한 모델/Checkpoint 를 출력
pd.DataFrame(open_clip.list_pretrained(), columns=["model_name", "checkpoint"]).head(10)

# OpenCLIP 임베딩 함수 객체 생성
image_embedding_function = OpenCLIPEmbeddings(
    model_name="ViT-H-14-378-quickgelu", checkpoint="dfn5b"
)

# 이미지의 경로를 리스트로 저장
image_uris = sorted(
    [
        os.path.join("tmp", image_name)
        for image_name in os.listdir("tmp")
        if image_name.endswith(".jpg")
    ]
)

llm = ChatOllama(model=gemma_model)
# MultiModal 모델 설정
model = MultiModal(
    model=llm,
    system_prompt="Your mission is to describe the image in detail",  # 시스템 프롬프트: 이미지를 상세히 설명하도록 지시
    user_prompt="Description should be written in one sentence(less than 60 characters)",  # 사용자 프롬프트: 60자 이내의 한 문장으로 설명 요청
)


# 이미지 설명 생성
model.invoke(image_uris[0])


# 이미지 설명
descriptions = dict()

for image_uri in image_uris:
    descriptions[image_uri] = model.invoke(image_uri, display_image=False)

# 생성된 결과물 출력
descriptions



# 원본 이미지, 처리된 이미지, 텍스트 설명을 저장할 리스트 초기화
original_images = []
images = []
texts = []

# 그래프 크기 설정 (20x10 인치)
plt.figure(figsize=(20, 10))

# 'tmp' 디렉토리에 저장된 이미지 파일들을 처리
for i, image_uri in enumerate(image_uris):
    # 이미지 파일 열기 및 RGB 모드로 변환
    image = Image.open(image_uri).convert("RGB")

    # 4x5 그리드의 서브플롯 생성
    plt.subplot(4, 5, i + 1)

    # 이미지 표시
    plt.imshow(image)

    # 이미지 파일명과 설명을 제목으로 설정
    plt.title(f"{os.path.basename(image_uri)}\n{descriptions[image_uri]}", fontsize=8)

    # x축과 y축의 눈금 제거
    plt.xticks([])
    plt.yticks([])

    # 원본 이미지, 처리된 이미지, 텍스트 설명을 각 리스트에 추가
    original_images.append(image)
    images.append(image)
    texts.append(descriptions[image_uri])

# 서브플롯 간 간격 조정
plt.tight_layout()


# 이미지와 텍스트 임베딩
# 이미지 URI를 사용하여 이미지 특징 추출
img_features = image_embedding_function.embed_image(image_uris)
# 텍스트 설명에 "This is" 접두사를 추가하고 텍스트 특징 추출
text_features = image_embedding_function.embed_documents(
    ["This is " + desc for desc in texts]
)

# 행렬 연산을 위해 리스트를 numpy 배열로 변환
img_features_np = np.array(img_features)
text_features_np = np.array(text_features)

# 유사도 계산
# 텍스트와 이미지 특징 간의 코사인 유사도를 계산
similarity = np.matmul(text_features_np, img_features_np.T)


# 유사도 행렬을 시각화하기 위한 플롯 생성
count = len(descriptions)
plt.figure(figsize=(20, 14))

# 유사도 행렬을 히트맵으로 표시
plt.imshow(similarity, vmin=0.1, vmax=0.3, cmap="coolwarm")
plt.colorbar()  # 컬러바 추가

# y축에 텍스트 설명 표시
plt.yticks(range(count), texts, fontsize=18)
plt.xticks([])  # x축 눈금 제거

# 원본 이미지를 x축 아래에 표시
for i, image in enumerate(original_images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

# 유사도 값을 히트맵 위에 텍스트로 표시
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

# 플롯 테두리 제거
for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)

# 플롯 범위 설정
plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])

# 제목 추가
plt.title("텍스트와 이미지 특징 간의 코사인 유사도", size=20)

