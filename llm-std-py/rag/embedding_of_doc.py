from langchain_ollama import OllamaEmbeddings

from config.load_sys import gemma_model

#임베딩 전용 gemma 모델
model = OllamaEmbeddings(model='embeddinggemma:300m')
embeddings = model.embed_documents([
    "여러분 안녕하세요!",
    "안녕?",
    "이름이 뭐에요?",
    "내 이름은 홍길동 이에요.",
    "반가워요"
])

print(embeddings)