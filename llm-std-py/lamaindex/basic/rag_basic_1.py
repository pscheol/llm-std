from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from config.load_sys import gemma_model

#1. LLM 설정
Settings.llm = Ollama(
    model=gemma_model,
    request_timeout=360
)

## 임베딩 모델 설정
Settings.embed_model = OllamaEmbedding(
    model_name='embeddinggemma:300m',
    base_url="http://localhost:11434", # Ollama 기본 주소
)


documents = [
    Document(text="라마인덱스 스터디 Ollama를 이용하여 RAG 구현이 가능."),
    Document(text="임베딩모델은 젬마 300m을 사용"),
]

# 인덱스 생성
index = VectorStoreIndex.from_documents(documents)

#검색 및 질의
query_engine = index.as_query_engine()

response = query_engine.query("Ollama 장점")

print("-" * 50)
print(f"답변:\n{response}")
print("-" * 50)