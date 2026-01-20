# 텍스트 로드 후 임베딩
import uuid

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

## 1. 문서 로드
loader = TextLoader('data/hello.txt')
docs = loader.load()

## 2. 문서 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

## 3. 임베딩 생성
embedding_model = OllamaEmbeddings(model='embeddinggemma:300m')
embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])

# 4. 벡터 데이터베이스 저장
connection = "postgresql+psycopg://postgres:postgres!qwe123@localhost:5432/test_vector_db"
db = PGVector.from_documents(chunks, embedding_model, connection=connection)


## query 조회
results = db.similarity_search('지역', k=5)
print(results)

## 문서 추가
ids = [str(uuid.uuid4()), str(uuid.uuid4())]
db.add_documents([
    Document(page_content='일하는 지역은 서울이에요',
             meta_data={'location:':'서울', 'topic':'지역'}),
    Document(page_content='서울에 있는 강남 지역에서 일해요.',
             meta_data={'location:':'지역', 'topic':'지역'}),
],
ids=ids)

## 문서 삭제
# db.delete(ids=ids)