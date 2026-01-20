from langchain_classic.indexes import SQLRecordManager, index
from langchain_ollama import OllamaEmbeddings
from langchain_classic.docstore.document import Document
from langchain_postgres import PGVector

connection = 'postgresql+psycopg://postgres:postgres!qwe123@localhost:5432/test_vector_db'
collection_name = "my_docs"

embedding_model = OllamaEmbeddings(model='embeddinggemma:300m')
namespace = "my_docs_namespace"

vectorStore = PGVector(
    embeddings=embedding_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True
)

record_manger = SQLRecordManager(
    namespace,
    db_url=connection
)

## 스키마가 없을 경우 생성
record_manger.create_schema()

## 문서 생성
docs = [
    Document(page_content='고양이가 연못에 있어요.',
             metadata={'id':1, 'source':'cat.txt'}),
    Document(page_content='오리도 연못에 있어요.',
             metadata={'id': 2, 'source': 'duck.txt'}),
]

# 문서 인덱싱 1회차ㅣ
index_1 = index(
    docs,
    record_manger,
    vectorStore,
    cleanup='incremental', ## 문서 중복 방지
    source_id_key='source' ## 출처를 source_id로 사용
)

print(f"인덱싱 1회차 : {index_1}")

# 문서 인덱싱 2회차, 중복 문서 생서 안됨
index_2 = index(
    docs,
    record_manger,
    vectorStore,
    cleanup='incremental',
    source_id_key='source'
)

print(f"인덱싱 2회차 : {index_2}")

# 문서를 수정하면 새 버전을 저장 후 출처가 같은 기존 문서는 제거
docs[0].page_content = '나는 문서를 수정해봅니다.'

index_3 = index(
    docs,
    record_manger,
    vectorStore,
    cleanup='incremental',
    source_id_key='source'
)

print(f"인덱싱 3회차 : {index_3}")