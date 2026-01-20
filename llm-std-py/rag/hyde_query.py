from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector

from config.load_sys import gemma_model

connection = 'postgresql+psycopg://postgres:postgres!qwe123@localhost:5432/test_vector_db'

# 문서에 대한 임베딩 생성
embedding_model = OllamaEmbeddings(model='embeddinggemma:300m')
# PGVector 벡터 스토어 초기화
db = PGVector(
    embeddings=embedding_model,
    connection=connection,
    use_jsonb=True, # 속도 및 인덱싱 최적화를 위해 JSONB 사용 권장
)

prompt_hyde = ChatPromptTemplate.from_template(
'''
질문에 답할 구절을 작성해주세요.
질문: {question}
구절:
'''
)

llm = ChatOllama(model=gemma_model, temperature=0)

generate_doc = (prompt_hyde | llm | StrOutputParser())


## 문서 검색 체인
retriever = db.as_retriever()
retrieval_chain = generate_doc | retriever

## 출력 생성
prompt = ChatPromptTemplate.from_template(
'''
다음 컨텍스트만 사용해 질문에 대답하세요.
컨텍스트: {context}
질문: {question}
'''
)

@chain
def qa(input):
    docs = retrieval_chain.invoke(input)
    formatted = prompt.invoke({'context':docs, 'question':input})
    return llm.invoke(formatted)

query = '고대 그리스 철학사의 주요 인물은 누기인가요?'
print('HyDE 실행\n')
result = qa.invoke(query)
print("결과 : ", result)