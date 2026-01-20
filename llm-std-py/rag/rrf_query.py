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


prompt_rag_fusion = ChatPromptTemplate.from_template(
'''
하나의 입력 쿼리를 기반으로 여러 개의 검색 쿼리를 생성하는 유용한 어시스턴트 입니다.
다음과 관련된 여러 쿼리를 생성합니다.
{question}

출력(쿼리4개):
'''
)

def parse_queries_output(message):
    return message.content.split('\n')

llm = ChatOllama(model=gemma_model)

query_gen = prompt_rag_fusion | llm | parse_queries_output

retriever = db.as_retriever()

def reciprocal_rank_fusion(results: list[list], k=60):
    '''여러 순위 문서 목록에 대한 상호 순위 융합 및 RRF 공식에 사용되는 선택적 매개변수 k 입니다.'''
    # 사전을 초기화해 각 문서에 대한 융합된 점수를 보관
    # 고유성을 보장하기 위해 문서가 콘테츠별로 키를 생성
    fused_scores = {}
    documents = {}
    for docs in results:
        # 목록에 있는 각 문서를 순위(목록 내 위치)에 따라 반복
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc
            fused_scores[doc_str] += 1 / (rank + k)
    #융합된 점수를 기준으로 문서를 내림차순으로 정렬하여 최종 재순위 결과를 정리
    reranked_doc_strs = sorted(
        fused_scores, key=lambda d: fused_scores[d], reverse=True)
    return [documents[doc_str] for doc_str in reranked_doc_strs]

retrieval_chain = query_gen | retriever.batch | reciprocal_rank_fusion


## 프롬프트 구성
prompt = ChatPromptTemplate.from_template(
'''
다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트 : {context}
질문 : {question}
'''
)

query = '고대 그리스 철학사의 주요 인물은 누기인가요?'

@chain
def rag_fusion(input):
    #관련 문서 검색
    docs = retrieval_chain.invoke(input)
    formatted = prompt.invoke({'context':docs, 'question':input})
    return llm.invoke(formatted)

## 실행
print('RAG 융합 실행\n')
result = rag_fusion.invoke(query)
print(result)