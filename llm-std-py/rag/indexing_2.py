from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import chain
from config.load_sys import gemma_model

connection = 'postgresql+psycopg://postgres:postgres!qwe123@localhost:5432/test_vector_db'

# 문서를 로드 후 분할
raw_documents = TextLoader('data/sample2.txt', encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(raw_documents)


# 문서에 대한 임베딩 생성
embedding_model = OllamaEmbeddings(model='embeddinggemma:300m')

db = PGVector.from_documents(documents, embedding_model, connection=connection)


retriever = db.as_retriever()
query = '내 이름은 뭐야?'
docs = retriever.invoke(query)

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트만 사용해 질문을 답하세요.
컨텍스트:{context}
질문:{question}
""")

llm = ChatOllama(model=gemma_model)
llmChain = prompt | llm

result = llmChain.invoke({'context':docs, 'question':query})
print("답변 : ", result)


## 단일함수로 캡슐화

retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트만 사용해 질문을 답하세요.
컨텍스트:{context}
질문:{question}
""")

llm = ChatOllama(model=gemma_model)

@chain
def qa(input):
    # 관련 문서 검색
    docs = retriever.invoke(input)
    # 프롬프트 포메팅
    formatted = prompt.invoke({'context':docs, 'question':input})
    # 답변 생성
    answer = llm.invoke(formatted)

    return answer

result = qa.invoke(query)
print("@chain 답변 : " ,result)