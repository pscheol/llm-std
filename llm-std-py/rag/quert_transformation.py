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

prompt = ChatPromptTemplate.from_template("""
당신은 질문-답변(Question-Answer) Task 를 수행한는 AI 어시스턴트 입니다.
검색된 문맥(context)를 사용하여 질문(question)에 답하세요. 
만약, 문맥(context) 으로부터 답을 찾을 수 없다면 '모른다' 고 말하세요. 
한국어로 대답하세요.

#Question: 
{question}

#Context: 
{context}
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

query = '일어나서 이를 닦고 뉴스를 읽었어요. 그러다 전자레인지에 음식을 넣어둔 걸 깜빡했네요. 고대 그리스 철학사의 주요 인물은 누구인가요?'
result = qa.invoke(query)
print("답변 : " ,result)


#RRR

rewrite_prompt = ChatPromptTemplate.from_template("""
웹 검색 엔진이 주어진 질문에 답할 수 있도록 더 나은 영문 검색어를 제공해주세요. 쿼리는 \'**\'로 끝내세요.
질문 : {x}

답변: 
""")

def parse_rewriter_output(message):
    return message.content.strip('\'').strip('**')


rewriter = rewrite_prompt | llm | parse_rewriter_output
@chain
def qa_rrr(input):
    #쿼리 재작성
    new_query = rewriter.invoke(input)
    #관련 문서 검색
    docs = retriever.invoke(new_query)
    # 프롬프트 포메팅
    formatted = prompt.invoke({'context': docs, 'question': input})
    # 답변 생성
    answer = llm.invoke(formatted)
    return answer


result = qa_rrr.invoke(query)
print("new 답변 : " ,result)