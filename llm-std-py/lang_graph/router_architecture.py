from typing import TypedDict, Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.constants import START, END

from config.load_sys import gemma_model
from langgraph.graph import add_messages, StateGraph



embeddings = OllamaEmbeddings(model='embeddinggemma:300m')

#SQL 쿼리 생성용
model_low_temp = ChatOllama(model=gemma_model, temperature=0.1)
# 자연어 출력 생성용
model_high_temp = ChatOllama(model=gemma_model, temperature=0.7)

class State(TypedDict):
    # 대화 기록
    messages: Annotated[list, add_messages]
    # 입력
    user_query: str
    # 출력
    domain: Literal['records', 'insurance']
    documents: list[Document]
    answer: str


class Input(TypedDict):
    user_query: str


class Output(TypedDict):
    documents: list[Document]
    answer: str


## inmemory 벡터 저장
medical_records_store = InMemoryVectorStore.from_documents([], embeddings)
medical_records_retriever = medical_records_store.as_retriever()

insurance_faqs_store = InMemoryVectorStore.from_documents([], embeddings)
insurance_faqs_retriever = insurance_faqs_store.as_retriever()


router_prompt = SystemMessage(
'''
사용자 문의는 어느 도메인으로 라우팅할지 결정하세요. 선택할 수 있는 두 가지 도메인은 다음과 같습니다.
- record: 진단, 치료, 처방과 같은 환자의 의료 기록을 포함
- insurance: 보험 정책, 청구, 보장에 대한 자주 묻는 질문을 포함

도메인 이름만 출력하세요.
'''
)


def router_node(state: State) -> State:
    user_message = HumanMessage(state['user_query'])
    messages = [router_prompt, *state['messages'], user_message]
    res = model_low_temp.invoke(messages)
    return {
        'domain': res.content,
        'messages': [user_message, res],
    }

def pick_retriever(state: State) -> Literal['retrieve_medical_records', 'retrieve_insurance_faqs']:
    if state['domain'] == 'records':
        return 'retrieve_medical_records'
    else:
        return 'retrieve_insurance_faqs'

def retrieve_medical_records(state: State) -> State:
    documents = medical_records_retriever.invoke(state['user_query'])
    return { 'documents': documents }


def retrieve_insurance_faqs(state: State) -> State:
    documents = insurance_faqs_retriever.invoke(state['user_query'])
    return { 'documents': documents }


medical_records_prompt = SystemMessage(
    '당신은 유능한 의료 챗봇입니다. 진단, 치료, 처방과 같은 환자의 의료 기록을 기반으로 질문에 답하세요.'
)
insurance_faqs_prompt = SystemMessage(
    '당신은 유능한 의료 보험 챗봇입니다. 보험 정책, 청구 및 보장에 대한 자주 묻는 질문에 답하세요.'
)

def generate_answer(state: State) -> State:
    if state['domain'] == 'records':
        prompt = medical_records_prompt
    else:
        prompt = insurance_faqs_prompt
    messages = [
        prompt,
        *state['messages'],
        HumanMessage(f'Documents: {state['documents']}'),
    ]
    res = model_high_temp.invoke(messages)
    return {
        'answer': res.content,
        'messages': res,
    }

## input=Input, output=Output 파라미터는 deprecated 됨
builder = StateGraph(State, input_schema=Input, output_schema=Output)
builder.add_node('router', router_node)
builder.add_node('retrieve_medical_records', retrieve_medical_records)
builder.add_node('retrieve_insurance_faqs', retrieve_insurance_faqs)
builder.add_node('generate_answer', generate_answer)
builder.add_edge(START, 'router')
builder.add_conditional_edges('router', pick_retriever)
builder.add_edge('retrieve_medical_records', 'generate_answer')
builder.add_edge('retrieve_insurance_faqs', 'generate_answer')
builder.add_edge('generate_answer', END)

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path='graph_router.png')

input = {'user_query': '알레르기 비염 면역치료도 실비보험이 적용되나요?'}
for chunk in graph.stream(input):
    print(chunk)