from typing import TypedDict, Annotated

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.constants import START, END

from config.load_sys import gemma_model
from langgraph.graph import add_messages, StateGraph

#SQL 쿼리 생성용
model_low_temp = ChatOllama(model=gemma_model, temperature=0.1)
# 자연어 출력 생성용
model_high_temp = ChatOllama(model=gemma_model, temperature=0.7)

class State(TypedDict):
    # 대화기록
    messages: Annotated[list, add_messages]
    # 입력
    user_query: str
    # 출력
    sql_query: str
    sql_explanation: str

class Input(TypedDict):
    user_query: str

class Output(TypedDict):
    sql_query: str
    sql_explanation: str

generate_prompt = SystemMessage(
    '당신은 친절한 데이터 분석가입니다. 사용자의 질문을 바탕으로 SQL 쿼리를 작성해주세요.'
)

def generate_sql(state: State) -> State:
    user_message = HumanMessage(state['user_query'])
    messages = [generate_prompt, *state['messages'], user_message]
    res = model_low_temp.invoke(messages)
    return {
        'sql_query': res.content,
        # 대화 기록 업데이트
        'messages': [user_message, res]
    }

explain_prompt = SystemMessage(
    '당신은 친절한 데이터 분석가입니다. 사용자에게 SQL 쿼리를 설명해주세요.'
)

def explain_sql(state: State) -> State:
    messages = [
        explain_prompt,
        # 이전 단계의 사용자의 질문과 SQL 쿼리
        *state['messages']
    ]
    res = model_high_temp.invoke(messages)
    return {
        'sql_explanation': res.content,
        # 대화 기록 업데이트
        'messages': res
    }

## input=Input, output=Output 파라미터는 deprecated 됨
builder = StateGraph(State, input_schema=Input, output_schema=Output)
builder.add_node('generate_sql', generate_sql)
builder.add_node('explain_sql', explain_sql)
builder.add_edge(START, 'generate_sql')
builder.add_edge('generate_sql', 'explain_sql')
builder.add_edge('explain_sql', END)

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path='graph_chain.png')

response = graph.invoke({'user_query': '각 품목의 판매량을 구해주세요.'})

print(response)