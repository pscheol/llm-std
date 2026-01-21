from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages

from config.load_sys import gemma_model


# 1. 상태 그래프 생성
class State(TypedDict):
    messages: Annotated[list, add_messages]

builder = StateGraph(State)

###################################
# 2. 노드 추가
from langchain_ollama import ChatOllama

model = ChatOllama(model=gemma_model)

def chatbot(state: State):
    answer = model.invoke(state['messages'])
    return {'messages': [answer]}

## 챗봇노드 추가
## 첫 번째 인자는 고유한 노드 이름
## 두 번재 인자는 실행할 함수 or Runnable
builder.add_node('chatbot', chatbot)

####################################
# 3. 엣지 추가
builder.add_edge(START, 'chatbot')
builder.add_edge('chatbot', END)

graph = builder.compile()

####################################
# 4. 그래프 시각화 저장
graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

####################################
# 5. 그래프 실행
input = {'messages' : [HumanMessage('안녕하세요!')]}
for chunk in graph.stream(input):
    print(chunk)


####################################
# 6.체크포인터 추가
graph = builder.compile(checkpointer=MemorySaver())

# 스레드 설정
thread1 = {'configurable': {'thread_id':'1'}}

# 영속성 추가 후 그래프 실행
result_1 = graph.invoke({
    'messages': [HumanMessage('안녕하세요, 저는 조슈아 입니다.!')]
},thread1)

result_2 = graph.invoke({
    'messages': [HumanMessage('제 이름이 뭐죠?')]
},thread1)