from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, add_messages

from config.load_sys import gemma_model


class State(TypedDict):
    ## 메시지 유형 : list
    ## add_messages func : 상태를 업데이트하는 방법
    #### 이전 메시지를 대체하는 대신 새 메시지를 추가
    messages: Annotated[list, add_messages]

builder = StateGraph(State)

model = ChatOllama(model=gemma_model)

def chatbot(state: State):
    answer = model.invoke(state['messages'])
    return {'messages': [answer]}


## 챗봇 노드 추가
## 첫 번째 인자 : 고유한 노드 이름
## 두 번재 인자 : 실행할 함수 또는 Runnable
builder.add_node('chatbot', chatbot)

## 엣지 추가
builder.add_edge(START, 'chatbot')
builder.add_edge('chatbot', END)

graph = builder.compile()

## 시각화 저장
graph.get_graph().draw_mermaid_png(output_file_path='graph1.png')

## 스트림을 활용하여 그래프 실행
input = {'messages' : [HumanMessage('안녕하세요.')]}
for chunk in graph.stream(input):
    print(chunk)

