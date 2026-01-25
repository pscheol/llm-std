from typing import TypedDict, Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from operator import add

from lang_graph.data_process_node import DataProcessorNode


## State(상태) - 데이터 저장소
# 모든 노드가 공유하는 데이터 저장소
class SimpleState(TypedDict):
    ## 단순 값들 덮어쓰기 방식
    count: int
    name: str
    ## 누적되는 값들 - 추가방식
    logs: Annotated[list[str], add]

## Node(노드) - 작업하는 함수
## 실젝 작업을 수행하는 함수 현재 상태를 받아 새로운 상태로 돌려줌

# 카운터를 1 증가시키는 노드
def add_one(state: SimpleState):
    return { 'count' : state["count"] + 1, 'logs': ['카운터 1 증가']}

## 이름을 설정하는 노드
def set_name(state: SimpleState):
    return {'name': 'Joshua', 'logs': ['이름을 설정']}

graph_builder = StateGraph(SimpleState)
## 노드를 추가
graph_builder.add_node('add_one', add_one)
graph_builder.add_node('set_name', set_name)


## Edge(엣지) - 연결하는 화살표
## 노드들의 사이를 연결을 정의

# 시작과 끝 연결
graph_builder.add_edge(START, 'add_one')
graph_builder.add_edge('set_name', END)

# 기본적인 엣지 연결
graph_builder.add_edge('add_one', 'set_name')

## 실행
graph = graph_builder.compile()
## 그래프 이미지 출력
graph.get_graph().draw_mermaid_png(output_file_path='graph_default.png')

result = graph.invoke({'count': 0})

## 결과 {'count': 1, 'name': 'Joshua'}
print(result)


