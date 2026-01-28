from typing import TypedDict, Literal
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from sympy.codegen.cnodes import goto


class State(TypedDict):
    score: float

graph = StateGraph(State)

#일반 엣지(Normal Edge)
# 고정된 경로로 이동
graph.add_edge('node_a', 'node_b')

# 조건부 엣지(Conditional Edge)
# 조건에 따라 다른 로드로 라우팅
def route_func(state: State) -> str:
    if (state['score'] > 0.8):
        return 'high_quality'
    else:
        return 'low_quality'

graph.add_conditional_edges(
    'evaluate', route_func, {
        "high_quality": "publish",
        "low_quality": "revise"
    }
)

# 조건부 진입점
# 처음 시작부터 조건 분기가 필요할 때
def entry_router(state: State) -> str:
    if state["task_type"] == "search":
        return "search_node"
    return "default_node"

graph.add_conditional_edges(START, entry_router)

# Command - 상태 업데이트 + 라우팅
# Command 객체를 사용하여 상태 업데이트 및 라우팅을 동시 처리

def m_node(state: State) -> Command[Literal["next_node_a", "next_node_b"]]:
    # 조건에 따라 다른 노드로 라우팅 하면서 상태 업데이트
    if state['condition']:
        return Command(
            update={'processed': True, 'route':'a'},
            goto='next_node_a'
        )
    else:
        return Command(
            update={'processed': True, 'route':'b'},
            goto='next_node_b'
        )

def subgraph_node(state: State) -> Command[Literal["parent_node"]]:
    return Command(
        update={"result": "done"},
        goto="parent_node",
        graph=Command.PARENT  # 부모 그래프로 네비게이션
    )
