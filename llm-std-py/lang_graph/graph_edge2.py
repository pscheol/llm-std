from typing import Literal, TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class ControlFlowState(TypedDict):
    value: int
    path_taken: str
    result: str

def evaluate(state: ControlFlowState) -> dict:
    """값을 평가하는 노드"""
    value = state["value"]

    if value > 100:
        path = "high"
    elif value > 50:
        path = "medium"
    else:
        path = "low"

    return {
        "path_taken": path,
        "result": f"Value {value} is {path}"
    }

def handle_high(state: ControlFlowState) -> dict:
    """높은 값 처리"""
    return {"result": f"HIGH: Special handling for {state['value']}"}

def handle_medium(state: ControlFlowState) -> dict:
    """중간 값 처리"""
    return {"result": f"MEDIUM: Standard handling for {state['value']}"}

def handle_low(state: ControlFlowState) -> dict:
    """낮은 값 처리"""
    return {"result": f"LOW: Basic handling for {state['value']}"}

# 라우팅 함수
def route_by_value(state: ControlFlowState) -> Literal["high", "medium", "low"]:
    """상태에 따라 경로 결정"""
    return state["path_taken"]

# 그래프 구성
control_graph = StateGraph(ControlFlowState)
control_graph.add_node("evaluate", evaluate)
control_graph.add_node("handle_high", handle_high)
control_graph.add_node("handle_medium", handle_medium)
control_graph.add_node("handle_low", handle_low)

# 조건부 엣지로 흐름 제어
control_graph.add_edge(START, "evaluate")
control_graph.add_conditional_edges(
    "evaluate",
    route_by_value,
    {
        "high": "handle_high",
        "medium": "handle_medium",
        "low": "handle_low"
    }
)
control_graph.add_edge("handle_high", END)
control_graph.add_edge("handle_medium", END)
control_graph.add_edge("handle_low", END)


graph = control_graph.compile()
graph.get_graph().draw_mermaid_png(output_file_path='graph_edge2.png')