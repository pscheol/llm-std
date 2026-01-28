from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import Annotated, TypedDict
from operator import add

class State(TypedDict):
    topics: list[str]
    summaries: Annotated[list[str], add]

def generate_topics(state: State):
    return {"topics": ["AI", "ML", "LLM"]}

def summarize_topic(state: dict) -> dict:
    # 개별 토픽 처리
    topic = state["topic"]
    return {"summaries": [f"{topic} 요약"]}

def fan_out(state: State) -> list[Send]:
    # 각 토픽에 대해 summarize_topic 노드 호출
    return [
        Send("summarize", {"topic": topic})
        for topic in state["topics"]
    ]

def aggregate(state: State):
    # 모든 요약 결과가 자동으로 수집됨
    return {"final": f"총 {len(state['summaries'])}개 요약 완료"}

graph = StateGraph(State)
graph.add_node('generate', generate_topics)
graph.add_node("summarize", summarize_topic)
graph.add_node("aggregate", aggregate)

graph.add_edge(START, "generate")
graph.add_conditional_edges("generate", fan_out)
graph.add_edge("summarize", "aggregate")
graph.add_edge("aggregate", END)

g = graph.compile()
g.get_graph().draw_mermaid_png(output_file_path='map_reduce.png')
