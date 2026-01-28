from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

# 1. 상태 정의
class CalculatorState(TypedDict):
    total: int        # 현재까지의 총합
    history: list     # 계산 기록

# 2. 노드 함수 정의
def add_number(state: CalculatorState) -> dict:
    current_total = state["total"]
    number_to_add = 10

    new_total = current_total + number_to_add
    new_history = state["history"] + [f"{current_total} + {number_to_add} = {new_total}"]

    print(f"📊 계산: {current_total} + {number_to_add} = {new_total}")

    return {
        "total": new_total,
        "history": new_history
    }

#3. 그래프 구현
graph = StateGraph(CalculatorState)
graph.add_node("add", add_number)
graph.add_edge(START, "add")
graph.add_edge("add", END)

#4. 메모리 연결
memory = InMemorySaver()
app = graph.compile(checkpointer=memory)

#5. 실행
config = {"configurable": {"thread_id": "calculator_session"}}


print("=== 첫 번째 계산 ===")
result = app.invoke({"total": 0, "history": []}, config=config)
print(f"결과: total={result['total']}, history={result['history']}\n")


print("=== 두 번째 계산 ===")
result = app.invoke({}, config=config)   # 모든 값 생략 가능!
print(f"결과: total={result['total']}, history={result['history']}\n")


print("=== 세 번째 계산 ===")
result = app.invoke({}, config=config)
print(f"결과: total={result['total']}, history={result['history']}")
