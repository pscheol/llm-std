from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


# State: 숫자와 결과 저장
class NumberState(TypedDict):
    number: int
    result: str

# Node 1: 큰 숫자 처리
def handle_big_number(state: NumberState):
    return {"result": f"{state['number']}는 큰 숫자입니다." }

def handle_small_number(state: NumberState):
    return {"result": f"{state['number']}는 작은 숫자입니다." }


def check_size(state: NumberState):
    if state['number'] > 10:
        return 'big'
    else:
        return 'small'

number_graph = StateGraph(NumberState)
number_graph.add_node('big_handler', handle_big_number)
number_graph.add_node('small_handler', handle_small_number)

number_graph.add_conditional_edges(
    START,
    check_size,
    {
        'big': 'big_handler',
        'small': 'small_handler'
    }
)
number_graph.add_edge("big_handler", END)
number_graph.add_edge("small_handler", END)

graph = number_graph.compile()

graph.get_graph().draw_mermaid_png(output_file_path='graph_number_compare.png')


result1 = graph.invoke({"number": 15, "result": ""})
print(result1)


result2 = graph.invoke({"number": 5, "result": ""})
print(result2)
