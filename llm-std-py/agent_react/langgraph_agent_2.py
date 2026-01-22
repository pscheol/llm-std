import  ast
from typing import TypedDict, Annotated
from uuid import uuid4

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.content import ToolCall
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.constants import START
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config.load_sys import llama_model

@tool
def calculator(query: str) -> str:
    '''계산기. 수식만 입력받는다.'''
    return ast.literal_eval(query)

search = DuckDuckGoSearchRun()
tools = [search, calculator]
model = ChatOllama(model=llama_model, temperature=0.1).bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]

def model_node(state: State) -> State:
    res = model.invoke(state['messages'])
    return { 'messages': [res] }


def first_model(state: State) -> State:
    query = state['messages'][-1].content
    search_tool_call = ToolCall(
        name='duckduckgo_search', args={'query':query}, id=uuid4().hex
    )
    return {'messages': AIMessage(content='', tool_calls=[search_tool_call])}

builder = StateGraph(State)
builder.add_node('first_model', first_model)
builder.add_node('model', model_node)
builder.add_node('tools', ToolNode(tools))
builder.add_edge(START, 'first_model')
builder.add_edge('first_model', 'tools')
builder.add_conditional_edges('model', tools_condition)
builder.add_edge('tools', 'model')

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path='graph_tools_2.png')


## 에이전트 실행
input = {
    'messages': [
        HumanMessage(
            '미국의 제 30대 대통령이 사망했을 때 몇 살이었나요?'
        )
    ]
}
response = graph.stream(input)

for chain in response:
    print(chain)
# output = stream_response(response, return_output=True)