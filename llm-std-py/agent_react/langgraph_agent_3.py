import  ast
from typing import TypedDict, Annotated
from uuid import uuid4

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.content import ToolCall
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
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
embeddings = OllamaEmbeddings(model='llama3.1:8b')
model = ChatOllama(model=llama_model, temperature=0.1)


tools_retriever = InMemoryVectorStore.from_documents(
    [Document(tool.description, metadata={'name':'tool.name'})for tool in tools]
, embeddings
).as_retriever()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]

def model_node(state: State) -> State:
    selected_tools = [tools for tool in tools if tool.name in state['selected_tools']]
    res = model.bind_tools(selected_tools).invoke(state['messages'])
    return { 'messages': [res] }


def select_tools(state: State) -> State:
    query = state['messages'][-1].content
    tool_docs = tools_retriever.invoke(query)
    return {'selected_tools': [doc.metadata['name'] for doc in tool_docs]}


builder = StateGraph(State)
builder.add_node('selected_tools', select_tools)
builder.add_node('model', model_node)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, 'selected_tools')
builder.add_edge('selected_tools', 'model')
builder.add_conditional_edges('model', tools_condition)
builder.add_edge('tools', 'model')

graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path='graph_tools_3.png')


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