import operator
from typing import Annotated, Sequence, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import (BaseMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.tools import tool
from langgraph.graph import StateGraph

from config.load_sys import gemma_model, llama_model


## State Type 정의
class AgentState(TypedDict):
    order: dict
    messages: Annotated[Sequence[BaseMessage], operator.add]

## 주문 취소 도구 정의
@tool
def cancel_order(order_id: str) -> str:
    '''배송되지 않은 주문 취소'''
    ##TODO 취소 백엔드 API 요청
    return f'주문 {order_id} 가 취소되었습니다.'


## 에이전트 구조 정의 [ LLM호출, 도구실행, 다시 LLM 호출]
def call_model(state: AgentState):
    msgs = state['messages']
    order = state.get('order', {'order_id': 'UNKNOWN'})

    # LLM 초기화
    llm = init_chat_model(model=llama_model, model_provider='ollama', temperature=0.0)
    llm_with_tools = llm.bind_tools([cancel_order]) ## 도구 바인딩

    ## 시스템 프롬프트에 모델이 할일을 명시
    prompt = (
        f'''
            당신은 훌륭한 이커머스 지원 에이전트 입니다.
            주문 ID : {order['order_id']}
            고객이 취소 요청을 하면 cancel_order(order_id)를 호출하고
            간단한 확인 메시지를 보내주세요.
            그렇지 않으면 일반적으로 응답해주세요.
        '''
    )
    full = [SystemMessage(content=prompt)] + msgs

    ## LLM 호출: 도구 호출 여부 결정
    first = llm_with_tools.invoke(full)
    out = [first]

    if getattr(first, 'tool_calls', None):
        ## 도구 실행
        tc = first.tool_calls[0]
        result = cancel_order.invoke(tc['args'])
        out.append(ToolMessage(content=result, tool_call_id=tc['id']))

        ## 최종 확인 컨텍스트 생성
        second = llm.invoke(full + out)
        out.append(second)

    return {'messages': out}


def construct_graph():
    g = StateGraph(AgentState)
    g.add_node('assistant', call_model)
    g.set_entry_point('assistant')
    return g.compile()


graph = construct_graph()


if __name__ == '__main__':
    order = {'order_id':'B123242'}
    convo = [HumanMessage(content='주문 #B123242를 취소해주세요.')]
    result = graph.invoke({'order':order, 'messages':convo})
    for msg in result['messages']:
        print(f'{msg.type}: {msg.content}')