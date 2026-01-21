from langchain_core.messages import HumanMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from torch.backends.opt_einsum import strategy

from config.load_sys import gemma_model
from lang_graph.create_state_graph import builder

graph = builder.compile(checkpointer=MemorySaver())

# 스레드 설정
thread1 = {'configurable': {'thread_id':'1'}}

# 영속성 추가 후 그래프 실행
result_1 = graph.invoke({
    'messages': [HumanMessage('안녕하세요, 저는 조슈아 입니다.!')]
},thread1)

print(result_1)
result_2 = graph.invoke({
    'messages': [HumanMessage('제 이름이 뭐죠?')]
},thread1)

print(result_2)

## 상태확인
state = graph.get_state(thread1)
print(state)

## 상태 업데이트
update_state = graph.update_state(thread1,{'messages': [HumanMessage('LLM은 복잡하넹')]})
print(update_state)

## 다시 메시지 출력
result_3 = graph.invoke({
    'messages': [HumanMessage('너는 무슨 모델이니')]
},thread1)

print(result_3)

##################################
