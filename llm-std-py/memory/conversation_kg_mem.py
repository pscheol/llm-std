'''
지식 그래프의 힘을 활용하여 정보를 저장하고 불러옴
이를 통해 모델이 서로 다른 개체 간의 관계를 이해하는 데 도움을 주고,
복잡한 연결망과 역사적 맥락을 기반으로 대응하는 능력을 향상
'''
from langchain_community.memory.kg import ConversationKGMemory
from langchain_ollama import ChatOllama

from config.load_sys import gemma_model

llm = ChatOllama(temperature=0, model=gemma_model)

memory = ConversationKGMemory(llm=llm, return_messages=True)
memory.save_context(
    {"input": "이쪽은 서울 에 거주중인 김말자씨 입니다."},
    {"output": "김셜리씨는 누구시죠?"},
)
memory.save_context(
    {"input": "김말자씨는 우리 회사의 신입 개발자입니다."},
    {"output": "만나서 반갑습니다."},
)

memory.load_memory_variables({"input": "김말자씨는 누구입니까?"})