# 메시지 축약
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    trim_messages
)
from langchain_ollama import ChatOllama
from config.load_sys import gemma_model

#샘플 메시지 설정
messages = [
    SystemMessage(content='당신은 친절한 어시스턴트 입니다.'),
    HumanMessage(content='안녕하세요. 나는 조슈아 입니다.'),
    AIMessage(content='안녕하세요?'),
    HumanMessage(content='치킨을 좋아합니다.'),
    AIMessage(content='좋구만요?'),
    HumanMessage(content='2 + 2는 얼마죠?'),
    AIMessage(content='4 입니다.'),
    HumanMessage(content='고마워요'),
    AIMessage(content='천만에요'),
    HumanMessage(content='즐거운가요?'),
    AIMessage(content='네')
]

# 축약 설정
trimmer = trim_messages(
    max_tokens=65,
    strategy='last',
    token_counter=ChatOllama(model=gemma_model),
    include_system=True,
    allow_partial=False,
    start_on='human'
)

## 축약 적용
trimmer = trimmer.invoke(messages)
print(trimmer)
