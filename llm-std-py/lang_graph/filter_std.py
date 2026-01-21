# 메시지 축약
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    filter_messages
)

#샘플 메시지 설정
messages = [
    SystemMessage(content='당신은 친절한 어시스턴트 입니다.', id='1'),
    HumanMessage(content='안녕하세요. 나는 조슈아 입니다.', id='2', name='user'),
    AIMessage(content='예시 입력', i='3', name='assistant'),
    HumanMessage(content='실제 입력', id='4', name='bob'),
    AIMessage(content='실제 출력', id='5', name='alice'),
]

## 사용자 메시지만 필터링
human_messages = filter_messages(messages, include_types='human')

print("include_types\n", human_messages)

# 특정 이름 메시지 제외
exclude_messages = filter_messages(messages, exclude_names=['bob', 'assistant'])
print("exclude_names\n", exclude_messages)

# 유형과 ID로 필터링
filtered_messages = filter_messages(messages, include_types=['human','ai'], exclude_ids=['3'])
print("filtered_messages\n", filtered_messages)
