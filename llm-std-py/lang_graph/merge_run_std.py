# 메시지 축약
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    filter_messages, merge_message_runs
)

#샘플 메시지 설정
messages = [
    SystemMessage(content='당신은 친절한 어시스턴트 입니다.'),
    SystemMessage(content='항상 농담으로 대답하세요.'),
    HumanMessage(
        content=[{'type': 'text', 'text': '어떤 피자가 제일 맛있어요?'}]
    ),
    HumanMessage(
        content='어떤 햄버거가 가장 맛있나요?'
    ),
    AIMessage(
        content='나는 항상 너만 "고르곤졸라"'
    ),
    AIMessage(
        content='너가 "버거" 싶어'
    )
]

# 연속된 메시지 병합
merged = merge_message_runs(messages)

print("merged\n", merged)