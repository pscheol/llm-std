from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
import asyncio

from config.load_sys import gemma_model

# 구성요소
template = ChatPromptTemplate.from_messages(
    [
        ('system', '너는 친절한 어시스턴트야'),
        ('human', '{question}')
    ]
)

model = ChatOllama(model=gemma_model)

## 함수로 결합
## 데코레이터 @chain을 추가해 작성한 함수에 Runnable 인터페이스르 추가
@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)

## 사용
response = chatbot.invoke({'question': '오늘 날씨는 어때?'})
print(response)

## 함수로 결합(스트림 방식)
@chain
def chatbot_streaming(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token

## 사용
for part in chatbot_streaming.stream({'question': '오늘 날씨는 어때?'}):
    print(part)

## 함수로 결합(비동기 방식)
@chain
async def chatbot_async(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)

    
async def main():
    aResponse = await chatbot_async.ainvoke({'question': '오늘 날씨는 어때?'})
    print(aResponse)

# 스크립트를 직접 실행할 때 main 함수를 실행
if __name__ == "__main__":
    # asyncio.run()을 사용하여 비동기 main 함수를 실행
    asyncio.run(main())