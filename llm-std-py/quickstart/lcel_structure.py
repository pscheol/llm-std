from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config.load_sys import gemma_model
import asyncio

template = ChatPromptTemplate.from_messages(
    [
        ('system', '너는 친절한 어시스턴트야.'),
        ('human','{question}')
    ]
)

model = ChatOllama(model=gemma_model)

# 연산자 | 로 결합
chatbot = template | model
print("\n################# invoke() ####################\n")
# 사용
response = chatbot.invoke({'question': '너는 어떤 모델이니?'})
print(response)

print("\n################# stream() ########################\n")
## 스트림 방식 사용
chatbot = template | model
response = chatbot.stream({'question': '너는 어떤 모델이니?'})
for token in response:
    print(token)

## 비동기 방식
print("\n################# ainvoke() ########################\n")
chatbot = template | model
async def main():
    response = await chatbot.ainvoke({'question': '너는 어떤 모델이니?'})
    print(response)


if __name__ == "__main__":
    asyncio.run(main())

