from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config.load_sys import gemma_model

prompt = ChatPromptTemplate.from_messages([
    ('system','당신은 친절한 어시스턴트 입니다. 모든 질문에 최선을 다해 답하세요.'),
    ('placeholder', '{message}')
])

model = ChatOllama(model=gemma_model)

chain = prompt | model

response = chain.invoke({
    'message': [
        ('human','다음 한국어 문장을 프랑스어로 번역하세요. : 나는 프로그래밍을 좋아해요.'),
        ('ai', 'J\'adore programmer.'),
        ('human', '뭐라고 말했지?')
    ]
})

print(response.content)