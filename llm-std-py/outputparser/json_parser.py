from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from config.load_sys import gemma_model

model = ChatOllama(model=gemma_model)

class Topic(BaseModel):
    description: str = Field(description="주제에 대한 간결한 설명")
    hashtags: str = Field(description="해시태그 형식의 키워드(2개 이상)")

question = "지구 온난화의 심각성 대해 알려주세요."

# 파서를 설정하고 프롬프트 템플릿에 지시사항을 주입
parser = JsonOutputParser(pydantic_object=Topic)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 친절한 AI 어시스턴트 입니다. 질문에 간결하게 답변하세요."),
        ("user", "#Format: {format_instructions}\n\n#Question: {question}"),
    ]
)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

res = chain.invoke({"question": question})

print(res)

'''
{'description': '지구 온난화는 해수면 상승, 극심한 기상 이변, 생태계 파괴 등 심각한 영향을 미치며, 인류의 생존을 위협하는 문제입니다.', 'hashtags': 'ClimateChange #GlobalWarming #Environment'}
'''