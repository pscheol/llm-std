from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from config.load_sys import gemma_model

# 인메모리 캐시를 사용하여 동일 질문에 대한 답변을 저장하고, 캐시에 저장된 답변을 반환
# 인메모리 캐시 사용
set_llm_cache(InMemoryCache())


# 모델을 생성합니다.
llm = ChatOllama(model=gemma_model)

# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template("{country} 에 대해서 200자 내외로 요약해줘")

# 체인을 생성합니다.
chain = prompt | llm

response = chain.invoke({"country": "한국"})
print(response.content)

response = chain.invoke({"country": "한국"})
print(response.content)