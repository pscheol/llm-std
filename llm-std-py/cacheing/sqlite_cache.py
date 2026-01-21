from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
import os

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from config.load_sys import gemma_model

# 캐시 디렉토리를 생성합니다.
if not os.path.exists("cache"):
    os.makedirs("cache")

# SQLiteCache를 사용합니다.
set_llm_cache(SQLiteCache(database_path="cache/llm_cache.db"))

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
response = chain.invoke({"country": "한국"})
print(response.content)
response = chain.invoke({"country": "한국"})
print(response.content)
response = chain.invoke({"country": "한국"})
print(response.content)
response = chain.invoke({"country": "한국"})
print(response.usage_metadata)