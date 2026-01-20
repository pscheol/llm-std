from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_ollama import ChatOllama

from config.load_sys import gemma_model

model = ChatOllama(model=gemma_model)

chain1 = (
    PromptTemplate.from_template("{country} 의 수도는 어디야?")
    | model
    | StrOutputParser()
)


chain2 = (
    PromptTemplate.from_template("{country} 의 면적은 얼마야?")
    | model
    | StrOutputParser()
)


combined = RunnableParallel(capital=chain1, area=chain2)

print(combined.invoke({"country": "대한민국"}))

print(combined.batch([{"country": "대한민국"}, {"country": "일본"}]))