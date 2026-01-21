from enum import Enum

from langchain_classic.output_parsers import EnumOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from config.load_sys import gemma_model


class Color(Enum):
    RED = '빨간색'
    GREEN = '초록색'
    BLUE = '파란색'

llm = ChatOllama(model=gemma_model)

parser = EnumOutputParser(enum=Color)

prompt = PromptTemplate.from_template(
    """
    다음의 물체는 어떤 색깔인가요?
    Object: {object}
    Instructions: {instructions}
    """
).partial(instructions=parser.get_format_instructions())


chain = prompt | llm | parser

question = {'object':'하늘은 무슥색이야'}
print(chain.invoke(question))