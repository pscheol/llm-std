# key value 로 반환할때 사용하는 파서
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from config.load_sys import gemma_model

# 사용자의 질문에 대한 답변
response_schemas = [
    ResponseSchema(name="answer", description="사용자의 질문에 대한 답변"),
    ResponseSchema(
        name="source",
        description="사용자의 질문에 답하기 위해 사용된 `출처`, `웹사이트주소` 이여야 합니다.",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    # 사용자의 질문에 최대한 답변하도록 템플릿을 설정합니다.
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    # 입력 변수로 'question'을 사용합니다.
    input_variables=["question"],
    # 부분 변수로 'format_instructions'을 사용합니다.
    partial_variables={"format_instructions": format_instructions},
)


model = ChatOllama(temperature=0, model=gemma_model)
chain = prompt | model | output_parser

res = chain.invoke({"question": "대한민국의 수도는 어디인가요?"})
print(res)
'''
{ 
  'answer': '대한민국의 수도는 서울입니다.', 
  'source': 'https://ko.wikipedia.org/wiki/%EC%84%9C%EC%9A%B8'
}
'''