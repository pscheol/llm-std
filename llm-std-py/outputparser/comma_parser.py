from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from config.load_sys import gemma_model

## 콤마 파서
output_parser = CommaSeparatedListOutputParser()


# 출력 형식 지침 가져오기
format_instructions = output_parser.get_format_instructions()
# 프롬프트 템플릿 설정
prompt = PromptTemplate(
    # 주제에 대한 다섯 가지를 나열하라는 템플릿
    template="주제에 대한 다섯 가지 리스트 {subject}.\n{format_instructions}",
    input_variables=["subject"],  # 입력 변수로 'subject' 사용
    partial_variables={"format_instructions": format_instructions},
)


model = ChatOllama(model=gemma_model, temperature=0)


chain = prompt | model | output_parser

for s in chain.stream({"subject": "대한민국 관광명소"}):
    print(s)  # 스트림의 내용을 출력합니다.

#['경복궁']
#['제주도']
#['남산타워']
#['부산해운대']
#['설악산']