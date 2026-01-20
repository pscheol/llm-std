from datetime import datetime

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain_ollama import ChatOllama

from config.load_sys import gemma_model

llm = ChatOllama(model=gemma_model)

# 템플릿 정의
template = "{country}의 수도는 어디인가요?"

#from_template() 메서드를 사용하여 PromptTemplate 객체 생성
prompt = PromptTemplate.from_template(template)

# country 변수에 값 대입
prompt.format(country='대한민국')

#chain 생성
chain = prompt | llm

#생성
print(chain.invoke({'country':'대한민국'}))


template = "{country}와 {country2}의 수도는 어디인가요?"

prompt_partial = PromptTemplate(
    template = template,
    input_variables = ["country"],
    partial_variables={
        "country2": "미국"
    },
)

# country 변수에 값 대입
prompt_partial.format(country='대한민국')

#chain 생성
chain_partial = prompt_partial | llm

#생성
print(chain_partial.invoke({'country':'대한민국'}))




def get_today():
    return datetime.now().strftime("%B %d")
prompt_today = PromptTemplate(
    template="오늘의 날짜는 {today} 입니다. 오늘이 생일인 유명인 {n}명을 나열해 주세요. 생년월일을 표기해주세요.",
    input_variables=["n"],
    partial_variables={
        "today": get_today  # dictionary 형태로 partial_variables를 전달
    },
)
prompt_today.format(n=3)
chain_today = prompt_today | llm

print(chain_today.invoke({'today':'Jan 02', 'n':3}))

chat_prompt = ChatPromptTemplate.from_template("{country}의 수도는 어디인가요?")
chat_prompt.format(country='대한민국')

chain_chat = chat_prompt | llm

print(chain_chat.invoke({'country':'대한민국'}))


chat_template = ChatPromptTemplate.from_messages(
    [
        # role, message
        ("system", "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 {name} 입니다."),
        ("human", "반가워요!"),
        ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(
    name="테디", user_input="당신의 이름은 무엇입니까?"
)



## 각 예시를 어떻게 포맷팅할지 정의 (개별 예시 템플릿)
example_formatter = PromptTemplate(
    input_variables=["word", "emoji"],
    template="단어: {word}\n이모지: {emoji}"
)

# AI에게 보여줄 예시 데이터 (Few-shot Examples)
examples = [
    {"word": "행복", "emoji": "😄"},
    {"word": "슬픔", "emoji": "😭"},
    {"word": "사랑", "emoji": "❤️"}
]
## FewShotPromptTemplate 생성
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,                # 예시 데이터 리스트
    example_prompt=example_formatter, # 예시를 보여줄 형식
    prefix="다음 단어에 알맞은 이모지를 하나만 출력하세요.", # 지시문
    suffix="단어: {input}\n이모지:",    # 사용자가 입력할 부분
    input_variables=["input"],        # 사용자 입력 변수명
    example_separator="\n\n"          # 예시 사이의 구분자
)

# 결과 확인
print(few_shot_prompt.format(input="분노"))