from langchain_core.prompts import PromptTemplate
from langchain_google_genai.llms import ChatGoogleGenerativeAI
from config.load_sys import google_model

## 정적 프롬프트 생성

template = PromptTemplate.from_template('''
    아래 작성한 컨텍스트(Context)를 기반으로 질문(Question)에 대답하세요. 제공된 정보롤 대답할 수 없는 질문이라면 "모르겠어요" 라고 대답하세요.
    
    Context: {context}
    
    Question: {question}
    
    Answer: '''
)

prompt = template.invoke({
    'context': '''랭체인(LangChain)은 대규모 언어 모델(LLM)을 기반으로 하는 애플리케이션 개발을 쉽게 만들어주는 오픈소스 프레임워크로, LLM을 외부 데이터 소스와 연결하고 복잡한 다단계 작업을 수행할 수 있도록 돕는 다양한 도구와 구성 요소를 제공합니다. 이름처럼 언어 모델(Language Model)과 외부 데이터/도구를 '사슬(Chain)'처럼 연결해, 챗봇, 가상 에이전트 등 실제 사용 가능한 AI 앱을 만들 수 있게 해줍니다. ''',
    'question': '''랭체인이란?'''
})


model = ChatGoogleGenerativeAI(
    model=google_model,
    max_tokens=250
)

print(model.invoke(prompt))