from langchain_core.prompts import PromptTemplate

## 동적 프롬프트 생성

template = PromptTemplate.from_template('''
    아래 작성한 컨텍스트(Context)를 기반으로 질문(Question)에 대답하세요. 제공된 정보롤 대답할 수 없는 질문이라면 "모르겠어요" 라고 대답하세요.
    
    Context: {context}
    
    Question: {question}
    
    Answer: '''
)

response = template.invoke({
    'context': '''대한민국의 수도는 서울 입니다.''',
    'question': '''대한민국의 수도는 어디 입니까?'''

})
print(response)

m