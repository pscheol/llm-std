from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config.load_sys import open_ai_model

##채팅 모델 호출
model = ChatOpenAI(model=open_ai_model)
prompt = [HumanMessage('한국의 수도는 어디인가요?')]

print(model.invoke(prompt).text)


