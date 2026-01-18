from config.load_sys import google_model
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

##채팅 모델 호출
model = ChatGoogleGenerativeAI(model=google_model)


## HumanMessage: 사용자 역할인 인간 관점으로 작성한 메시지
human_prompt = HumanMessage('한국의 수도는 어디인가요?')
## AIMessage : 어시스턴트 역할인 AI 관점으로 작성한 메시지
sys_prompt = SystemMessage('저는 친절한 어시스턴트 입니당.')
print(model.invoke([sys_prompt, human_prompt]).text)