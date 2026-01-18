from langchain_openai.llms import OpenAI
from config.load_sys import open_ai_model

## 기본 LLM 모델 호출
model = OpenAI(
    model=open_ai_model,
    temperature=0.5,
)

print(model.invoke('안녕하세요'))