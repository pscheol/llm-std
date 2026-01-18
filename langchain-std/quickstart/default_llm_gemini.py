
from langchain_google_genai.llms import ChatGoogleGenerativeAI
from config.load_sys import google_model
## 기본 LLM 모델 호출

## .env GOOGLE_API_KEY를 통해 자동으로 API 키 인식
model = ChatGoogleGenerativeAI(
    model=google_model,
    temperature =0.5,
    max_tokens=100
)

print(model.invoke('안녕하세요').text)