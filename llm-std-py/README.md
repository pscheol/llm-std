# 라이브러리 설치
```shell
uv add python-dotenv

uv add langchain  
uv add langchain-community 
uv add langchain-text-splitters 
uv add langchain-postgres

## OpenAI, gemini, ollama
uv add langchain-openai
uv add langchain-google-genai
uv add langchain-ollama


uv add scikit-learn
uv add torch torchvision torchaudio
uv add numpy 
uv add tqdm 
uv add transformers datasets accelerate
```

# 제미나이 OpenAIsms 비용이 발생함으로 Ollama나 Huggingface 사용
# # ollama 설치
https://ollama.com/ 

## ollama 다운로드 및 실행
```
# 1. Ollama에서 모델 다운로드
ollama pull gemma3:12b

ollama run gemma3:12b
```
## Ollama 라이브러리 사용
```shell
uv add ollama
```

## Huggingface 라이브러리 사용
```shell
uv add transformers
```

## 사용 모델
```
gemma3:12b : 구글이 개발한 멀티모달(텍스트, 이미지) 대규모 언어 모델(LLM) 제품군 중 하나로, 120억(12 Billion) 개의 매개변수를 가진 모델
https://huggingface.co/google/gemma-3-12b-it
```