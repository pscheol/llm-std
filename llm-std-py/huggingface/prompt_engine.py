import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

## 모델과 토크나이저 로드
model_name = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",
    torch_dtype="auto",
    trust_remote_code=False  # 로컬 라이브러리 구현체 사용으로 에러 방지
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

## 파이프라인 생성
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False,
    use_cache=True
)


##prompt
messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]

## 출력 생성
output = pipe(messages)
print(output[0]["generated_text"])


prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
print(prompt)

## temperature 매개변는 모델이 랜덤하게 토큰을 선택하게 되어 확률적으로 동작하게 된다.
## 예측한 확률 분포의 집중도를 조절. 수학적으로 소프트맥스 함수에 적용되는 분모 값
## 낮은 값(0.1~0.5) : 가장 확률이 높은 단어에 가중치를 훨씬 더 많이 만든다.
#### 결과가 매우 보수적이고 일관적이며, 같은 질문에 비슷한 답변을 할 확률이 높다.
#### 수학 문제 풀이, 코딩, 사실 기반 답변에 유리
## 높은 값(0.8~1.5): 낮은 확률의 단어들도 선택될 기회가 생긴다.
#### 결과가 창의적이고 다양해지지만, 때로는 앞뒤가 맞지 않는 말(환각 현상)을 할 가능성이 커진다
#### 소설 쓰기, 시 짓기, 아이디어 브레인스토밍에 유리.
output = pipe(messages, do_sample=True, temperature=1)
print(output[0]["generated_text"])

##  Why don't chickens use computers? Because they're afraid of crows and the QWERTY-UI-POOP!
##  Why was the math book sad at the chicken farm?


## top_p (핵심 생플링)
## 단어 집합 중에서 누적 확률이 p 이내인 상위 후보들만 남기고 나머지는 제외하는 방식
## 작동방식 : 확률이 높은 순서대로 단어들을 나열 후 그 확률의 합이 설정 값 p가 될 때까지 단어들만 후보군으로 유지
#### 동적필터링 : 모델이 다음에 올 단어를 확실 할 때는 후보군이 적어지고, 불확실할 때는 후보군이 자동으로 넓어진다.
#### Top_k와의 차이: top_K가 무조건 k개의 단어만 보는 것과 달리, top_p는 맥락에 따라 선택 범위를 유연하게 조절
#### 0.0~1.0 (ex: 0.9-> 90%)
output = pipe(messages, do_sample=True, top_p=1)
print(output[0]["generated_text"])

one_shot_prompt = [
    {
        "role" : "user",
        "content": "A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:"
    },
    {
        "role": "assistant",
        "content":"I have a Gigamuru that my uncle gave me as a gift. Ilove to play it at home"
    },
    {
        "role":"user",
        "content":"To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is:"
    }
]

print(tokenizer.apply_chat_template(one_shot_prompt, tokenize=False))

print("+3=")
outputs = pipe(one_shot_prompt)
print(outputs[0]["generated_text"])