import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# 모델명
model_name = "microsoft/Phi-3-mini-4k-instruct"

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",
    torch_dtype="auto",
    trust_remote_code=False  # 로컬 라이브러리 구현체 사용으로 에러 방지
)
#토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 파이프라인 객체 생성
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=100,
    do_sample=False,
    use_cache=False
)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. %%timeit"

start = time.time()
for i in range(5):
    output = generator(prompt)
end = time.time()
print(f"실행 시간: {end - start}초")


# output = generator(prompt)
#
# print(output[0]['generated_text'])
# print(model)
#
# prompt = "The capital of France is"

# tokenizer.pad_token = tokenizer.eos_token
#
# prompt = "Hello, Nice to meet you."
#
# # 텍스트를 텐서로 변환하고 MPS 장치로 이동
# input_ids = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("mps")
#
# # 모델 생성 실행
# generation_output = model.generate(
#     **input_ids,
#     max_new_tokens = 20,
#     do_sample = True,
#     temperature = 0.7
# )
#
# # 전체 결과 디코딩
# print("--- Generation Result ---")
# print(tokenizer.decode(generation_output[0], skip_special_tokens=True))
#
# print("\n--- Input IDs Structure ---")
# print(input_ids)
#
#
# # 토크나이저 디코딩
# print("\n--- Individual Token Decoding ---")
# for id_v in input_ids["input_ids"][0]:
#     print(tokenizer.decode(id_v))
#
#
# print(tokenizer.decode(304))
#
#
# colors_list = [
#     '102;194;165','252;141;98','141;160;203'
# ]
#
# def show_token(sentence, tokenizer_name):
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     token_ids = tokenizer(sentence).input_ids
#     for idx, t in enumerate(token_ids):
#         print(f"\x1b[0;30;48;2;{colors_list[idx % len(colors_list)]}m {tokenizer.decode(t)}  \x1b[0m]")
#
# show_token(colors_list, model_name)