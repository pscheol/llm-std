from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

## 모델 불러오기
model_id = 'klue/roberta-base'
model = AutoModel.from_pretrained(model_id)

## 분류 헤드가 포함된 모델 불러오기
# model_id = 'SamLowe/roberta-base-go_emotions'
# classification_model = AutoModelForSequenceClassification.from_pretrained(model_id)


prompt = "토크나이저는 텍스트를 토큰 단위로 나뉜다."
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenized = tokenizer(prompt)
print(tokenized)

print(tokenizer.convert_ids_to_tokens(tokenized['input_ids']))

print(tokenizer.decode(tokenized['input_ids']))

print(tokenizer.decode(tokenized['input_ids'], skip_special_tokens=True))



prompt = ["첫 번째 문장", "두 번째 문장"]
first_tokenizer_result = tokenizer(prompt)['input_ids']
print(tokenizer.batch_decode(first_tokenizer_result))

second_tokenizer_result = tokenizer(prompt)['input_ids']
print(tokenizer.batch_decode(second_tokenizer_result))

## 151
## 6장 프롬프트 엔지니어링
