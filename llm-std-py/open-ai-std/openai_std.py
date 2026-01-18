import openai

client = openai.OpenAI(api_key="API KEY")

def chatgpt_generation(prompt, document, model="gpt-4o-mini"):
    """프롬프트와 문서를 입력받아 출력을 생성"""
    messages = [
        {
            "role": "system",
            "content":"You are a helpful assistant."
        },
        {
            "role":"user",
            "content": prompt.replace("[DOCUMENT]", document)
        }
    ]

    chat_completion = client.chat.completions.create(
        messages = messages,
        model = model,
        temperature = 0
    )
    return chat_completion.choices[0].message.content


prompt = """
    Predict whether the following document is a positive or negative movie review:
    
    [DOCUMENT]
    
    If it is positive return 1 and if it is negative return 0. Do not give any other answers.
"""

document = "unpretentious , charming , quirky , original"
print(chatgpt_generation(prompt, document))