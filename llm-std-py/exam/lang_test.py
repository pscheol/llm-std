from langchain import LlamaCpp
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

llm = LlamaCpp(
    model_path="/ai/model/Phi-3-mini-4k-instruct-fp16.gguf",
    chat_format="phi-3",
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=4096,
    seed=42,
    verbose=False,
)

# --- 첫 번째 체인 ---
template = """
<|user|>
Current conversation:{chat_history}
{input_prompt}
<|end|>
<|assistant|>
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt", "chat_history"]
)
# basic_chain = prompt | llm
# response = basic_chain.invoke(
#     {"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"}
# )
# print("첫 번째 응답:", response)
#
# response2 = basic_chain.invoke({"input_prompt":"What is my name?"})
# print("두 번째 응답 : ", response2)

### 메모리 적용
## 사용할 메모리를 정의
memory = ConversationBufferMemory(memory_key="chat_history")

## 마지막 두 개의 대화만 유지하도록 캐싱
memoryBuffer = ConversationBufferWindowMemory(k=2, memory_key="chat_history")

##LLM, 프롬프트, 메모리를 연결
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memoryBuffer)
responseChain = llm_chain.invoke({"input_prompt":"Hi! My name is Maarten. With is 1 + 1"})
print("메모리 첫 번째 응답 : ", responseChain)
responseChain2 = llm_chain.invoke({"input_prompt":"Wh is 3 + 3"})
print("메모리 두 번째 응답 : ", responseChain2)
responseChain3 = llm_chain.invoke({"input_prompt":"What is my name?"})
print("메모리 세 번째 응답 : ", responseChain3)
responseChain4 = llm_chain.invoke({"input_prompt":"What is my age?"})
print("메모리 네 번째 응답 : ", responseChain4)

# --- 두 번째 체인 (회사 이름 생성) ---
template = "Create a funny name for a business that sells {product}."
prompt = PromptTemplate(
    template=template,
    input_variables=["product"]
)
company_name_chain = prompt | llm
response2 = company_name_chain.invoke({"product": "socks"})
print("회사 이름 생성:", response2)


# --- 스토리 제목 체인 1 ---
template = """
<|user|>
Create a title for a story about {summary}. Only return the title.<|end|>
<|assistant|>
"""
title_prompt = PromptTemplate(template=template, input_variables=["summary"])
title = LLMChain(llm=llm, prompt=title_prompt, output_key="title")

## --- 캐릭터 설명 체인 2 ---
template = """
Describe the main character of a story about {summary} with the title {title}. Use only two sentences.<|end|>
<|assistant|>
"""
character_prompt = PromptTemplate(
    template=template, input_variables=["summary", "title"]
)

character = LLMChain(llm=llm, prompt=character_prompt, output_key="character")

## --- 요약, 제목, 캐릭터 설명 야이가 체인 ---
template = """<|user|>
Create a story about {summary} with the title {title}. The main character is {character}.
Only return the story and it cannot be longer then one paragraph. <|end|>
<|assistant|>
"""

story_prompt = PromptTemplate(
    template=template, input_variables=["summary", "title", "character"]
)
story = LLMChain(llm=llm, prompt=story_prompt, output_key="story")

llm_chain = title | character | story
response = llm_chain.invoke("a girl that lost her mother")

print(f"스토리 체인 응답 : {response}")


# 요약 프롬프트 템플릿 생성
from langchain.memory import ConversationSummaryMemory
summary_prompt_template = """
<|user|>Summarize the conversation and update with the new lines.

Current summary:
{summary}

new line of conversation:
{new_lines}

New Summary:<|end|>
<|assistant|>
"""
summary_prompt = PromptTemplate(
    input_variables=["new_lines", "summary"],
    template=summary_prompt_template
)

# 메모리 정의
memorySummary = ConversationSummaryMemory(llm=llm, prompt=summary_prompt, memory_key="chat_history")
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memorySummary,
)
llm_chain.invoke({"input_prompt":"Hi! My name is Maarten. What is 1 + 1?"})
llm_chain.invoke({"input_prompt":"What is my name?"})

res = llm_chain.invoke({"input_prompt":"What was the first question I asked?"})
print(f"ConversationSummaryMemory 체인 응답 : {res}")