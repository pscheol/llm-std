from langchain_classic.chains.llm import LLMChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.chat_models import ChatOllama

from config.load_sys import gemma_model


# 1. 메모리 초기화 (return_messages=True로 설정하여 메시지 객체 리스트를 반환)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 2. 이전에 저장된 대화 내용 추가
memory.save_context(
    inputs={
        "human": "안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
    },
    outputs={
        "ai": "안녕하세요! 계좌 개설을 원하신다니 기쁩니다. 먼저, 본인 인증을 위해 신분증을 준비해 주시겠어요?"
    },
)

# 3. 프롬프트 설정 (MessagesPlaceholder를 사용하여 대화 기록이 들어갈 위치 지정)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# 4. LLMChain을 사용하여 메모리와 LLM 연결
llm = ChatOllama(model=gemma_model)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

# 5. 체인 실행 (메모리에 저장된 대화 내용을 바탕으로 답변 생성)
response = chain.invoke({"question": "네, 신분증 준비했습니다. 다음 단계는 무엇인가요?"})
print(response['text'])