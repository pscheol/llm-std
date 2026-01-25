

'''
ConversationEntityMemory
엔티티 메모리는 대화에서 특정 엔티티에 대한 주어진 사실을 기억합니다.

엔티티 메모리는 엔티티에 대한 정보를 추출하고(LLM 사용) 시간이 지남에 따라 해당 엔티티에 대한 지식을 축적합니다(역시 LLM 사용).

'''
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory import ConversationEntityMemory
from langchain_classic.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_ollama import ChatOllama

from config.load_sys import gemma_model

llm = ChatOllama(model = gemma_model)

# ConversationChain 을 생성합니다.
conversation = ConversationChain(
    llm=llm,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm),
)

conversation.predict(
    input="나는 LLM 배우고 있습니다."
    "LLM은 참 복잡합니다."
    "어떻게 해야 효율적으로 습득할 수 있을까요?"
)
print(conversation.memory.entity_store.store)