package com.llmstd.llmstdai.memory

import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor
import org.springframework.ai.chat.memory.ChatMemory
import org.springframework.core.Ordered
import org.springframework.stereotype.Service

@Service
class InMemoryAIService(
    chatMemory: ChatMemory,
    chatClientBuilder: ChatClient.Builder
) {

    private val chatClient: ChatClient = chatClientBuilder
        .defaultAdvisors(
            MessageChatMemoryAdvisor.builder(chatMemory).build(),
            SimpleLoggerAdvisor(Ordered.LOWEST_PRECEDENCE - 1)
        )
        .build()

    fun chat(userText: String, conversationId: String): String {
        return chatClient.prompt()
            .user(userText)
            .advisors { spec -> spec.param(
                ChatMemory.CONVERSATION_ID, conversationId
            )
            }.call()
            .content()?: ""
    }
}