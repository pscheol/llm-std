package com.llmstd.llmstdai.memory

import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor
import org.springframework.ai.chat.memory.ChatMemory
import org.springframework.ai.chat.memory.MessageWindowChatMemory
import org.springframework.ai.chat.memory.repository.jdbc.JdbcChatMemoryRepository
import org.springframework.core.Ordered
import org.springframework.stereotype.Service

@Service
class JdbcMemoryAIService(
    chatMemoryRepository: JdbcChatMemoryRepository,
    chatClientBuilder: ChatClient.Builder
) {
    private val chatMemory: ChatMemory = MessageWindowChatMemory.builder()
        .chatMemoryRepository(chatMemoryRepository)
        .maxMessages(20)
        .build()

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
            }
            .call()
            .content()?: ""
    }

}