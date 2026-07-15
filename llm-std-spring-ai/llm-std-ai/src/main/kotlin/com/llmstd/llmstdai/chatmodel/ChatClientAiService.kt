package com.llmstd.llmstdai.chatmodel

import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.prompt.ChatOptions
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux

@Service
class ChatClientAiService(
    private val chatClient: ChatClient
) {

    fun generateText(question: String) : String {
        return chatClient.prompt()
            .system { "사용자 질문에 대해 한국어로 답변을 해야 합니다." }
            .user { question }
            .options(
                ChatOptions.builder()
                    .temperature(0.3)
                    .maxTokens(1000)
            )
            .call()
            .content() ?: ""
    }

    fun generateTextByStream(question: String): Flux<String> {
        return chatClient.prompt()
            .system { "사용자 질문에 대해 한국어로 답변을 해야 합니다." }
            .user { question }
            .options(
                ChatOptions.builder()
                    .temperature(0.3)
                    .maxTokens(1000)
            )
            .stream()
            .content()
    }
}