package com.llmstd.llmstdai.prompt

import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.prompt.ChatOptions
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import java.lang.classfile.MethodBuilder

@Service
class AIDefaultMethodService(chatClientBuilder: ChatClient.Builder) {

    private val chatClient: ChatClient = chatClientBuilder
        .defaultSystem("적절한 감탄사, 웃음 등을 넣어서 친절하게 대화해 주세요.")
        .defaultOptions(ChatOptions.builder()
            .temperature(1.0)
            .maxTokens(300)
            .build()
        )
        .build()


    fun defaultMethod(question: String): Flux<String> {
        return chatClient.prompt()
            .user { question }
            .stream()
            .content()

    }
}