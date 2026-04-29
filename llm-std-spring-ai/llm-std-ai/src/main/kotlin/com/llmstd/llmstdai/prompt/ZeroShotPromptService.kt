package com.llmstd.llmstdai.prompt

import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.model.ChatModel
import org.springframework.ai.chat.prompt.ChatOptions
import org.springframework.ai.chat.prompt.Prompt
import org.springframework.ai.chat.prompt.PromptTemplate
import org.springframework.ai.ollama.OllamaChatModel
import org.springframework.ai.ollama.api.OllamaChatOptions
import org.springframework.stereotype.Service

@Service
class ZeroShotPromptService(chatClientBuilder: ChatClient.Builder) {

    private val chatClient: ChatClient = chatClientBuilder
        .defaultOptions(ChatOptions.builder()
            .temperature(0.0)
            .maxTokens(4)
            .build()
        )
        .build()

    private val promptTemplate = PromptTemplate.builder()
        .template("""
            영화 리뷰를 [긍정적, 중립적, 부정적] 중에서 하나로 분류하세요.
            레이블만 반환하세요.
            리뷰: {review}
        """.trimIndent())
        .build()

    fun zeroShotPrompt(review: String): String? {
        return chatClient.prompt()
            .user(promptTemplate.render(mapOf("review" to review)))
            .call()
            .content()
    }

}