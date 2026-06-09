package com.llmstd.llmstdai.advisor

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.springframework.ai.chat.client.ChatClient
import org.springframework.core.Ordered
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux

@Service
class AdvisorService2(
    chatClientBuilder: ChatClient.Builder
) {
    companion object {
        private val log = LoggerFactory.getLogger(AdvisorService2::class.java)
    }
    private val chatClient: ChatClient = chatClientBuilder
        .defaultAdvisors(MaxCharLengthAdvisor(Ordered.HIGHEST_PRECEDENCE))
        .build()




    fun advisorContext(question: String): String {
        log.info("Processing advisor context for question: $question")
        val response: String? = chatClient.prompt()
            .advisors { advisorSpec ->
                advisorSpec.param(MaxCharLengthAdvisor.MAX_CHAR_LENGTH, 100)
            }
            .user { question }
            .call()
            .content()

        return response ?: ""
    }
}