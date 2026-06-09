package com.llmstd.llmstdai.advisor

import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.client.advisor.SafeGuardAdvisor
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux

@Service
class AdvisorService1(
    chatClientBuilder: ChatClient.Builder
) {
    private val chatClient: ChatClient = chatClientBuilder
        .defaultAdvisors(
            SimpleLoggerAdvisor(),
            SafeGuardAdvisor(
                listOf("욕설", "계좌번호", "폭력", "민가한 콘텐츠 질문")
            ),
            AdvisorA(),
            AdvisorB())
        .build()




    fun advisorChain1(question: String): String {
        val response: String? = chatClient.prompt(question)
            .advisors { AdvisorC() }
            .call()
            .content()

        return response ?: ""
    }

    fun advisorChain2(question: String): Flux<String> {
        val response: Flux<String> = chatClient.prompt(question)
            .advisors { AdvisorC() }
            .stream()
            .content()

        return response
    }
}