package com.llmstd.llmstdai.advisor

import org.slf4j.LoggerFactory
import org.springframework.ai.chat.client.ChatClientRequest
import org.springframework.ai.chat.client.ChatClientResponse
import org.springframework.ai.chat.client.advisor.api.CallAdvisor
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain
import org.springframework.ai.chat.client.advisor.api.StreamAdvisor
import org.springframework.ai.chat.client.advisor.api.StreamAdvisorChain
import org.springframework.core.Ordered
import reactor.core.publisher.Flux

class AdvisorA : CallAdvisor, StreamAdvisor {

    companion object {
        private val log = LoggerFactory.getLogger(AdvisorA::class.java)
    }
    override fun getName(): String = this.javaClass.simpleName

    override fun getOrder(): Int = Ordered.HIGHEST_PRECEDENCE + 1

    override fun adviseCall(
        request: ChatClientRequest,
        chain: CallAdvisorChain,
    ): ChatClientResponse {
        log.info("[전처리]")
        val response = chain.nextCall(request)
        log.info("[후처리]")
        return response
    }

    override fun adviseStream(
        request: ChatClientRequest,
        chain: StreamAdvisorChain,
    ): Flux<ChatClientResponse> {
        log.info("[전처리]")
        val response: Flux<ChatClientResponse> = chain.nextStream(request)
        return response
    }
}