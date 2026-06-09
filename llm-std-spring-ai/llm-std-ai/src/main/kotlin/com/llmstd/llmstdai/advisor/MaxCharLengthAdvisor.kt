package com.llmstd.llmstdai.advisor

import org.slf4j.LoggerFactory
import org.springframework.ai.chat.client.ChatClientRequest
import org.springframework.ai.chat.client.ChatClientResponse
import org.springframework.ai.chat.client.advisor.api.CallAdvisor
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain
import org.springframework.ai.chat.messages.UserMessage

class MaxCharLengthAdvisor(
    private val _order: Int = 0
): CallAdvisor {
    companion object {
        const val MAX_CHAR_LENGTH = "maxCharLength"
        private val log = LoggerFactory.getLogger(MaxCharLengthAdvisor::class.java)
    }

    private val maxCharLength = 300


    override fun getName(): String {
        return javaClass.simpleName
    }

    override fun getOrder(): Int {
        return _order
    }

    override fun adviseCall(
        request: ChatClientRequest,
        chain: CallAdvisorChain,
    ): ChatClientResponse {
        log.info("MaxCharLengthAdvisor advising call with maxCharLength: $maxCharLength")
        val mutatedRequest = augmentPrompt(request)
        val response = chain.nextCall(mutatedRequest)
        return response
    }

    private fun augmentPrompt(request: ChatClientRequest): ChatClientRequest {
        var userText = "${this.maxCharLength}자 이내로 답변해주세요."
        val maxCharLength: Int? = request.context.get(MAX_CHAR_LENGTH) as Int?
        if (maxCharLength != null) {
            userText = "${maxCharLength}자 이내로 답변해주세요."
        }

        val finalUserText = userText

        val originPrompt = request.prompt
        val augmentedPrompt = originPrompt.augmentUserMessage { userMessage ->
                UserMessage.builder()
                    .text(userMessage.text + " " + finalUserText)
                    .build()
            }

        return request.mutate()
            .prompt(augmentedPrompt)
            .build()
    }
}