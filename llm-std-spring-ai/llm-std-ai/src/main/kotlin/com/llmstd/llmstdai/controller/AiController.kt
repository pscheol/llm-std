package com.llmstd.llmstdai.controller

import com.llmstd.llmstdai.chatmodel.AiService
import com.llmstd.llmstdai.chatmodel.ChatClientAiService
import org.springframework.ai.chat.client.ChatClient
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import reactor.core.publisher.Flux

@RestController
@RequestMapping("/ai")
class AiController(
    private val aiService: AiService,
    private val chatClientAiService: ChatClientAiService
) {

    @PostMapping(
        value = ["/chat"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun chat(@RequestParam("question") question: String): String {
        return chatClientAiService.generateText(question)
    }

    @PostMapping(
        value = ["/chat-stream"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.APPLICATION_NDJSON_VALUE]
    )
    fun chatStream(@RequestParam("question") question: String): Flux<String> {
        return chatClientAiService.generateTextByStream(question)
    }
}