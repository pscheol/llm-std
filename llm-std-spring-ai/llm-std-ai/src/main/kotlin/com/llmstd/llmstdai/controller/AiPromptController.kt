package com.llmstd.llmstdai.controller

import com.llmstd.llmstdai.chatmodel.AiService
import com.llmstd.llmstdai.chatmodel.ChatClientAiService
import com.llmstd.llmstdai.prompt.AIPromptTemplateService
import com.llmstd.llmstdai.prompt.ZeroShotPromptService
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import reactor.core.publisher.Flux

@RestController
@RequestMapping("/ai")
class AiPromptController(
    private val aiPromptTemplateService: AIPromptTemplateService,
    private val zeroShotPromptService: ZeroShotPromptService
) {



    @PostMapping(
        value = ["/prompt-template"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.APPLICATION_NDJSON_VALUE]
    )
    fun chatStream(
        @RequestParam("statement") statement: String,
        @RequestParam("language") language: String
    ): Flux<String> {
        return aiPromptTemplateService.promptTemplate1(statement, language)
    }

    @PostMapping(
        value = ["/zero-shot-prompt"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun zeroShotChat(
        @RequestParam("review") review: String
    ): String? {
        return zeroShotPromptService.zeroShotPrompt(review)
    }


}