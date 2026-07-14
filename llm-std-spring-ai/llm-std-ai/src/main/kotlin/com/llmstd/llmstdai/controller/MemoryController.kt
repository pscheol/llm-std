package com.llmstd.llmstdai.controller

import com.llmstd.llmstdai.memory.InMemoryAIService
import com.llmstd.llmstdai.memory.JdbcMemoryAIService
import jakarta.servlet.http.HttpSession
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/ai")
class MemoryController(
    private val memoryAIService: InMemoryAIService,
    private val jdbcAIService: JdbcMemoryAIService
) {

    @PostMapping(
        value = ["/in-memory"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun inMemoryChatMemory(
        @RequestParam("question") question: String, session: HttpSession,
    ): String {
        val answer: String = memoryAIService.chat(question, session.id)
        return answer
    }

    @PostMapping(
        value = ["/jdbc-chat"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun jdbcChatMemory(
        @RequestParam("question") question: String, session: HttpSession,
    ): String {
        val answer: String = jdbcAIService.chat(question, session.id)
        return answer
    }

}