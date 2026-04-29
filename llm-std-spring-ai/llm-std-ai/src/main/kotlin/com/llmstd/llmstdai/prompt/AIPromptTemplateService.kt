package com.llmstd.llmstdai.prompt

import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.prompt.PromptTemplate
import org.springframework.ai.chat.prompt.SystemPromptTemplate
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux

@Service
class AIPromptTemplateService(
    private val chatClient: ChatClient
) {

    companion object {
        private val systemTemplate: PromptTemplate = SystemPromptTemplate.builder()
            .template(
                """
                    답변을 생성할 때 HTML과 CSS를 사용하여 파란 글자로 출력해주세요.
                    <span> 태그 안에 들어갈 내용만 출력해주세요.
                """
            ).build()

        private val userTemplate: PromptTemplate = PromptTemplate.builder()
            .template("""
                다음 한국어 문장을 {language}로 번역해주세요. \n 문장 : {statement}    
            """.trimIndent())
            .build()
    }

    fun promptTemplate1(statement: String, language: String): Flux<String> {
        val prompt = userTemplate.create(mapOf("statement" to statement, "language" to language))
        return chatClient.prompt(prompt)
            .stream()
            .content()
    }

    fun promptTemplate2(statement: String, language: String): Flux<String> {
        return chatClient.prompt()
            .messages(systemTemplate.createMessage(), userTemplate.createMessage(mapOf("statement" to statement, "language" to language)))
            .stream()
            .content()
    }

    fun promptTemplate3(statement: String, language: String): Flux<String> {
        return chatClient.prompt()
            .system(systemTemplate.render())
            .user(userTemplate.render(mapOf("statement" to statement, "language" to language)))
            .stream()
            .content()
    }

    fun promptTemplate4(statement: String, language: String): Flux<String> {
        val systemText = """
            답변을 생성할 때 HTML과 CSS를 사용하여 파란 글자로 출력해주세요.
                    <span> 태그 안에 들어갈 내용만 출력해주세요.
        """
        val userText = """
            다음 한국어 문장을 %s로 번역해주세요. \n 문장 : %s
        """.format(language, statement)

        return chatClient.prompt()
            .system(systemText)
            .user(userText)
            .stream()
            .content()
    }
}