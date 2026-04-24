package com.llmstd.llmstdai.chatmodel

import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.ai.chat.model.ChatModel
import org.springframework.ai.chat.model.ChatResponse
import org.springframework.ai.chat.prompt.ChatOptions
import org.springframework.ai.chat.prompt.Prompt
import org.springframework.ai.ollama.api.OllamaApi
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux

@Service
class AiService(
    private val chatModel: ChatModel
) {

    fun generateText(question: String): String {
        val (systemMessage: SystemMessage, userMessage: UserMessage, chatOption: ChatOptions) = chatOption(question)

        //프롬프트 설정
        val prompt: Prompt = Prompt.builder()
            .messages(systemMessage, userMessage)
            .chatOptions(chatOption)
            .build()

        val chatResponse: ChatResponse = chatModel.call(prompt)

        val assistantMessage: AssistantMessage? = chatResponse.result?.output


        return assistantMessage?.text ?: ""
    }

    private fun chatOption(question: String): Triple<SystemMessage, UserMessage, ChatOptions> {
        //시스템 메시지 새성
        val systemMessage: SystemMessage = SystemMessage.builder()
            .text("사용자 질문에 대해 한국어로 답변해야 합니다.")
            .build()

        //사용자 메시지 생성
        val userMessage: UserMessage = UserMessage.builder()
            .text(question)
            .build()

        //대화 옵션 설정
        val chatOption: ChatOptions = ChatOptions.builder()
            .model("gemma4:e4b-it-q8_0")
            .temperature(0.3)
            .maxTokens(1000)
            .build()
        return Triple(systemMessage, userMessage, chatOption)
    }

    fun generateTextByStream(question: String): Flux<String> {
        val (systemMessage: SystemMessage, userMessage: UserMessage, chatOption: ChatOptions) = chatOption(question)

        //프롬프트 설정
        val prompt: Prompt = Prompt.builder()
            .messages(systemMessage, userMessage)
            .chatOptions(chatOption)
            .build()

        val chatResponse: Flux<ChatResponse> = chatModel.stream(prompt)

        return chatResponse.map { response ->
            val assistantMessage = response.result?.output
            assistantMessage?.text ?: ""
        }
    }

}