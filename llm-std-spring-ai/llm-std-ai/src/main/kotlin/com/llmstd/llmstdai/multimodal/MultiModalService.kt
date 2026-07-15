package com.llmstd.llmstdai.multimodal

import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.ai.chat.prompt.Prompt
import org.springframework.ai.image.ImageModel
import org.springframework.ai.image.ImageOptions
import org.springframework.ai.image.ImageOptionsBuilder
import org.springframework.stereotype.Service
import org.springframework.util.MimeTypeUtils
import reactor.core.publisher.Flux

@Service
class MultiModalService(
    private val chatClient: ChatClient,
//    private val imageModel: ImageModel
) {
    fun editImage(description: String, bytes: ByteArray, bytes2: ByteArray): String {
        throw UnsupportedOperationException("Ollama는 Spring AI ImageModel 기반 이미지 편집을 지원하지 않습니다.")
    }

    fun generateImage(description: String): String {
        throw UnsupportedOperationException("Ollama starter는 ImageModel 빈을 생성하지 않으므로 이미지 생성 옵션(width/height/N)을 사용할 수 없습니다.")
    }




    private fun ollamaVisionOptions(): ImageOptions {
        return ImageOptionsBuilder.builder()
            .width(1536)
            .height(1024)
            .n(1)
            .model("gemma4:e4b-it-q8_0")
            .build()
    }

    private fun koToEn(text: String): String {
        val question = """
            당신은 번역사 입니다. 아래 한글 문장을 영어 문장으로 번역해주세요.
            %s
        """.format(text)
        val userMessage = UserMessage.builder()
            .text(question)
            .build()

        val prompt = Prompt.builder()
            .messages(userMessage)
            .build()

        return chatClient.prompt(prompt)
            .call()
            .content() ?: ""
    }
}
