package com.llmstd.llmstdai.controller

import com.llmstd.llmstdai.advisor.AdvisorService1
import com.llmstd.llmstdai.advisor.AdvisorService2
import com.llmstd.llmstdai.chatmodel.AiService
import com.llmstd.llmstdai.chatmodel.ChatClientAiService
import com.llmstd.llmstdai.multimodal.MultiModalService
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import reactor.core.publisher.Flux
import java.io.IOException

@RestController
@RequestMapping("/ai")
class AiController(
    private val aiService: AiService,
    private val chatClientAiService: ChatClientAiService,
    private val multiModalService: MultiModalService,
    private val advisorService1: AdvisorService1,
    private val advisorService2: AdvisorService2
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


    @PostMapping(
        value = ["/image-analysis"],
        consumes = [MediaType.MULTIPART_FORM_DATA_VALUE],
        produces = [MediaType.APPLICATION_NDJSON_VALUE]
    )
    @Throws(IOException::class)
    fun imageAnalysis(
        @RequestParam("question") question: String?,
        @RequestParam(value = "attach", required = false) attach: MultipartFile?,
    ): Flux<String> {
        val contentType = attach?.contentType

        // 이미지가 업로드 되지 않았을 경우
        if (attach == null || contentType == null || !contentType.contains("image/")) {
            val response = Flux.just<String>("이미지를 올려주세요.")
            return response
        }

//        return multiModalService.imageAnalysis(question ?: "이미지를 설명해주세요.", contentType, attach.bytes)
        return Flux.empty()
    }

    @PostMapping(
        value = ["/image-generate"],
        consumes = [MediaType.MULTIPART_FORM_DATA_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun imageGenerate(@RequestParam("description") description: String?): String? {
        return try {
            multiModalService.generateImage(description ?: "")
        } catch (e: Exception) {
            e.printStackTrace()
            return "Error: " + e.message
        }
    }

    @PostMapping(
        value = ["/image-edit"],
        consumes = [MediaType.MULTIPART_FORM_DATA_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun imageEdit(
        @RequestParam("description") description: String?,
        @RequestParam("originalImage") originalImage: MultipartFile,
        @RequestParam("maskImage") maskImage: MultipartFile,
    ): String? {
        return try {
            multiModalService.editImage(description ?: "", originalImage.bytes, maskImage.bytes)
        } catch (e: Exception) {
            e.printStackTrace()
            return "Error: " + e.message
        }
    }

//    @PostMapping(
//        value = ["/advisor-chain"],
//        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
//        produces = [MediaType.TEXT_PLAIN_VALUE]
//    )
//    fun advisorChain(@RequestParam("question") question: String): String {
//        val response: String = advisorService1.advisorChain1(question)
//        return response
//    }

    @PostMapping(
        value = ["/advisor-chain"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.APPLICATION_NDJSON_VALUE]
    )
    fun advisorChainByStream(@RequestParam("question") question: String): Flux<String> {
        return advisorService1.advisorChain2(question)
    }

    @PostMapping(
        value = ["/advisor-context"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun advisorContext(@RequestParam("question") question: String): String {
        val response: String = advisorService2.advisorContext(question)
        return response
    }
//
//    @PostMapping(
//        value = ["/advisor-logging"],
//        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
//        produces = [MediaType.TEXT_PLAIN_VALUE]
//    )
//    fun advisorLogging(@RequestParam("question") question: String?): String? {
//        val response: String? = aiService3.advisorLogging(question)
//        return response
//    }
//
//    @PostMapping(
//        value = ["/advisor-safe-guard"],
//        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
//        produces = [MediaType.TEXT_PLAIN_VALUE]
//    )
//    fun advisorSafeGuard(@RequestParam("question") question: String?): String? {
//        val response: String? = aiService4.advisorSafeGuard(question)
//        return response
//    }
}
