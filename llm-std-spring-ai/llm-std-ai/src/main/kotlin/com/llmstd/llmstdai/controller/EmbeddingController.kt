package com.llmstd.llmstdai.controller

import com.llmstd.llmstdai.embedding.EmbeddingService
import org.slf4j.LoggerFactory
import org.springframework.ai.document.Document
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/ai")
class EmbeddingController(
    private val embeddingService: EmbeddingService
) {
    companion object {
        private val log = LoggerFactory.getLogger(AiController::class.java)
    }
    // ##### 요청 매핑 메소드 #####
    @PostMapping(
        value = ["/text-embedding"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun textEmbedding(@RequestParam("question") question: String): String {
        embeddingService.textEmbedding(question)
        return "서버 터미널(콘솔) 출력을 확인하세요."
    }

    @PostMapping(
        value = ["/add-document"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun addDocument(@RequestParam("question") question: String): String {
        embeddingService.addDoc(question)
        return "벡터 저장소에 Document들이 저장되었습니다."
    }

    @PostMapping(
        value = ["/search-document-1"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun searchDocument1(@RequestParam("question") question: String): String {
        val documents: List<Document> = embeddingService.searchDocument1(question)

        var text = ""
        for (document in documents) {
            text += "<div class='mb-2'>"
            text += "  <span class='me-2'>유사도 점수: ${document.score},</span>"
            text += "  <span>${document.text}(${document.metadata.get("year")})</span>"
            text += "</div>"
        }
        return text
    }

    @PostMapping(
        value = ["/search-document-2"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun searchDocument2(@RequestParam("question") question: String): String {
        val documents: List<Document> = embeddingService.searchDocument2(question)
        var text = ""
        for (document in documents) {
            text += "<div class='mb-2'>"
            text += "  <span class='me-2'>유사도 점수: ${document.score},</span>"
            text += "  <span>${document.text}(${document.metadata.get("year")})</span>"
            text += "</div>"
        }
        return text
    }

    @PostMapping(
        value = ["/delete-document"],
        consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun deleteDocument(@RequestParam("question") question: String): String {
        embeddingService.deleteDocument()
        return "Document들이 삭제되었습니다."
    }
}
