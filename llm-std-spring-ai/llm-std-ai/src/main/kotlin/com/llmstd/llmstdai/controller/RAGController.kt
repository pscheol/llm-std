package com.llmstd.llmstdai.controller

import com.llmstd.llmstdai.rag.ETLService
import com.llmstd.llmstdai.rag.RetrievalService
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import java.io.IOException

@RestController
@RequestMapping("/ai")
class RAGController(
    private val etlService: ETLService,
    private val retrievalService: RetrievalService
) {

    @Throws(IOException::class)
    @PostMapping(
        value = ["/txt-pdf-docx-etl"],
        consumes = [MediaType.MULTIPART_FORM_DATA_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun txtPdfDocxEtl(
        @RequestParam("title") title: String,
        @RequestParam("author") author: String,
        @RequestParam("attach") attach: MultipartFile
    ): String {
        return etlService.etlFromFile(title, author, attach)
    }


    @Throws(IOException::class)
    @PostMapping(
        value = ["/html-etml"],
        consumes = [MediaType.MULTIPART_FORM_DATA_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun txtPdfDocxEtl(
        @RequestParam("title") title: String,
        @RequestParam("author") author: String,
        @RequestParam("url") url: String
    ): String {
        return etlService.etlFromHtml(title, author, url)
    }

    @GetMapping("/rag-clear", produces = [MediaType.TEXT_PLAIN_VALUE])
    fun ragClear(): String {
        retrievalService.clearVectorStore()
        return "벡터 저장소 데이터를 모두 삭제했습니다."
    }
    @Throws(IOException::class)
    @PostMapping(
        value = ["/rag-etl"],
        consumes = [MediaType.MULTIPART_FORM_DATA_VALUE],
        produces = [MediaType.TEXT_PLAIN_VALUE]
    )
    fun ragETL(
        @RequestParam("attach") attach: MultipartFile,
        @RequestParam("source") source: String,
        @RequestParam("chunkSize") chunkSize: Int = 200,
        @RequestParam("minChunkSizeChars") minChunkSizeChars: Int = 100,
    ): String {
        etlService.etlFrom(attach, source, chunkSize, minChunkSizeChars)
        return "PDF ETL 성공 처리"
    }

    @PostMapping("/rag-chat", consumes = [MediaType.APPLICATION_FORM_URLENCODED_VALUE], produces = [MediaType.TEXT_PLAIN_VALUE])
    fun ragChat(
        @RequestParam("question") question: String,
        @RequestParam("score") defaultValue: String = "0.0",
        @RequestParam("source") source: String
    ): String {
        return retrievalService.ragChat(question, defaultValue.toDouble(), source)
    }

}