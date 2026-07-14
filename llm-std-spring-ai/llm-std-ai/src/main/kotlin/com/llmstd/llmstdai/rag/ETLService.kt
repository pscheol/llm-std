package com.llmstd.llmstdai.rag

import org.slf4j.LoggerFactory
import org.springframework.ai.chat.model.ChatModel
import org.springframework.ai.document.Document
import org.springframework.ai.model.transformer.KeywordMetadataEnricher
import org.springframework.ai.reader.TextReader
import org.springframework.ai.reader.jsoup.JsoupDocumentReader
import org.springframework.ai.reader.jsoup.config.JsoupDocumentReaderConfig
import org.springframework.ai.reader.pdf.PagePdfDocumentReader
import org.springframework.ai.reader.tika.TikaDocumentReader
import org.springframework.ai.transformer.splitter.TokenTextSplitter
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.core.io.ByteArrayResource
import org.springframework.core.io.UrlResource
import org.springframework.stereotype.Service
import org.springframework.web.multipart.MultipartFile
import java.io.IOException

@Service
class ETLService(
    private val chatModel: ChatModel,
    private val vectorStore: VectorStore
) {

    companion object {
        private val log = LoggerFactory.getLogger(ETLService::class.java)
    }
    @Throws(IOException::class)
    fun etlFromFile(title: String, author: String, attach: MultipartFile): String {

        //extract
        val documents: List<Document> = extractFromFile(attach);

        log.info("변환된 Document 수 : {} 개", documents.size)

        //Trans
        documents.forEach { document ->
            document.metadata += mapOf(
                "title" to title,
                "author" to author,
                "source" to (attach.originalFilename ?: "unknown"),
            )
        }

        val transDocuments = transform(documents)

        //적재
        vectorStore.add(transDocuments)

        return "추출-변환-적재 완료"
    }

    @Throws(Exception::class)
    fun etlFromHtml(title: String, author: String, url: String): String {
        val resource = UrlResource(url)

        val reader = JsoupDocumentReader(resource,
            JsoupDocumentReaderConfig.builder()
                .charset("utf-8")
                .selector("#content")
                .additionalMetadata(mapOf(
                    "title" to title,
                    "author" to author,
                    "url" to url
                ))
                .build()
            )

        val transform = transform(reader.read())

        vectorStore.add(transform)

        return "추출-변환-적재 완료"
    }

    @Throws(IOException::class)
    fun extractFromFile(attach: MultipartFile): List<Document> {
        val resource = ByteArrayResource(attach.bytes)

        val document = if (attach.contentType == "text/plain") {
            TextReader(resource)
        } else if (attach.contentType == "application/pdf") {
            PagePdfDocumentReader(resource)
        } else {
            TikaDocumentReader(resource)
        }

        return document.read()
    }

    fun transform(documents: List<Document>): List<Document> {

        //분할
        val transDocuments: TokenTextSplitter = TokenTextSplitter.builder()
            .build()


        //메타데이터 키워드 추가
        val enricher = KeywordMetadataEnricher.builder(chatModel)
            .keywordCount(5)
            .build()

        return enricher.apply(transDocuments.apply(documents))
    }
}