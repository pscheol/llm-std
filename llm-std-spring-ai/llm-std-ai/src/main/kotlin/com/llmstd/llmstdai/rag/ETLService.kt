package com.llmstd.llmstdai.rag

import org.apache.poi.hdgf.chunks.Chunk
import org.slf4j.LoggerFactory
import org.springframework.ai.chat.model.ChatModel
import org.springframework.ai.document.Document
import org.springframework.ai.model.transformer.KeywordMetadataEnricher
import org.springframework.ai.reader.JsonMetadataGenerator
import org.springframework.ai.reader.JsonReader
import org.springframework.ai.reader.TextReader
import org.springframework.ai.reader.jsoup.JsoupDocumentReader
import org.springframework.ai.reader.jsoup.config.JsoupDocumentReaderConfig
import org.springframework.ai.reader.pdf.PagePdfDocumentReader
import org.springframework.ai.reader.tika.TikaDocumentReader
import org.springframework.ai.transformer.splitter.TokenTextSplitter
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.core.io.ByteArrayResource
import org.springframework.core.io.Resource
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

    @Throws(Exception::class)
    fun etlFrom(attach: MultipartFile, source: String, chunkSize: Int, minChunkSizeChars: Int) {
        val resource = ByteArrayResource(attach.bytes)
        val reader = PagePdfDocumentReader(resource)
        val documents = reader.read()

        for (document in documents) {
            document.metadata.putAll(mapOf("source" to source))
        }

        // 분할만 수행 후 적재.
        // KeywordMetadataEnricher(transform)는 청크마다 Ollama 챗 모델을 동기 호출하여
        // 로컬 LLM 환경에서는 ETL이 사실상 멈추고 저장이 되지 않으므로 사용하지 않는다.
        val splitter = TokenTextSplitter.builder()
            .withChunkSize(chunkSize)
            .withMinChunkSizeChars(minChunkSizeChars)
            .build()

        val chunks = splitter.apply(documents)
        log.info("분할된 청크 수 : {} 개, 적재 시작", chunks.size)

        vectorStore.add(chunks)

        log.info("벡터 저장소 적재 완료 : {} 개", chunks.size)
    }


    fun etlFromJson(url: String): String {
        val resource: Resource = UrlResource(url)

        val reader = JsonReader(resource,
            JsonMetadataGenerator { jsonMap ->
                mapOf(
                    "title" to (jsonMap["title"] ?: ""),
                    "author" to (jsonMap["author"] ?: ""),
                    "url" to url
                )
            }
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

    fun transform(documents: List<Document>, chunkSize: Int = 800, minChunkSizeChars: Int = 350): List<Document> {

        //분할
        val transDocuments: TokenTextSplitter = TokenTextSplitter.builder()
            .withChunkSize(chunkSize)
            .withMinChunkSizeChars(minChunkSizeChars)
            .build()


        //메타데이터 키워드 추가
        val enricher = KeywordMetadataEnricher.builder(chatModel)
            .keywordCount(5)
            .build()

        return enricher.apply(transDocuments.apply(documents))
    }
}