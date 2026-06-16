package com.llmstd.llmstdai.embedding

import org.slf4j.LoggerFactory
import org.springframework.ai.document.Document
import org.springframework.ai.embedding.EmbeddingModel
import org.springframework.ai.embedding.EmbeddingResponse
import org.springframework.ai.vectorstore.SearchRequest
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.ai.vectorstore.pgvector.PgVectorStore
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.stereotype.Service
import java.util.logging.Filter

@Service
class EmbeddingService(
    private val embeddingModel: EmbeddingModel,
    @Qualifier("lawVectorStore")
    private val lawVectorStore: VectorStore,
    @Qualifier("carVectorStore")
    private val carVectorStore: VectorStore,
    @Qualifier("userQuestionVectorStore")
    private val userQuestionVectorStore: VectorStore,
    private val vectorStore: PgVectorStore
) {
    companion object {
        private const val VECTOR_STORE_DIMENSIONS = 768
        private val log = LoggerFactory.getLogger(EmbeddingService::class.java)
    }

    fun textEmbedding(question: String) {
        //embedding
        val response = embeddingModel.embedForResponse(listOf(question))

        //embedding 정보 보기
        val metadata = response.metadata

        log.info("모델 이름 : ${metadata.model}")
        log.info("모델의 임베딩 차원 : ${embeddingModel.dimensions()}")
        log.info("모델의 임베딩 결과 : ${response.result}")


        val embedding = response.results[0]
        log.info("벡터 차원 : ${embedding.output.size}")
        log.info("벡터 : ${embedding.output}")

    }

    fun addDoc(question: String) {
        validateEmbeddingDimensions()

        val userQuestionDocuments = listOf(
            Document(question, metadata("source" to "user", "type" to "question"))
        )
        val lawDocuments = listOf(
            Document("대통령 선거는 5년마다 있습니다.", metadata("source" to "헌법", "year" to 1987)),
            Document("대통령 임기는 4년입니다.", metadata("source" to "헌법", "year" to 1980)),
            Document("국회의원은 법률안을 심의·의결합니다.", metadata("source" to "헌법", "year" to 1987)),
            Document("대통령은 행정부의 수반입니다.", metadata("source" to "헌법", "year" to 1987)),
            Document("국회의원은 4년마다 투표로 뽑습니다.", metadata("source" to "헌법", "year" to 1987))
        )
        val carDocuments = listOf(
            Document("자동차를 사용하려면 등록을 해야합니다.", metadata("source" to "자동차관리법")),
            Document("승용차는 정규적인 점검이 필요합니다.", metadata("source" to "자동차관리법"))
        )

        userQuestionVectorStore.add(userQuestionDocuments)
        lawVectorStore.add(lawDocuments)
        carVectorStore.add(carDocuments)
        log.info(
            "벡터 저장소에 사용자 질문 {}건, 법률 문서 {}건, 자동차 문서 {}건을 저장했습니다.",
            userQuestionDocuments.size,
            lawDocuments.size,
            carDocuments.size
        )
    }

    private fun metadata(vararg entries: Pair<String, Any>): Map<String, Any> =
        mapOf(*entries)

    private fun validateEmbeddingDimensions() {
        val actualDimensions = embeddingModel.dimensions()
        require(actualDimensions == VECTOR_STORE_DIMENSIONS) {
            "pgvector는 ${VECTOR_STORE_DIMENSIONS}차원으로 고정되어 있지만 현재 embedding model은 ${actualDimensions}차원입니다. " +
                "embedding model과 vectorstore.pgvector.dimensions 값을 같은 차원으로 맞춰야 합니다."
        }
    }

    fun searchDocument1(question: String): List<Document> {
        return vectorStore.similaritySearch(question)
    }

    fun searchDocument2(question: String): List<Document> {
        return vectorStore.similaritySearch(SearchRequest.builder()
            .query(question)
            .topK(1)    //최상위 1개
            .similarityThreshold(0.4) //유사성 0.4 이상
            .filterExpression("source == '헌법' && year >= 1987")
            .build()
        )
    }

    fun deleteDocument() {
        vectorStore.delete("source == '헌법' && year >= 1987")

    }
}
