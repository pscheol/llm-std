package com.llmstd.llmstdai.config

import org.springframework.ai.embedding.EmbeddingModel
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.ai.vectorstore.pgvector.PgVectorStore
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.context.annotation.Primary
import org.springframework.jdbc.core.JdbcTemplate

@Configuration
class VectorStoreConfig(
    private val jdbcTemplate: JdbcTemplate,
    private val embeddingModel: EmbeddingModel
) {
    companion object {
        private const val VECTOR_STORE_DIMENSIONS = 768
    }

    @Bean
    @Primary
    fun defaultVectorStore(): VectorStore =
        pgVectorStore("vector_store")

    @Bean
    fun lawVectorStore(): VectorStore =
        pgVectorStore("law_vector_store")

    @Bean
    fun carVectorStore(): VectorStore =
        pgVectorStore("car_vector_store")

    @Bean
    fun userQuestionVectorStore(): VectorStore =
        pgVectorStore("user_question_vector_store")

    private fun pgVectorStore(tableName: String): VectorStore =
        PgVectorStore.builder(jdbcTemplate, embeddingModel)
            .vectorTableName(tableName)
            .dimensions(VECTOR_STORE_DIMENSIONS)
            .initializeSchema(true)
            .build()
}
