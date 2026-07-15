package com.llmstd.llmstdai.rag

import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor
import org.springframework.ai.vectorstore.SearchRequest
import org.springframework.ai.vectorstore.VectorStore
import org.springframework.core.Ordered
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.stereotype.Service

@Service
class RetrievalService(
    private val chatClientBuilder: ChatClient.Builder,
    private val vectorStore: VectorStore,
    private val jdbcTemplate: JdbcTemplate
) {

    private val chatClient: ChatClient = chatClientBuilder
        .defaultAdvisors(
            SimpleLoggerAdvisor(Ordered.LOWEST_PRECEDENCE - 1)
        )
        .build()

    fun clearVectorStore() {
        jdbcTemplate.update("TRUNCATE TABLE vector_store")
    }

    fun ragChat(question: String, score: Double, source: String): String {
        val searchRequestBuilder = SearchRequest.builder()
            .similarityThreshold(score)
            .topK(3)


//        searchRequestBuilder.filterExpression("source == '$source'")
        val searchRequest = searchRequestBuilder.build()

        val answerAdvisor = QuestionAnswerAdvisor.builder(vectorStore)
            .searchRequest(searchRequest)
            .build()

        val answer = chatClient.prompt()
            .user(question)
            .advisors(answerAdvisor)
            .call()
            .content()

        return answer?: ""
    }
}