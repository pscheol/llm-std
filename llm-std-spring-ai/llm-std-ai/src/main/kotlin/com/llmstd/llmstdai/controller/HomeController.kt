package com.llmstd.llmstdai.controller

import org.springframework.stereotype.Controller
import org.springframework.web.bind.annotation.GetMapping

@Controller
class HomeController {

    @GetMapping("/")
    fun home(): String {
        return "home"
    }

    @GetMapping("/inmemory")
    fun inmemory(): String {
        return "inmemory"
    }

    @GetMapping("/txt-pdf-word-etl")
    fun txtPdfDocxEtl(): String {
        return "txt-pdf-word-etl"
    }

    @GetMapping("/html-etl")
    fun htmlEtl(): String {
        return "html-etl"
    }

    @GetMapping("/jdbcmemory")
    fun jdbcmemory(): String {
        return "jdbc-chat"
    }


    @GetMapping("/advisor")
    fun advisorHome(): String {
        return "advisor-home"
    }

    @GetMapping("/embeddings")
    fun embeddingHome(): String {
        return "embedding-home"
    }

    @GetMapping("/text-embedding")
    fun textEmbedding(): String {
        return "text-embedding"
    }

    @GetMapping("/add-document")
    fun addDocument(): String {
        return "add-document"
    }

    @GetMapping("/search-document-1")
    fun searchDocument1(): String {
        return "search-document-1"
    }

    @GetMapping("/search-document-2")
    fun searchDocument2(): String {
        return "search-document-2"
    }

    @GetMapping("/delete-document")
    fun deleteDocument(): String {
        return "delete-document"
    }

    @GetMapping("/image-embedding")
    fun faceRecognition(): String {
        return "image-embedding"
    }

    @GetMapping("/stream")
    fun homeStream(): String {
        return "home-stream"
    }
    @GetMapping("/prompt-template")
    fun promptStream(): String {
        return "prompt-template"
    }

    @GetMapping("/zero-shot-prompt")
    fun zeroShotPrompt(): String {
        return "zero-shot-prompt"
    }

    @GetMapping("/image-analysis")
    fun imageAnalysis(): String {
        return "image-analysis"
    }

    @GetMapping("/video-analysis")
    fun videoAnalysis(): String {
        return "video-analysis"
    }

    @GetMapping("/image-generation")
    fun imageGeneration(): String {
        return "image-generation"
    }


    @GetMapping("/advisor-chain")
    fun advisorChain(): String {
        return "advisor-chain"
    }

    @GetMapping("/advisor-context")
    fun advisorContext(): String {
        return "advisor-context"
    }

    @GetMapping("/advisor-logging")
    fun advisorLogging(): String {
        return "advisor-logging"
    }

    @GetMapping("/advisor-safe-guard")
    fun advisorSafeGuard(): String {
        return "advisor-safe-guard"
    }
}