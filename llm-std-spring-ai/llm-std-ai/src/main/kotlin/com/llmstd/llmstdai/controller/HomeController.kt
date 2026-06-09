package com.llmstd.llmstdai.controller

import org.springframework.stereotype.Controller
import org.springframework.web.bind.annotation.GetMapping

@Controller
class HomeController {

    @GetMapping("/")
    fun home(): String {
        return "home"
    }

    @GetMapping("/advisor")
    fun advisorHome(): String {
        return "advisor-home"
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