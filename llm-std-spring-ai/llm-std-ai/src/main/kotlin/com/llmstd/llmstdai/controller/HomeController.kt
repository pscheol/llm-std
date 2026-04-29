package com.llmstd.llmstdai.controller

import org.springframework.stereotype.Controller
import org.springframework.web.bind.annotation.GetMapping

@Controller
class HomeController {

    @GetMapping("/")
    fun home(): String {
        return "home"
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
}