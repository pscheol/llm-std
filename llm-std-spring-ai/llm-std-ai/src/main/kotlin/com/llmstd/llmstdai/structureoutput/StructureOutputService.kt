package com.llmstd.llmstdai.structureoutput

import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.prompt.PromptTemplate
import org.springframework.ai.converter.BeanOutputConverter
import org.springframework.ai.converter.ListOutputConverter
import org.springframework.core.ParameterizedTypeReference
import org.springframework.stereotype.Service


@Service
class StructureOutputService(private val chatClient: ChatClient) {

    fun listOutputConverterLowLevel(city: String): List<String> {
        //구조화된 출력 변환기 생성
        val converter = ListOutputConverter();
        //프롬프트 템플릿 생성
        val promptTemplate = PromptTemplate.builder()
            .template("{city}에서 유명한 호텔 목록 5개 출력하세요. {format}")
            .build()

        //프롬프트 생성
        val prompt = promptTemplate.create(
            mapOf("city" to city, "format" to converter.format)
        )
        //LLM의 ㅟㅁ표로 구분도니 텍스트 출력 얻기
        val commaSeparatedString: String? = chatClient.prompt(prompt)
            .call()
            .content()

        val hotelList = converter.convert(commaSeparatedString!!)
        return hotelList
    }

    fun listOutputConverterHighLevel(city: String): List<String>? {
        return chatClient.prompt()
            .user("%s에서 유명한 호텔 목록 5개를 출력하세요".format(city))
            .call()
            .entity(ListOutputConverter())
    }

    fun beanOutputConverterLowLevel(city: String): Hotel {

        val beanOutputConverter: BeanOutputConverter<Hotel> = BeanOutputConverter<Hotel>(Hotel::class.java)

        val promptTemplate = PromptTemplate.builder()
            .template("{city}에서 유명한 호텔 목록 5개 출력하세요. {format}")
            .build()

        //프롬프트 생성

        val prompt = promptTemplate.create(mapOf(
            "city" to city,
            "format" to beanOutputConverter.format
        ))

        val json: String = chatClient.prompt(prompt)
            .call()
            .content().orEmpty()

        val hotel = beanOutputConverter.convert(json)
        return hotel
    }

    fun beanOutputConverterHighLevel(city: String): Hotel? {
        return chatClient.prompt()
            .user("%s에서 유명한 호텔 목록 5개를 출력하세요".format(city))
            .call()
            .entity(Hotel::class.java)
    }


    fun genericBeanOutputConverterLowLevel(city: String): List<Hotel> {

        val beanOutputConverter: BeanOutputConverter<List<Hotel>> =
            BeanOutputConverter<List<Hotel>>(object : ParameterizedTypeReference<List<Hotel>>() {})

        val promptTemplate = PromptTemplate.builder()
            .template("{city}에서 유명한 호텔 목록 5개 출력하세요. {format}")
            .build()

        //프롬프트 생성

        val prompt = promptTemplate.create(mapOf(
            "city" to city,
            "format" to beanOutputConverter.format
        ))

        val json: String = chatClient.prompt(prompt)
            .call()
            .content().orEmpty()

        val hotel = beanOutputConverter.convert(json)
        return hotel
    }

    fun genericBeanOutputConverterHighLevel(city: String): List<Hotel>? {
        return chatClient.prompt()
            .user("%s에서 유명한 호텔 목록 5개를 출력하세요".format(city))
            .call()
            .entity(object : ParameterizedTypeReference<List<Hotel>>(){})
    }
}