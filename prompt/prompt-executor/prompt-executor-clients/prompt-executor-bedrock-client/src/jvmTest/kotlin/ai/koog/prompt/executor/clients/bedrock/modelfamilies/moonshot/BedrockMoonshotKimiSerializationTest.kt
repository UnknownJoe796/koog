package ai.koog.prompt.executor.clients.bedrock.modelfamilies.moonshot

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.ToolParameterDescriptor
import ai.koog.agents.core.tools.ToolParameterType
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.bedrock.BedrockModels
import ai.koog.prompt.executor.clients.bedrock.modelfamilies.moonshot.KimiRequest.Companion.MAX_TOKENS_DEFAULT
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.prompt.params.LLMParams
import ai.koog.prompt.streaming.StreamFrame
import kotlin.time.Clock
import kotlin.time.Instant
import kotlinx.serialization.json.jsonObject
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class BedrockMoonshotKimiSerializationTest {

    private val mockClock = object : Clock {
        override fun now(): Instant = Clock.System.now()
    }

    private val model = BedrockModels.MoonshotKimiK2Thinking
    private val systemMessage = "You are a helpful assistant."
    private val userMessage = "Tell me about Paris."
    private val toolName = "get_weather"

    @Test
    fun `createKimiRequest with basic prompt`() {
        val temperature = 0.7

        val prompt = Prompt.build("test", params = LLMParams(temperature = temperature)) {
            system(systemMessage)
            user(userMessage)
        }

        val request = BedrockMoonshotKimiSerialization.createKimiRequest(prompt, model, emptyList())

        assertNotNull(request)
        assertEquals(model.id, request.model)
        assertEquals(MAX_TOKENS_DEFAULT, request.maxTokens)
        assertEquals(temperature, request.temperature)
        assertEquals(true, request.reasoning)

        assertEquals(2, request.messages.size)

        assertEquals("system", request.messages[0].role)
        assertEquals(systemMessage, request.messages[0].content)

        assertEquals("user", request.messages[1].role)
        assertEquals(userMessage, request.messages[1].content)
    }

    @Test
    fun `createKimiRequest with conversation history`() {
        val userNewMessage = "Hello, who are you?"
        val assistantMessage = "I'm an AI assistant. How can I help you today?"

        val prompt = Prompt.build("test") {
            system(systemMessage)
            user(userNewMessage)
            assistant(assistantMessage)
            user(userMessage)
        }

        val request = BedrockMoonshotKimiSerialization.createKimiRequest(prompt, model, emptyList())

        assertNotNull(request)

        assertEquals(4, request.messages.size)

        assertEquals("system", request.messages[0].role)
        assertEquals(systemMessage, request.messages[0].content)

        assertEquals("user", request.messages[1].role)
        assertEquals(userNewMessage, request.messages[1].content)

        assertEquals("assistant", request.messages[2].role)
        assertEquals(assistantMessage, request.messages[2].content)

        assertEquals("user", request.messages[3].role)
        assertEquals(userMessage, request.messages[3].content)
    }

    @Test
    fun `createKimiRequest with tools`() {
        val description = "Get current weather for a city"

        val tools = listOf(
            ToolDescriptor(
                name = toolName,
                description = description,
                requiredParameters = listOf(
                    ToolParameterDescriptor("city", "The city name", ToolParameterType.String)
                ),
                optionalParameters = listOf(
                    ToolParameterDescriptor("units", "Temperature units", ToolParameterType.String)
                )
            )
        )

        val prompt = Prompt.build("test") {
            user("What's the weather in Paris?")
        }

        val request = BedrockMoonshotKimiSerialization.createKimiRequest(prompt, model, tools)

        assertNotNull(request)

        assertNotNull(request.tools)
        assertEquals(1, request.tools.size)
        assertEquals(toolName, request.tools[0].function.name)
        assertEquals(description, request.tools[0].function.description)

        val parameters = request.tools[0].function.parameters
        assertNotNull(parameters)
        assertTrue(
            parameters.jsonObject.toString().contains("city"),
            "Required parameter \"city\" not found"
        )
    }

    @Test
    fun `createKimiRequest default temperature`() {
        val prompt = Prompt.build("test") {
            user("Tell me a story.")
        }

        val request = BedrockMoonshotKimiSerialization.createKimiRequest(prompt, model, emptyList())
        assertEquals(null, request.temperature)
    }

    @Test
    fun `createKimiRequest respects model temperature capability`() {
        val temperature = 0.3

        val promptWithTemperature = Prompt.build("test", params = LLMParams(temperature = temperature)) {
            user("Tell me a story.")
        }

        val request = BedrockMoonshotKimiSerialization.createKimiRequest(promptWithTemperature, model, emptyList())
        assertEquals(temperature, request.temperature)

        val modelWithoutTemperature = LLModel(
            provider = LLMProvider.Bedrock,
            id = "test-model",
            capabilities = listOf(LLMCapability.Completion), // No temperature capability
            contextLength = 1_000L,
        )

        val requestWithoutTemp = BedrockMoonshotKimiSerialization.createKimiRequest(
            promptWithTemperature,
            modelWithoutTemperature,
            emptyList()
        )
        assertEquals(null, requestWithoutTemp.temperature)
    }

    @Test
    fun `createKimiRequest with reasoning disabled`() {
        val prompt = Prompt.build("test") {
            user("Tell me a story.")
        }

        val request = BedrockMoonshotKimiSerialization.createKimiRequest(
            prompt,
            model,
            emptyList(),
            enableReasoning = false
        )
        assertEquals(null, request.reasoning)
    }

    @Test
    fun `parseKimiResponse with text content`() {
        val responseContent = "Paris is the capital of France"
        // language=json
        val responseJson = """
            {
                "id": "resp_01234567",
                "model": "moonshot.kimi-k2-thinking",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "$responseContent and one of the most visited cities in the world."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 20,
                    "total_tokens": 45
                }
            }
        """.trimIndent()

        val messages = BedrockMoonshotKimiSerialization.parseKimiResponse(responseJson, mockClock)

        assertNotNull(messages)
        assertEquals(1, messages.size)

        val message = messages.first()
        assertTrue(message is Message.Assistant)
        assertContains(message.content, responseContent)
        assertEquals("stop", message.finishReason)

        assertEquals(25, message.metaInfo.inputTokensCount)
        assertEquals(20, message.metaInfo.outputTokensCount)
        assertEquals(45, message.metaInfo.totalTokensCount)
    }

    @Test
    fun `parseKimiResponse with reasoning content`() {
        val reasoningContent = "Let me think about this step by step..."
        val responseContent = "Paris is the capital of France."
        // language=json
        val responseJson = """
            {
                "id": "resp_01234567",
                "model": "moonshot.kimi-k2-thinking",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "$reasoningContent",
                            "content": "$responseContent"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 50,
                    "total_tokens": 75,
                    "reasoning_tokens": 30
                }
            }
        """.trimIndent()

        val messages = BedrockMoonshotKimiSerialization.parseKimiResponse(responseJson, mockClock)

        assertNotNull(messages)
        assertEquals(2, messages.size)

        // First message should be reasoning
        val reasoning = messages[0]
        assertTrue(reasoning is Message.Reasoning, "First message should be Reasoning, was: ${reasoning::class.simpleName}")
        assertEquals(reasoningContent, reasoning.content)

        // Second message should be assistant text
        val assistant = messages[1]
        assertTrue(assistant is Message.Assistant)
        assertEquals(responseContent, assistant.content)
    }

    @Test
    fun `parseKimiResponse with tool call content`() {
        val callId = "call_01234567"
        // language=json
        val responseJson = """
            {
                "id": "resp_01234567",
                "model": "moonshot.kimi-k2-thinking",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "$callId",
                                    "type": "function",
                                    "function": {
                                        "name": "$toolName",
                                        "arguments": "{\"city\":\"Paris\",\"units\":\"celsius\"}"
                                    }
                                }
                            ]
                        },
                        "finish_reason": "tool_calls"
                    }
                ],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 15,
                    "total_tokens": 40
                }
            }
        """.trimIndent()

        val messages = BedrockMoonshotKimiSerialization.parseKimiResponse(responseJson, mockClock)

        assertNotNull(messages)
        assertEquals(1, messages.size)

        val message = messages.first()
        assertTrue(message is Message.Tool.Call)
        assertEquals(callId, message.id)
        assertEquals(toolName, message.tool)

        assertEquals(25, message.metaInfo.inputTokensCount)
        assertEquals(15, message.metaInfo.outputTokensCount)
        assertEquals(40, message.metaInfo.totalTokensCount)
    }

    @Test
    fun `parseKimiResponse with both text and tool calls`() {
        val textContent = "I'll check the weather for you."
        val callId = "call_01234567"

        // language=json
        val responseJson = """
            {
                "id": "resp_01234567",
                "model": "moonshot.kimi-k2-thinking",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "$textContent",
                            "tool_calls": [
                                {
                                    "id": "$callId",
                                    "type": "function",
                                    "function": {
                                        "name": "$toolName",
                                        "arguments": "{\"city\":\"Paris\"}"
                                    }
                                }
                            ]
                        },
                        "finish_reason": "tool_calls"
                    }
                ],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 30,
                    "total_tokens": 55
                }
            }
        """.trimIndent()

        val messages = BedrockMoonshotKimiSerialization.parseKimiResponse(responseJson, mockClock)

        assertNotNull(messages)
        assertEquals(2, messages.size)

        val assistantMessage = messages[0]
        assertTrue(assistantMessage is Message.Assistant)
        assertEquals(textContent, assistantMessage.content)

        val toolMessage = messages[1]
        assertTrue(toolMessage is Message.Tool.Call)
        assertEquals(callId, toolMessage.id)
        assertEquals(toolName, toolMessage.tool)
    }

    @Test
    fun `parseKimiStreamChunk with content delta`() {
        val chunkJson = """
            {
                "id": "resp_01234567",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "Paris is "
                        }
                    }
                ]
            }
        """.trimIndent()

        val content = BedrockMoonshotKimiSerialization.parseKimiStreamChunk(chunkJson)
        assertEquals(listOf("Paris is ").map(StreamFrame::Append), content)
    }

    @Test
    fun `parseKimiStreamChunk with reasoning content delta`() {
        val chunkJson = """
            {
                "id": "resp_01234567",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "Let me think..."
                        }
                    }
                ]
            }
        """.trimIndent()

        val content = BedrockMoonshotKimiSerialization.parseKimiStreamChunk(chunkJson)
        assertEquals(listOf("Let me think...").map(StreamFrame::Append), content)
    }

    @Test
    fun `parseKimiStreamChunk with empty content`() {
        val chunkJson = """
            {
                "id": "resp_01234567",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": ""
                        }
                    }
                ]
            }
        """.trimIndent()

        val content = BedrockMoonshotKimiSerialization.parseKimiStreamChunk(chunkJson)
        assertEquals(listOf("").map(StreamFrame::Append), content)
    }

    @Test
    fun `parseKimiStreamChunk with null content`() {
        val chunkJson = """
            {
                "id": "resp_01234567",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": null
                        }
                    }
                ]
            }
        """.trimIndent()

        val content = BedrockMoonshotKimiSerialization.parseKimiStreamChunk(chunkJson)
        assertEquals(emptyList(), content)
    }

    @Test
    fun `parseKimiStreamChunk with finish reason`() {
        val chunkJson = """
            {
                "id": "resp_01234567",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
        """.trimIndent()

        val frames = BedrockMoonshotKimiSerialization.parseKimiStreamChunk(chunkJson, mockClock)
        assertEquals(1, frames.size)
        assertTrue(frames[0] is StreamFrame.End)

        val endFrame = frames[0] as StreamFrame.End
        assertEquals("stop", endFrame.finishReason)
        assertEquals(10, endFrame.metaInfo.inputTokensCount)
        assertEquals(20, endFrame.metaInfo.outputTokensCount)
        assertEquals(30, endFrame.metaInfo.totalTokensCount)
    }

    @Test
    fun `parseKimiStreamChunk with tool call`() {
        val callId = "call_01234567"
        val chunkJson = """
            {
                "id": "resp_01234567",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "$callId",
                                    "type": "function",
                                    "function": {
                                        "name": "$toolName",
                                        "arguments": "{\"city\":\"Paris\"}"
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        """.trimIndent()

        val frames = BedrockMoonshotKimiSerialization.parseKimiStreamChunk(chunkJson)
        assertEquals(1, frames.size)
        assertTrue(frames[0] is StreamFrame.ToolCall)

        val toolCallFrame = frames[0] as StreamFrame.ToolCall
        assertEquals(callId, toolCallFrame.id)
        assertEquals(toolName, toolCallFrame.name)
    }
}
