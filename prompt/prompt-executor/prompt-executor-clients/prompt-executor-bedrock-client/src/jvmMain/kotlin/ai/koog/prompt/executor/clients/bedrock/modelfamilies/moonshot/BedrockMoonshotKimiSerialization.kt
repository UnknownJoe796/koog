package ai.koog.prompt.executor.clients.bedrock.modelfamilies.moonshot

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.bedrock.modelfamilies.BedrockToolSerialization
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.ResponseMetaInfo
import ai.koog.prompt.streaming.StreamFrame
import kotlin.time.Clock
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/**
 * Serialization utilities for Moonshot Kimi K2 models on AWS Bedrock.
 *
 * Kimi K2 uses an OpenAI-compatible API format with additional support for:
 * - Extended reasoning/thinking mode via the `reasoning` parameter
 * - Tool calling with function support
 * - Streaming responses including reasoning content
 */
internal object BedrockMoonshotKimiSerialization {

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        explicitNulls = false
    }

    /**
     * Creates a Kimi request from a prompt and model configuration.
     *
     * @param prompt The prompt to convert
     * @param model The model to use (affects temperature capability check)
     * @param tools List of available tools for function calling
     * @param enableReasoning Whether to enable extended thinking mode (defaults to true for Thinking models)
     */
    @OptIn(ExperimentalUuidApi::class)
    internal fun createKimiRequest(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
        enableReasoning: Boolean = true
    ): KimiRequest {
        val messages = mutableListOf<KimiMessage>()

        prompt.messages.forEach { msg ->
            when (msg) {
                is Message.System -> messages.add(
                    KimiMessage(role = "system", content = msg.content)
                )

                is Message.User -> messages.add(
                    KimiMessage(role = "user", content = msg.content)
                )

                is Message.Assistant -> messages.add(
                    KimiMessage(role = "assistant", content = msg.content)
                )

                is Message.Reasoning -> {
                    // Kimi K2 reasoning content is part of the assistant message
                    // When replaying history, include reasoning as a separate assistant message
                    messages.add(
                        KimiMessage(
                            role = "assistant",
                            reasoningContent = msg.content,
                            content = null
                        )
                    )
                }

                is Message.Tool.Call -> {
                    // Find or create assistant message with tool calls
                    val lastMessage = messages.lastOrNull()
                    if (lastMessage?.role == "assistant" && lastMessage.toolCalls != null) {
                        // Add to existing tool calls
                        val updatedToolCalls = lastMessage.toolCalls + KimiToolCall(
                            id = msg.id ?: Uuid.random().toString(),
                            function = KimiFunctionCall(
                                name = msg.tool,
                                arguments = msg.content
                            )
                        )
                        messages[messages.lastIndex] = lastMessage.copy(toolCalls = updatedToolCalls)
                    } else {
                        // Create new assistant message with tool call
                        messages.add(
                            KimiMessage(
                                role = "assistant",
                                content = null,
                                toolCalls = listOf(
                                    KimiToolCall(
                                        id = msg.id ?: Uuid.random().toString(),
                                        function = KimiFunctionCall(
                                            name = msg.tool,
                                            arguments = msg.content
                                        )
                                    )
                                )
                            )
                        )
                    }
                }

                is Message.Tool.Result -> messages.add(
                    KimiMessage(
                        role = "tool",
                        content = msg.content,
                        toolCallId = msg.id ?: Uuid.random().toString()
                    )
                )
            }
        }

        val kimiTools = if (tools.isNotEmpty()) {
            tools.map { tool ->
                KimiTool(
                    function = KimiFunction(
                        name = tool.name,
                        description = tool.description,
                        parameters = buildJsonObject {
                            put("type", "object")
                            put(
                                "properties",
                                buildJsonObject {
                                    (tool.requiredParameters + tool.optionalParameters).forEach { param ->
                                        put(param.name, BedrockToolSerialization.buildToolParameterSchema(param))
                                    }
                                }
                            )
                            if (tool.requiredParameters.isNotEmpty()) {
                                put(
                                    "required",
                                    buildJsonObject {
                                        tool.requiredParameters.forEachIndexed { index, param ->
                                            put(index.toString(), param.name)
                                        }
                                    }
                                )
                            }
                        }
                    )
                )
            }
        } else {
            null
        }

        return KimiRequest(
            model = model.id,
            messages = messages,
            maxTokens = KimiRequest.MAX_TOKENS_DEFAULT,
            temperature = if (model.capabilities.contains(LLMCapability.Temperature)) {
                prompt.params.temperature
            } else {
                null
            },
            tools = kimiTools,
            reasoning = if (enableReasoning) true else null
        )
    }

    /**
     * Parses a Kimi response into a list of response messages.
     *
     * Handles:
     * - Text content (as Assistant messages)
     * - Reasoning content (as Reasoning messages)
     * - Tool calls (as Tool.Call messages)
     */
    internal fun parseKimiResponse(responseBody: String, clock: Clock = Clock.System): List<Message.Response> {
        val response = json.decodeFromString<KimiResponse>(responseBody)

        val metaInfo = parseMetaInfo(clock, response.usage)

        return response.choices.flatMap { choice ->
            val messages = mutableListOf<Message.Response>()

            // Handle reasoning content first (if present)
            choice.message.reasoningContent?.let { reasoningContent ->
                if (reasoningContent.isNotEmpty()) {
                    messages.add(
                        Message.Reasoning(
                            content = reasoningContent,
                            metaInfo = metaInfo
                        )
                    )
                }
            }

            // Handle text content
            choice.message.content?.let { content ->
                if (content.isNotEmpty()) {
                    messages.add(
                        Message.Assistant(
                            content = content,
                            finishReason = choice.finishReason,
                            metaInfo = metaInfo
                        )
                    )
                }
            }

            // Handle tool calls
            choice.message.toolCalls?.forEach { toolCall ->
                messages.add(
                    Message.Tool.Call(
                        id = toolCall.id,
                        tool = toolCall.function.name,
                        content = toolCall.function.arguments,
                        metaInfo = metaInfo
                    )
                )
            }

            messages
        }
    }

    /**
     * Parses a Kimi streaming chunk into stream frames.
     *
     * Handles:
     * - Content deltas (as Append frames)
     * - Reasoning content deltas (as Append frames - streaming reasoning)
     * - Tool call deltas (as ToolCall frames)
     * - Finish reason (as End frame)
     */
    internal fun parseKimiStreamChunk(chunkJsonString: String, clock: Clock = Clock.System): List<StreamFrame> {
        val streamResponse = json.decodeFromString<KimiStreamResponse>(chunkJsonString)
        return buildList {
            val choice = streamResponse.choices.firstOrNull()
            choice?.delta?.let { delta ->
                // Handle reasoning content delta
                delta.reasoningContent?.let { reasoning ->
                    if (reasoning.isNotEmpty()) {
                        add(StreamFrame.Append(reasoning))
                    }
                }
                // Handle regular content delta
                delta.content?.let(StreamFrame::Append)?.let(::add)
                // Handle tool calls
                delta.toolCalls?.map { kimiToolCall ->
                    StreamFrame.ToolCall(
                        id = kimiToolCall.id,
                        name = kimiToolCall.function.name,
                        content = kimiToolCall.function.arguments
                    )
                }?.let(::addAll)
            }
            choice?.finishReason?.let { finishReason ->
                add(
                    StreamFrame.End(
                        finishReason = finishReason,
                        metaInfo = parseMetaInfo(clock, streamResponse.usage)
                    )
                )
            }
        }
    }

    private fun parseMetaInfo(
        clock: Clock,
        usage: KimiUsage?
    ): ResponseMetaInfo = ResponseMetaInfo.create(
        clock = clock,
        totalTokensCount = usage?.totalTokens,
        inputTokensCount = usage?.promptTokens,
        outputTokensCount = usage?.completionTokens
    )
}
