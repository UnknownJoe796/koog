package ai.koog.prompt.executor.clients.bedrock.modelfamilies.moonshot

import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/**
 * Request model for Moonshot Kimi K2 API.
 *
 * Kimi K2 uses an OpenAI-compatible API format with additional fields for reasoning/thinking.
 */
@Serializable
internal data class KimiRequest(
    val model: String,
    val messages: List<KimiMessage>,
    @SerialName("max_tokens") val maxTokens: Int? = MAX_TOKENS_DEFAULT,
    val temperature: Double? = null,
    @SerialName("top_p") val topP: Double? = null,
    val stop: List<String>? = null,
    val stream: Boolean? = null,
    val tools: List<KimiTool>? = null,
    /**
     * When true, enables extended thinking mode for the model.
     * The model will include reasoning_content in its response.
     */
    val reasoning: Boolean? = null
) {
    init {
        if (maxTokens != null) {
            require(maxTokens > 0) { "maxTokens must be greater than 0, but was $maxTokens" }
        }
        if (temperature != null) {
            require(temperature >= 0) { "temperature must be greater than 0, but was $temperature" }
        }
        if (topP != null) {
            require(topP in 0.0..1.0) { "topP must be between 0 and 1, but was $topP" }
        }
    }

    companion object {
        const val MAX_TOKENS_DEFAULT: Int = 8192
    }
}

@Serializable
internal data class KimiMessage(
    val role: String,
    val content: String? = null,
    /**
     * Reasoning/thinking content from the model (only present in assistant messages when reasoning is enabled).
     */
    @SerialName("reasoning_content")
    val reasoningContent: String? = null,
    @SerialName("tool_calls")
    val toolCalls: List<KimiToolCall>? = null,
    @SerialName("tool_call_id")
    val toolCallId: String? = null
)

@Serializable
internal data class KimiTool(
    val type: String = "function",
    val function: KimiFunction
)

@Serializable
internal data class KimiFunction(
    val name: String,
    val description: String,
    val parameters: JsonObject
)

@Serializable
internal data class KimiToolCall(
    val id: String,
    @EncodeDefault
    val type: String = "function",
    val function: KimiFunctionCall
)

@Serializable
internal data class KimiFunctionCall(
    val name: String,
    val arguments: String
)

@Serializable
internal data class KimiResponse(
    val id: String,
    val model: String,
    val choices: List<KimiChoice>,
    val usage: KimiUsage? = null
)

@Serializable
internal data class KimiChoice(
    val index: Int,
    val message: KimiMessage,
    @SerialName("finish_reason")
    val finishReason: String? = null
)

@Serializable
internal data class KimiUsage(
    @SerialName("prompt_tokens")
    val promptTokens: Int,
    @SerialName("completion_tokens")
    val completionTokens: Int,
    @SerialName("total_tokens")
    val totalTokens: Int,
    /**
     * Token count for reasoning/thinking content (if reasoning was enabled).
     */
    @SerialName("reasoning_tokens")
    val reasoningTokens: Int? = null
)

// Streaming response models

@Serializable
internal data class KimiStreamResponse(
    val id: String,
    val choices: List<KimiStreamChoice>,
    val usage: KimiUsage? = null
)

@Serializable
internal data class KimiStreamChoice(
    val index: Int,
    val delta: KimiStreamDelta,
    @SerialName("finish_reason")
    val finishReason: String? = null
)

@Serializable
internal data class KimiStreamDelta(
    val role: String? = null,
    val content: String? = null,
    /**
     * Reasoning/thinking content delta from the model (streamed when reasoning is enabled).
     */
    @SerialName("reasoning_content")
    val reasoningContent: String? = null,
    @SerialName("tool_calls")
    val toolCalls: List<KimiToolCall>? = null
)
