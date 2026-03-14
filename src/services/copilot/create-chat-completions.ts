import consola from "consola"
import { events } from "fetch-event-stream"

import { copilotHeaders, copilotBaseUrl } from "~/lib/api-config"
import { HTTPError } from "~/lib/error"
import { state } from "~/lib/state"
import { refreshCopilotToken } from "~/lib/token"

/** Valid values for the Copilot API's output_config.effort field. */
const COPILOT_EFFORT_VALUES = new Set(["low", "medium", "high", "max"])

/**
 * Sanitize the payload to ensure compatibility with the Copilot API.
 *
 * The Copilot API is stricter than the OpenAI spec in several ways:
 * - Does not support the `developer` role (OpenAI alias for `system`) → map to `system`
 * - Returns 500 when an assistant message has `content: null` (common in multi-turn
 *   tool-call conversations) → replace with empty string
 * - Some versions reject an empty `tools` array → omit when empty
 * - Only accepts `response_format: { type: "json_object" }` → strip other types
 * - Uses `output_config.effort` (not OpenAI's `reasoning_effort`); accepts only
 *   `"low"`, `"medium"`, `"high"`, `"max"` → map from `reasoning_effort` and strip
 *   unsupported values like `"auto"`
 */
export function sanitizePayload(
  payload: ChatCompletionsPayload,
): ChatCompletionsPayload {
  const messages: Array<Message> = payload.messages.map((msg) => {
    const sanitized: Message = { ...msg }

    // Map unsupported 'developer' role to 'system'
    if (sanitized.role === "developer") {
      sanitized.role = "system"
    }

    // Copilot API returns 500 on null content; use empty string instead
    if (sanitized.content === null) {
      sanitized.content = ""
    }

    return sanitized
  })

  const sanitized: ChatCompletionsPayload = { ...payload, messages }

  // Remove empty tools array and tool_choice to avoid API errors
  if (Array.isArray(sanitized.tools) && sanitized.tools.length === 0) {
    delete sanitized.tools
    delete sanitized.tool_choice
  }

  // Strip response_format types unsupported by the Copilot API
  if (
    sanitized.response_format !== null
    && sanitized.response_format !== undefined
    && sanitized.response_format.type !== "json_object"
  ) {
    delete sanitized.response_format
  }

  // Map OpenAI's reasoning_effort to Copilot's output_config.effort.
  // reasoning_effort is always removed from the outgoing payload; Copilot only
  // understands output_config.effort. Values "low"/"medium"/"high" are valid for
  // both OpenAI and Copilot. "auto" and other OpenAI-only values are dropped.
  if (sanitized.reasoning_effort !== null && sanitized.reasoning_effort !== undefined) {
    const effort = sanitized.reasoning_effort
    delete sanitized.reasoning_effort
    if (COPILOT_EFFORT_VALUES.has(effort)) {
      sanitized.output_config = { ...sanitized.output_config, effort }
    }
  }

  // Sanitize output_config.effort if already present (e.g. sent directly by client).
  // Strip the field if the value is not in the Copilot-accepted set.
  if (
    sanitized.output_config !== null
    && sanitized.output_config !== undefined
    && sanitized.output_config.effort !== undefined
    && !COPILOT_EFFORT_VALUES.has(sanitized.output_config.effort)
  ) {
    const { effort: _effort, ...rest } = sanitized.output_config
    sanitized.output_config = Object.keys(rest).length > 0 ? rest : undefined
  }

  return sanitized
}

export const createChatCompletions = async (
  payload: ChatCompletionsPayload,
) => {
  if (!state.copilotToken) throw new Error("Copilot token not found")

  const sanitizedPayload = sanitizePayload(payload)

  const enableVision = sanitizedPayload.messages.some(
    (x) =>
      typeof x.content !== "string"
      && x.content?.some((x) => x.type === "image_url"),
  )

  // Agent/user check for X-Initiator header
  // Determine if any message is from an agent ("assistant" or "tool")
  const isAgentCall = sanitizedPayload.messages.some((msg) =>
    ["assistant", "tool"].includes(msg.role),
  )

  // Build headers fresh each call (token may be refreshed between attempts)
  const buildHeaders = (): Record<string, string> => ({
    ...copilotHeaders(state, enableVision),
    "X-Initiator": isAgentCall ? "agent" : "user",
  })

  consola.debug("Sending request to Copilot:", {
    model: sanitizedPayload.model,
    endpoint: `${copilotBaseUrl(state)}/chat/completions`,
  })

  const url = `${copilotBaseUrl(state)}/chat/completions`

  // Request usage stats in the final stream chunk
  const body =
    sanitizedPayload.stream ?
      { ...sanitizedPayload, stream_options: { include_usage: true } }
    : sanitizedPayload

  const bodyString = JSON.stringify(body)

  // Retry on transient network errors (TLS disconnect, connection reset, etc.)
  const maxRetries = 2
  let lastError: unknown
  let response: Response | undefined

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      response = await fetch(url, {
        method: "POST",
        headers: buildHeaders(),
        body: bodyString,
      })
      break
    } catch (error: unknown) {
      lastError = error
      if (attempt < maxRetries) {
        const delay = 1000 * (attempt + 1)
        consola.warn(
          `Network error on attempt ${attempt + 1}/${maxRetries + 1}, retrying in ${delay}ms:`,
          error instanceof Error ? error.message : error,
        )
        await new Promise((r) => setTimeout(r, delay))
      }
    }
  }

  if (!response) {
    throw lastError
  }

  // On 401 (token expired), refresh the Copilot token and retry once
  if (response.status === 401) {
    consola.warn("Copilot token expired, refreshing and retrying...")
    try {
      await refreshCopilotToken()
      response = await fetch(url, {
        method: "POST",
        headers: buildHeaders(),
        body: bodyString,
      })
    } catch (refreshError) {
      consola.error("Failed to refresh token:", refreshError)
      // Fall through to the error handling below
    }
  }

  if (!response.ok) {
    const errorBody = await response.text()

    if (response.status === 400) {
      consola.warn(`400: ${errorBody}`)
    } else {
      consola.error("Failed to create chat completions", {
        status: response.status,
        statusText: response.statusText,
        body: errorBody,
      })
    }

    throw new HTTPError(
      `Failed to create chat completions: ${response.status} ${errorBody}`,
      response,
    )
  }

  if (sanitizedPayload.stream) {
    return events(response)
  }

  return (await response.json()) as ChatCompletionResponse
}

// Streaming types

export interface ChatCompletionChunk {
  id: string
  object: "chat.completion.chunk"
  created: number
  model: string
  choices: Array<Choice>
  system_fingerprint?: string
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    prompt_tokens_details?: {
      cached_tokens: number
    }
    completion_tokens_details?: {
      accepted_prediction_tokens: number
      rejected_prediction_tokens: number
    }
  }
}

interface Delta {
  content?: string | null
  role?: "user" | "assistant" | "system" | "tool"
  tool_calls?: Array<{
    index: number
    id?: string
    type?: "function"
    function?: {
      name?: string
      arguments?: string
    }
  }>
}

interface Choice {
  index: number
  delta: Delta
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null
  logprobs: object | null
}

// Non-streaming types

export interface ChatCompletionResponse {
  id: string
  object: "chat.completion"
  created: number
  model: string
  choices: Array<ChoiceNonStreaming>
  system_fingerprint?: string
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    prompt_tokens_details?: {
      cached_tokens: number
    }
  }
}

interface ResponseMessage {
  role: "assistant"
  content: string | null
  tool_calls?: Array<ToolCall>
}

interface ChoiceNonStreaming {
  index: number
  message: ResponseMessage
  logprobs: object | null
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter"
}

// Payload types

export interface ChatCompletionsPayload {
  messages: Array<Message>
  model: string
  temperature?: number | null
  top_p?: number | null
  max_tokens?: number | null
  stop?: string | Array<string> | null
  n?: number | null
  stream?: boolean | null

  frequency_penalty?: number | null
  presence_penalty?: number | null
  logit_bias?: Record<string, number> | null
  logprobs?: boolean | null
  response_format?: { type: string } | null
  seed?: number | null
  tools?: Array<Tool> | null
  tool_choice?:
    | "none"
    | "auto"
    | "required"
    | { type: "function"; function: { name: string } }
    | null
  user?: string | null

  /** OpenAI reasoning effort (o1/o3 models). Mapped to output_config.effort for Copilot. */
  reasoning_effort?: string | null
  /** Copilot-native output configuration. */
  output_config?: { effort?: string; [key: string]: unknown } | null
}

export interface Tool {
  type: "function"
  function: {
    name: string
    description?: string
    parameters: Record<string, unknown>
  }
}

export interface Message {
  role: "user" | "assistant" | "system" | "tool" | "developer"
  content: string | Array<ContentPart> | null

  name?: string
  tool_calls?: Array<ToolCall>
  tool_call_id?: string
}

export interface ToolCall {
  id: string
  type: "function"
  function: {
    name: string
    arguments: string
  }
}

export type ContentPart = TextPart | ImagePart

export interface TextPart {
  type: "text"
  text: string
}

export interface ImagePart {
  type: "image_url"
  image_url: {
    url: string
    detail?: "low" | "high" | "auto"
  }
}
