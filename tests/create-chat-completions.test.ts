import { test, expect, mock, describe } from "bun:test"

import type { ChatCompletionsPayload } from "../src/services/copilot/create-chat-completions"

import { state } from "../src/lib/state"
import {
  createChatCompletions,
  normalizeEffort,
  sanitizePayload,
} from "../src/services/copilot/create-chat-completions"

// Mock state
state.copilotToken = "test-token"
state.vsCodeVersion = "1.0.0"
state.accountType = "individual"

// Helper to mock fetch
const fetchMock = mock(
  (_url: string, opts: { headers: Record<string, string> }) => {
    return {
      ok: true,
      json: () => ({ id: "123", object: "chat.completion", choices: [] }),
      headers: opts.headers,
    }
  },
)
// @ts-expect-error - Mock fetch doesn't implement all fetch properties
;(globalThis as unknown as { fetch: typeof fetch }).fetch = fetchMock

test("sets X-Initiator to agent if tool/assistant present", async () => {
  const payload: ChatCompletionsPayload = {
    messages: [
      { role: "user", content: "hi" },
      { role: "tool", content: "tool call" },
    ],
    model: "gpt-test",
  }
  await createChatCompletions(payload)
  expect(fetchMock).toHaveBeenCalled()
  const headers = (
    fetchMock.mock.calls[0][1] as { headers: Record<string, string> }
  ).headers
  expect(headers["X-Initiator"]).toBe("agent")
})

test("sets X-Initiator to user if only user present", async () => {
  const payload: ChatCompletionsPayload = {
    messages: [
      { role: "user", content: "hi" },
      { role: "user", content: "hello again" },
    ],
    model: "gpt-test",
  }
  await createChatCompletions(payload)
  expect(fetchMock).toHaveBeenCalled()
  const headers = (
    fetchMock.mock.calls[1][1] as { headers: Record<string, string> }
  ).headers
  expect(headers["X-Initiator"]).toBe("user")
})

describe("sanitizePayload", () => {
  test("converts developer role to system", () => {
    const payload: ChatCompletionsPayload = {
      messages: [
        { role: "developer", content: "You are a helpful assistant." },
      ],
      model: "gpt-test",
    }
    const result = sanitizePayload(payload)
    expect(result.messages[0].role).toBe("system")
  })

  test("converts null message content to empty string", () => {
    const payload: ChatCompletionsPayload = {
      messages: [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "fn", arguments: "{}" },
            },
          ],
        },
      ],
      model: "gpt-test",
    }
    const result = sanitizePayload(payload)
    expect(result.messages[0].content).toBe("")
  })

  test("removes empty tools array and tool_choice", () => {
    const payload: ChatCompletionsPayload = {
      messages: [{ role: "user", content: "hi" }],
      model: "gpt-test",
      tools: [],
      tool_choice: "auto",
    }
    const result = sanitizePayload(payload)
    expect(result.tools).toBeUndefined()
    expect(result.tool_choice).toBeUndefined()
  })

  test("preserves non-empty tools array", () => {
    const tools = [
      {
        type: "function" as const,
        function: { name: "myFn", parameters: {} },
      },
    ]
    const payload: ChatCompletionsPayload = {
      messages: [{ role: "user", content: "hi" }],
      model: "gpt-test",
      tools,
      tool_choice: "auto",
    }
    const result = sanitizePayload(payload)
    expect(result.tools).toEqual(tools)
    expect(result.tool_choice).toBe("auto")
  })

  test("strips unsupported response_format type", () => {
    const payload: ChatCompletionsPayload = {
      messages: [{ role: "user", content: "hi" }],
      model: "gpt-test",
      response_format: { type: "text" },
    }
    const result = sanitizePayload(payload)
    expect(result.response_format).toBeUndefined()
  })

  test("preserves supported response_format json_object", () => {
    const payload: ChatCompletionsPayload = {
      messages: [{ role: "user", content: "hi" }],
      model: "gpt-test",
      response_format: { type: "json_object" },
    }
    const result = sanitizePayload(payload)
    expect(result.response_format).toEqual({ type: "json_object" })
  })

  test("does not mutate the original payload", () => {
    const payload: ChatCompletionsPayload = {
      messages: [{ role: "developer", content: null }],
      model: "gpt-test",
      tools: [],
      tool_choice: "none",
      response_format: { type: "text" },
    }
    const result = sanitizePayload(payload)
    // Sanitized result should have all transformations applied
    expect(result.messages[0].role).toBe("system")
    expect(result.messages[0].content).toBe("")
    expect(result.tools).toBeUndefined()
    expect(result.tool_choice).toBeUndefined()
    expect(result.response_format).toBeUndefined()
    // Original should be unchanged
    expect(payload.messages[0].role).toBe("developer")
    expect(payload.messages[0].content).toBeNull()
    expect(payload.tools).toEqual([])
    expect(payload.tool_choice).toBe("none")
    expect(payload.response_format).toEqual({ type: "text" })
  })

  test("maps reasoning_effort string 'medium' to output_config.effort", () => {
    const payload: ChatCompletionsPayload = {
      messages: [{ role: "user", content: "hi" }],
      model: "gpt-test",
      reasoning_effort: "medium",
    }
    const result = sanitizePayload(payload)
    expect(result.reasoning_effort).toBeUndefined()
    expect(result.output_config?.effort).toBe("medium")
  })

  test("drops reasoning_effort 'auto' and does not set output_config.effort", () => {
    const payload: ChatCompletionsPayload = {
      messages: [{ role: "user", content: "hi" }],
      model: "gpt-test",
      reasoning_effort: "auto",
    }
    const result = sanitizePayload(payload)
    expect(result.reasoning_effort).toBeUndefined()
    expect(result.output_config?.effort).toBeUndefined()
  })

  test("maps numeric reasoning_effort 0.5 (float) to 'medium'", () => {
    const payload: ChatCompletionsPayload = {
      messages: [{ role: "user", content: "hi" }],
      model: "gpt-test",
      reasoning_effort: 0.5,
    }
    const result = sanitizePayload(payload)
    expect(result.reasoning_effort).toBeUndefined()
    expect(result.output_config?.effort).toBe("medium")
  })

  test("maps numeric reasoning_effort 0.8 (float) to 'high'", () => {
    const payload: ChatCompletionsPayload = {
      messages: [{ role: "user", content: "hi" }],
      model: "gpt-test",
      reasoning_effort: 0.8,
    }
    const result = sanitizePayload(payload)
    expect(result.reasoning_effort).toBeUndefined()
    expect(result.output_config?.effort).toBe("high")
  })

  test("maps numeric reasoning_effort 2 (integer) to 'high'", () => {
    const payload: ChatCompletionsPayload = {
      messages: [{ role: "user", content: "hi" }],
      model: "gpt-test",
      reasoning_effort: 2,
    }
    const result = sanitizePayload(payload)
    expect(result.reasoning_effort).toBeUndefined()
    expect(result.output_config?.effort).toBe("high")
  })

  test("normalizes numeric output_config.effort 2 to 'high'", () => {
    const payload: ChatCompletionsPayload = {
      messages: [{ role: "user", content: "hi" }],
      model: "gpt-test",
      output_config: { effort: 2 },
    }
    const result = sanitizePayload(payload)
    expect(result.output_config?.effort).toBe("high")
  })

  test("strips invalid string output_config.effort", () => {
    const payload: ChatCompletionsPayload = {
      messages: [{ role: "user", content: "hi" }],
      model: "gpt-test",
      output_config: { effort: "auto" },
    }
    const result = sanitizePayload(payload)
    expect(result.output_config).toBeUndefined()
  })
})

describe("normalizeEffort", () => {
  test("passes through valid string values unchanged", () => {
    expect(normalizeEffort("low")).toBe("low")
    expect(normalizeEffort("medium")).toBe("medium")
    expect(normalizeEffort("high")).toBe("high")
    expect(normalizeEffort("max")).toBe("max")
  })

  test("returns undefined for unsupported strings", () => {
    expect(normalizeEffort("auto")).toBeUndefined()
    expect(normalizeEffort("default")).toBeUndefined()
    expect(normalizeEffort("")).toBeUndefined()
  })

  test("maps float 0-1 scale to string tiers", () => {
    expect(normalizeEffort(0)).toBe("low")
    expect(normalizeEffort(0.1)).toBe("low")
    expect(normalizeEffort(0.33)).toBe("low")
    expect(normalizeEffort(0.5)).toBe("medium")
    expect(normalizeEffort(0.67)).toBe("medium")
    expect(normalizeEffort(0.8)).toBe("high")
    expect(normalizeEffort(1)).toBe("high")
  })

  test("maps integer tiers 2+ to string values", () => {
    expect(normalizeEffort(2)).toBe("high")
    expect(normalizeEffort(3)).toBe("max")
    expect(normalizeEffort(4)).toBe("max")
  })

  test("maps float values between tiers correctly", () => {
    expect(normalizeEffort(1.5)).toBe("high") // between integer 1 and 2
    expect(normalizeEffort(2.5)).toBe("max") // between integer 2 and 3
  })

  test("returns undefined for invalid numeric inputs", () => {
    expect(normalizeEffort(Number.NaN)).toBeUndefined()
    expect(normalizeEffort(Number.POSITIVE_INFINITY)).toBeUndefined()
    expect(normalizeEffort(Number.NEGATIVE_INFINITY)).toBeUndefined()
    expect(normalizeEffort(-1)).toBeUndefined()
    expect(normalizeEffort(-0.5)).toBeUndefined()
  })
})
