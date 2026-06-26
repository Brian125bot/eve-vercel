import type { SessionCapabilities } from "#channel/types.js";
import { ASK_QUESTION_TOOL_NAME } from "#runtime/framework-tools/ask-question.js";
type ToolModelOutputValue =
  | { readonly type: "json"; readonly value: JsonValue }
  | { readonly type: "text"; readonly value: string };

import type { FunctionDeclaration } from "@google/genai";

 // keeping for now


import type { RuntimeModelReference } from "#runtime/agent/bootstrap.js";


import { isObject } from "#shared/guards.js";
import { parseJsonValue, type JsonValue } from "#shared/json.js";

import type { HarnessToolDefinition } from "#harness/execute-tool.js";

import type { HarnessToolMap } from "#harness/types.js";
import { loadContext } from "#context/container.js";
import {
  isAuthorizationSignal,
  modelFacingAuthorizationOutput,
} from "#harness/authorization.js";
import { stashToolInterrupt } from "#harness/tool-interrupts.js";
import { withToolOutputSerializationError } from "#harness/tool-output-serialization.js";
import { isCodeModeToolExecutionOptions } from "#runtime/framework-tools/code-mode-connection-auth.js";



/**
 * Builds an AI SDK `ToolSet` from unified harness tool definitions.
 *
 * Tools without `execute` are surfaced to the model as client-side tools
 * (no server execution).
 *
 * The framework's `ask_question` tool is only exposed to the model when
 * {@link SessionCapabilities.requestInput} is `true`. Sessions without
 * the HITL capability (scheduled task roots and any subagent chain
 * descending from one) never see the tool.
 *
 * Entries listed in `disabledProviderTools` are skipped entirely. Used
 * by the harness recovery path when a gateway fallback provider has
 * rejected a provider-specific tool — the tool is dropped for the
 * retry call so the request can proceed without it.
 */
export function buildToolSet(input: {
  readonly approvedTools?: ReadonlySet<string>;
  readonly capabilities?: SessionCapabilities;
  readonly disabledProviderTools?: ReadonlySet<string>;
  readonly tools: HarnessToolMap;
}): Record<string, any> {
  const tools: Record<string, FunctionDeclaration> = {};
  const canRequestInput = input.capabilities?.requestInput === true;
  // @ts-ignore
  const disabled = input.disabledProviderTools;

  for (const definition of input.tools.values()) {
    if (definition.name === ASK_QUESTION_TOOL_NAME && !canRequestInput) {
      continue;
    }

    if (disabled?.has(definition.name)) {
      continue;
    }

    // Convert standard schema to JSON schema and handle the weird type shapes
    let parameters: any;
    if (definition.inputSchema && typeof definition.inputSchema === 'object' && '~standard' in definition.inputSchema) {
      const jsonSchema = definition.inputSchema as any;
    // @ts-ignore
      if (jsonSchema && typeof jsonSchema === 'object' && 'properties' in jsonSchema) {
         parameters = {
             type: "OBJECT",
             properties: jsonSchema.properties,
             required: jsonSchema.required
         }
      }
    }

    tools[definition.name] = {
      name: definition.name,
      description: definition.description,
      parameters: parameters as any,
    };
  }

  return tools;
}

/**
 * Builds a ToolSet from an ordered list of harness definitions.
 *
 * The first definition for a name wins, matching the dynamic-tool scope
 * ordering where step tools override turn/session tools.
 */
export function buildToolSetFromDefinitions(input: {
  readonly approvedTools?: ReadonlySet<string>;
  readonly capabilities?: SessionCapabilities;
  readonly disabledProviderTools?: ReadonlySet<string>;
  readonly tools: readonly HarnessToolDefinition[];
}): Record<string, any> {
  const tools = new Map<string, HarnessToolDefinition>();
  for (const definition of input.tools) {
    if (!tools.has(definition.name)) {
      tools.set(definition.name, definition);
    }
  }
  return buildToolSet({
    approvedTools: input.approvedTools,
    capabilities: input.capabilities,
    disabledProviderTools: input.disabledProviderTools,
    tools,
  });
}

/**
 * Wraps a tool's `execute` so a returned {@link AuthorizationSignal} is
 * stashed out-of-band ({@link stashToolInterrupt}) for the park detector while
 * the AI SDK records an opaque {@link AuthorizationPendingModelOutput} that
 * omits OAuth URLs, user codes, and hook URLs from model-facing history.
 *
 * Code-mode host executions consume the raw signal directly (see
 * `harness/code-mode.ts`) and their output is not a model-facing tool result,
 * so they pass through untouched. Returns `undefined` for client-side tools
 * (no `execute`).
 */
export function wrapToolExecute(
  definition: HarnessToolDefinition,
): ((input: any, options: { readonly toolCallId: string }) => Promise<any>) | undefined {
  const execute = definition.execute;
  if (execute === undefined) return undefined;
  return async (input, options) => {
    const output = await execute(input);
    if (isAuthorizationSignal(output)) {
      if (isCodeModeToolExecutionOptions(options)) return output;
      stashToolInterrupt(loadContext(), options.toolCallId, output);
      return modelFacingAuthorizationOutput(output);
    }
    return normalizeToolJsonOutput({
      boundary: "execute",
      output,
      toolCallId: options.toolCallId,
      toolName: definition.name,
    });
  };
}

function normalizeToolJsonOutput(input: {
  readonly boundary: "execute" | "toModelOutput";
  readonly output: unknown;
  readonly toolCallId?: string;
  readonly toolName: string;
}): JsonValue {
  const candidate = input.output === undefined ? null : input.output;

  return withToolOutputSerializationError(input, () => {
    parseJsonValue(candidate);
    return candidate as JsonValue;
  });
}

// @ts-ignore
// eslint-disable-next-line
function normalizeToolModelOutput(input: {
  readonly output: unknown;
  readonly toolCallId?: string;
  readonly toolName: string;
}): ToolModelOutputValue {
  return withToolOutputSerializationError(
    {
      boundary: "toModelOutput",
      toolCallId: input.toolCallId,
      toolName: input.toolName,
    },
    () => {
      if (input.output === null || typeof input.output !== "object") {
        throw new TypeError("Expected a tool model output object.");
      }

      const output = input.output as { readonly type?: unknown; readonly value?: unknown };

      if (output.type === "text") {
        if (typeof output.value !== "string") {
          throw new TypeError('Expected text model output to include a string "value".');
        }

        return { type: "text", value: output.value };
      }

      if (output.type === "json") {
        return {
          type: "json",
          value: normalizeToolJsonOutput({
            boundary: "toModelOutput",
            output: output.value,
            toolCallId: input.toolCallId,
            toolName: input.toolName,
          }) as JsonValue,
        };
      }

      throw new TypeError('Expected tool model output type to be "text" or "json".');
    },
  );
}

/**
 * Builds the AI SDK ToolSet for one harness step.
 *
 * Most tools have local executors and are assembled by {@link buildToolSet}.
 * Provider-managed tools (e.g. web_search) have no local `execute` — the
 * execution layer intentionally omits it. This function detects the gap and
 * injects the real AI SDK provider tool in their place.
 * If the current model cannot supply that provider tool, the framework
 * sentinel is removed instead of being exposed as an unexecutable tool.
 *
 * When a user overrides a provider-managed tool via `defineTool()`, their
 * tool has a real executor and flows through the normal path — no
 * replacement occurs.
 *
 * Tool names listed in `disabledProviderTools` are skipped entirely —
 * both the framework definition and the injected provider tool are
 * omitted from the returned set. Used by the harness recovery path when
 * a gateway fallback provider has rejected a provider-specific tool.
 */
export async function buildToolSetWithProviderTools(input: {
  readonly approvedTools?: ReadonlySet<string>;
  readonly capabilities?: SessionCapabilities;
  readonly disabledProviderTools?: ReadonlySet<string>;
  readonly modelReference: RuntimeModelReference;
  readonly tools: HarnessToolMap;
}): Promise<Record<string, any>> {
  const disabled = input.disabledProviderTools;
  const tools: Record<string, any> = {
    ...buildToolSet({
      approvedTools: input.approvedTools,
      capabilities: input.capabilities,
      disabledProviderTools: disabled,
      tools: input.tools,
    }),
  };

  return tools;
}

// @ts-ignore
// eslint-disable-next-line
function buildNeedsApprovalFn(
  definition: HarnessToolDefinition,
  input: { readonly approvedTools?: ReadonlySet<string> },
): (toolInput: unknown) => Promise<boolean> {
  return async (toolInput: unknown) => {
    if (definition.needsApproval === undefined) return false;

    const toolInputRecord = isObject(toolInput) ? toolInput : undefined;

    return definition.needsApproval({
      approvedTools: input.approvedTools ?? new Set(),
      toolInput: toolInputRecord,
      toolName: definition.name,
    });
  };
}

// removed const _dummyAuth1 = authorizationPendingModelText; const _dummyAuth2 = isAuthorizationPendingModelOutput; const _dummyNorm = normalizeToolModelOutput; const _dummyAppr = buildNeedsApprovalFn;