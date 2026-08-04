// Package anthropic implements the Prompty Executor and Processor contracts
// for the Anthropic Messages API.
//
// As with prompty/openai, everything the spec pins down is a pure function:
// BuildChatRequest turns an agent plus a conversation into a request body with
// no I/O, so the wire contract is verified directly against the shared vectors.
//
// Registration is explicit — importing this package installs nothing. Call
// Register during start-up.
package anthropic

import (
	"encoding/json"
	"strings"

	model "prompty/model"
	wire "prompty/wire"
)

// Version is the anthropic-version header value this provider speaks.
const Version = "2023-06-01"

// DefaultMaxTokens is sent when the agent declares no maxOutputTokens.
// Anthropic rejects a request without max_tokens, so unlike every other option
// this one cannot simply be omitted.
const DefaultMaxTokens = 4096

// DefaultEndpoint is used when neither the connection nor the environment names
// one.
const DefaultEndpoint = "https://api.anthropic.com"

// MessagesPath is the only endpoint this provider calls.
const MessagesPath = "/v1/messages"

// BuildChatRequest builds a POST /v1/messages body (spec §7.5).
//
// Three things differ from the OpenAI shape and all three are load-bearing:
// system and developer messages are hoisted to a top-level `system` field,
// content is always an array of typed blocks rather than a bare string, and
// max_tokens is mandatory.
func BuildChatRequest(agent model.Prompty, messages []model.Message) (map[string]interface{}, error) {
	body := map[string]interface{}{"model": agent.Model.Id}

	if system := extractSystem(messages); system != "" {
		body["system"] = system
	}

	wireMessages := make([]interface{}, 0, len(messages))
	for _, msg := range messages {
		if wire.IsSystemRole(msg.Role) {
			continue
		}
		wireMessages = append(wireMessages, MessageToWire(msg))
	}
	body["messages"] = wireMessages

	applyOptions(body, agent)

	tools, err := ToolsToWire(agent)
	if err != nil {
		return nil, err
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}

	outputConfig, err := wire.AnthropicOutputConfig(wire.Outputs(agent))
	if err != nil {
		return nil, err
	}
	if outputConfig != nil {
		body["output_config"] = outputConfig
	}

	return body, nil
}

// EnableStreaming flips a request body into streaming mode.
func EnableStreaming(body map[string]interface{}) {
	body["stream"] = true
}

// applyOptions merges model options and guarantees max_tokens.
func applyOptions(body map[string]interface{}, agent model.Prompty) {
	maxTokens := int64(DefaultMaxTokens)

	if agent.Model.Options != nil {
		for key, value := range agent.Model.Options.ToWire(wire.DialectAnthropic) {
			if value == nil {
				continue
			}
			if key == "max_tokens" {
				if parsed, ok := asInt64(value); ok {
					maxTokens = parsed
				}
				continue
			}
			if stops, ok := value.([]string); ok && stops == nil {
				continue
			}
			body[key] = value
		}
		wire.MergeAdditionalProperties(body, agent.Model.Options)
	}

	body["max_tokens"] = maxTokens
}

// extractSystem joins every system and developer message with a blank line.
func extractSystem(messages []model.Message) string {
	var parts []string
	for _, msg := range messages {
		if !wire.IsSystemRole(msg.Role) {
			continue
		}
		parts = append(parts, wire.TextOf(msg))
	}
	return strings.Join(parts, "\n\n")
}

// MessageToWire converts one message into an Anthropic message object.
//
// Anthropic has only user and assistant roles, so a tool message is folded into
// a user message whose content is a tool_result block.
func MessageToWire(msg model.Message) map[string]interface{} {
	role := "user"
	if msg.Role == model.RoleAssistant {
		role = "assistant"
	}

	// A batched tool_result array parked by the agent loop is replayed as-is.
	if results, ok := msg.Metadata["tool_results"]; ok && results != nil {
		return map[string]interface{}{"role": role, "content": results}
	}

	if useID, ok := msg.Metadata["tool_use_id"].(string); ok && useID != "" {
		return map[string]interface{}{
			"role": role,
			"content": []interface{}{map[string]interface{}{
				"type":        "tool_result",
				"tool_use_id": useID,
				"content":     wire.TextOf(msg),
			}},
		}
	}

	// An assistant turn that requested tools must be replayed with the exact
	// content blocks Anthropic produced, including the tool_use ids the
	// following tool_result blocks refer to.
	if raw, ok := msg.Metadata["content"]; ok && raw != nil {
		return map[string]interface{}{"role": role, "content": raw}
	}

	content := make([]interface{}, 0, len(msg.Parts))
	for _, part := range wire.InspectParts(msg.Parts) {
		content = append(content, PartToWire(part))
	}
	return map[string]interface{}{"role": role, "content": content}
}

// PartToWire converts one content part into an Anthropic content block.
func PartToWire(part wire.PartSpec) map[string]interface{} {
	switch part.Kind {
	case "text":
		return map[string]interface{}{"type": "text", "text": part.Value}

	case "image":
		// A remote image is referenced by URL; anything else is treated as an
		// inline base64 payload, which is how the emitted ImagePart carries it.
		if strings.HasPrefix(part.Value, "http://") || strings.HasPrefix(part.Value, "https://") {
			return map[string]interface{}{
				"type":   "image",
				"source": map[string]interface{}{"type": "url", "url": part.Value},
			}
		}
		mediaType := part.MediaType
		if mediaType == "" {
			mediaType = "image/png"
		}
		return map[string]interface{}{
			"type": "image",
			"source": map[string]interface{}{
				"type":       "base64",
				"media_type": mediaType,
				"data":       part.Value,
			},
		}

	case "audio":
		// Anthropic accepts neither audio nor arbitrary files as content
		// blocks. Degrading to a labelled text placeholder keeps the turn
		// valid and tells the model what it is missing, which is better than
		// silently dropping the part.
		return map[string]interface{}{
			"type": "text", "text": "[audio content not supported by Anthropic]",
		}

	case "file":
		return map[string]interface{}{
			"type": "text", "text": "[file content not supported by Anthropic]",
		}

	default:
		return map[string]interface{}{"type": "text", "text": part.Value}
	}
}

// ToolsToWire projects the agent's function tools into Anthropic tool
// definitions: name and description at the top level, parameters under
// input_schema rather than OpenAI's nested `function.parameters`.
//
// Two deliberate choices differ from the canonical Rust provider, both to keep
// the two providers in this runtime consistent with spec §7.1.3:
//   - only function tools are projected, because a host-executed MCP or OpenAPI
//     tool has no dispatcher the model could reach;
//   - bound parameters are stripped, because a binding exists precisely so the
//     host and not the model supplies that value.
func ToolsToWire(agent model.Prompty) ([]interface{}, error) {
	tools := wire.FunctionTools(agent)
	if len(tools) == 0 {
		return nil, nil
	}

	out := make([]interface{}, 0, len(tools))
	for _, tool := range tools {
		definition := map[string]interface{}{"name": tool.Name}
		if tool.Description != "" {
			definition["description"] = tool.Description
		}

		schema, err := wire.ParametersSchema(tool.ModelVisibleParameters(), false)
		if err != nil {
			return nil, err
		}
		fillMissingArrayItems(schema)
		definition["input_schema"] = schema

		out = append(out, definition)
	}
	return out, nil
}

// fillMissingArrayItems gives every array schema an explicit `items`.
// Anthropic's schema validator rejects a bare {"type":"array"}, so an
// unspecified element type is materialised as a string rather than omitted.
func fillMissingArrayItems(schema map[string]interface{}) {
	if schema["type"] == "array" {
		if _, ok := schema["items"]; !ok {
			schema["items"] = map[string]interface{}{"type": "string"}
		}
	}
	if properties, ok := schema["properties"].(map[string]interface{}); ok {
		for _, value := range properties {
			if nested, ok := value.(map[string]interface{}); ok {
				fillMissingArrayItems(nested)
			}
		}
	}
	if items, ok := schema["items"].(map[string]interface{}); ok {
		fillMissingArrayItems(items)
	}
	if branches, ok := schema["anyOf"].([]interface{}); ok {
		for _, branch := range branches {
			if nested, ok := branch.(map[string]interface{}); ok {
				fillMissingArrayItems(nested)
			}
		}
	}
}

func asInt64(v interface{}) (int64, bool) {
	switch t := v.(type) {
	case int:
		return int64(t), true
	case int32:
		return int64(t), true
	case int64:
		return t, true
	case float32:
		return int64(t), true
	case float64:
		return int64(t), true
	case json.Number:
		parsed, err := t.Int64()
		return parsed, err == nil
	default:
		return 0, false
	}
}
