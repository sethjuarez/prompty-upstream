// Package openai implements the Prompty Executor and Processor contracts for
// every OpenAI-compatible endpoint: OpenAI itself, Azure OpenAI and Azure AI
// Foundry.
//
// The package is split so that everything shaped by the spec is a pure
// function. BuildChatRequest and friends turn an agent plus a conversation into
// a request body with no I/O at all, which is what lets them be verified
// directly against the shared wire vectors; the Executor only adds transport.
//
// Registration is explicit. Importing this package has no side effects — call
// Register (or RegisterFoundry) to install the provider into the runtime's
// registries, so a host that never talks to OpenAI never links a client it did
// not ask for.
package openai

import (
	"encoding/json"
	"strings"

	model "prompty/model"
	wire "prompty/wire"
)

// jsonString renders a non-string tool output for the Responses API, which
// requires `output` to be a string. A value that will not marshal degrades to
// its Go rendering rather than dropping the tool result entirely.
func jsonString(v interface{}) string {
	encoded, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(encoded)
}

// API type identifiers understood by this provider.
const (
	APITypeChat      = "chat"
	APITypeAgent     = "agent"
	APITypeResponses = "responses"
	APITypeEmbedding = "embedding"
	APITypeImage     = "image"
)

// Fallback model ids, applied when the agent omits model.id. They match the
// canonical Rust provider so a portable agent behaves identically.
const (
	defaultEmbeddingModel = "text-embedding-ada-002"
	defaultImageModel     = "dall-e-3"
	defaultResponsesModel = "gpt-4o"
)

// BuildRequest dispatches to the builder for the agent's declared API type.
func BuildRequest(agent model.Prompty, messages []model.Message) (map[string]interface{}, error) {
	switch apiType := wire.APIType(agent); apiType {
	case APITypeChat, APITypeAgent:
		return BuildChatRequest(agent, messages)
	case APITypeResponses:
		return BuildResponsesRequest(agent, messages)
	case APITypeEmbedding:
		return BuildEmbeddingRequest(agent, messages), nil
	case APITypeImage:
		return BuildImageRequest(agent, messages), nil
	default:
		return nil, &wire.SchemaError{Message: "unsupported OpenAI apiType: " + apiType}
	}
}

// BuildChatRequest builds a POST /v1/chat/completions body (spec §7.1).
func BuildChatRequest(agent model.Prompty, messages []model.Message) (map[string]interface{}, error) {
	body := map[string]interface{}{
		"model": agent.Model.Id,
	}

	wireMessages := make([]interface{}, 0, len(messages))
	for _, msg := range messages {
		wireMessages = append(wireMessages, MessageToWire(msg))
	}
	body["messages"] = wireMessages

	wire.ApplyOptions(body, agent.Model.Options, wire.DialectOpenAI)

	tools, err := ToolsToWire(agent)
	if err != nil {
		return nil, err
	}
	// The key is omitted entirely rather than sent as an empty array: some
	// OpenAI-compatible gateways treat `tools: []` as "tool choice required"
	// and reject the request.
	if len(tools) > 0 {
		body["tools"] = tools
	}

	responseFormat, err := wire.ChatResponseFormat(wire.Outputs(agent))
	if err != nil {
		return nil, err
	}
	if responseFormat != nil {
		body["response_format"] = responseFormat
	}

	return body, nil
}

// BuildResponsesRequest builds a POST /v1/responses body.
//
// The Responses API differs from chat completions in three ways that matter:
// system and developer messages are hoisted into `instructions`, the
// conversation is called `input`, and tool definitions are flat instead of
// nested under a `function` key.
func BuildResponsesRequest(agent model.Prompty, messages []model.Message) (map[string]interface{}, error) {
	var (
		instructions []string
		input        []interface{}
	)
	for _, msg := range messages {
		if wire.IsSystemRole(msg.Role) {
			instructions = append(instructions, textContent(msg))
			continue
		}
		input = append(input, MessageToResponsesInput(msg))
	}
	if input == nil {
		input = []interface{}{}
	}

	body := map[string]interface{}{
		"model": wire.ModelID(agent, defaultResponsesModel),
		"input": input,
	}
	if len(instructions) > 0 {
		body["instructions"] = strings.Join(instructions, "\n\n")
	}

	wire.ApplyOptions(body, agent.Model.Options, wire.DialectResponses)

	tools, err := ResponsesToolsToWire(agent)
	if err != nil {
		return nil, err
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}

	textFormat, err := wire.ResponsesTextFormat(wire.Outputs(agent))
	if err != nil {
		return nil, err
	}
	if textFormat != nil {
		body["text"] = textFormat
	}

	return body, nil
}

// BuildEmbeddingRequest builds a POST /v1/embeddings body. Only the
// escape-hatch options apply; the chat sampling options are meaningless here.
func BuildEmbeddingRequest(agent model.Prompty, messages []model.Message) map[string]interface{} {
	body := map[string]interface{}{
		"model": wire.ModelID(agent, defaultEmbeddingModel),
		"input": textInput(messages),
	}
	wire.MergeAdditionalProperties(body, agent.Model.Options)
	return body
}

// BuildImageRequest builds a POST /v1/images/generations body.
func BuildImageRequest(agent model.Prompty, messages []model.Message) map[string]interface{} {
	prompt := ""
	switch input := textInput(messages).(type) {
	case string:
		prompt = input
	case []interface{}:
		parts := make([]string, 0, len(input))
		for _, item := range input {
			if text, ok := item.(string); ok {
				parts = append(parts, text)
			}
		}
		prompt = strings.Join(parts, " ")
	}

	body := map[string]interface{}{
		"model":  wire.ModelID(agent, defaultImageModel),
		"prompt": prompt,
	}
	wire.MergeAdditionalProperties(body, agent.Model.Options)
	return body
}

// EnableStreaming flips a chat or responses body into streaming mode.
//
// The chat dialect also asks for usage on the terminal event; without
// stream_options the final chunk carries no token counts and a streamed turn
// would report zero usage where a non-streamed one reports real numbers.
func EnableStreaming(body map[string]interface{}, apiType string) {
	body["stream"] = true
	if apiType == APITypeChat || apiType == APITypeAgent {
		body["stream_options"] = map[string]interface{}{"include_usage": true}
	}
}

// textInput collapses a conversation into the scalar-or-array `input` the
// embedding endpoint accepts: one string for a single message, an array
// otherwise.
func textInput(messages []model.Message) interface{} {
	texts := make([]string, 0, len(messages))
	for _, msg := range messages {
		if text := textContent(msg); text != "" {
			texts = append(texts, text)
		}
	}
	if len(texts) == 1 {
		return texts[0]
	}
	out := make([]interface{}, 0, len(texts))
	for _, text := range texts {
		out = append(out, text)
	}
	return out
}

// MessageToWire converts one message into the chat-completions wire shape.
//
// Metadata is copied onto the message object because that is where the agent
// loop parks the fields OpenAI expects as siblings of `content`: tool_calls on
// an assistant message and tool_call_id on a tool message. role and content are
// excluded so metadata can never overwrite the message's own identity.
func MessageToWire(msg model.Message) map[string]interface{} {
	obj := map[string]interface{}{"role": string(msg.Role)}

	for key, value := range msg.Metadata {
		if key == "role" || key == "content" {
			continue
		}
		obj[key] = value
	}

	if text, ok := allTextContent(msg); ok {
		obj["content"] = text
		return obj
	}

	parts := make([]interface{}, 0, len(msg.Parts))
	for _, part := range wire.InspectParts(msg.Parts) {
		parts = append(parts, PartToWire(part))
	}
	obj["content"] = parts
	return obj
}

// MessageToResponsesInput converts one message into a Responses input item.
func MessageToResponsesInput(msg model.Message) interface{} {
	// A provider-owned function_call item is replayed verbatim: rewriting it
	// would break continuation against a previous_response_id.
	if raw, ok := msg.Metadata["responses_function_call"]; ok && raw != nil {
		return raw
	}

	content := responsesContent(msg)

	if callID, ok := msg.Metadata["tool_call_id"].(string); ok && callID != "" {
		output, ok := content.(string)
		if !ok {
			output = jsonString(content)
		}
		return map[string]interface{}{
			"type":    "function_call_output",
			"call_id": callID,
			"output":  output,
		}
	}

	// The Responses API has no `tool` role; a tool message that is not a
	// function_call_output is presented as user content.
	role := string(msg.Role)
	if msg.Role == model.RoleTool {
		role = string(model.RoleUser)
	}
	return map[string]interface{}{"role": role, "content": content}
}

func responsesContent(msg model.Message) interface{} {
	if text, ok := allTextContent(msg); ok {
		return text
	}
	parts := make([]interface{}, 0, len(msg.Parts))
	for _, part := range wire.InspectParts(msg.Parts) {
		parts = append(parts, PartToWire(part))
	}
	return parts
}

// PartToWire converts one content part into an OpenAI content block.
func PartToWire(part wire.PartSpec) map[string]interface{} {
	switch part.Kind {
	case "text":
		return map[string]interface{}{"type": "text", "text": part.Value}

	case "image":
		image := map[string]interface{}{"url": part.Value}
		if part.Detail != "" {
			image["detail"] = part.Detail
		}
		return map[string]interface{}{"type": "image_url", "image_url": image}

	case "audio":
		return map[string]interface{}{
			"type": "input_audio",
			"input_audio": map[string]interface{}{
				"data":   part.Value,
				"format": wire.MimeToAudioFormat(part.MediaType),
			},
		}

	case "file":
		return map[string]interface{}{
			"type": "file",
			"file": map[string]interface{}{"url": part.Value},
		}

	default:
		return map[string]interface{}{"type": "text", "text": part.Value}
	}
}

// ToolsToWire projects the agent's function tools into chat-completions tool
// definitions. Non-function tools are host concerns and never reach the wire.
func ToolsToWire(agent model.Prompty) ([]interface{}, error) {
	tools := wire.FunctionTools(agent)
	if len(tools) == 0 {
		return nil, nil
	}
	out := make([]interface{}, 0, len(tools))
	for _, tool := range tools {
		definition, err := functionDefinition(tool)
		if err != nil {
			return nil, err
		}
		out = append(out, map[string]interface{}{
			"type":     "function",
			"function": definition,
		})
	}
	return out, nil
}

// ResponsesToolsToWire projects function tools into the flat Responses shape.
func ResponsesToolsToWire(agent model.Prompty) ([]interface{}, error) {
	tools := wire.FunctionTools(agent)
	if len(tools) == 0 {
		return nil, nil
	}
	out := make([]interface{}, 0, len(tools))
	for _, tool := range tools {
		definition, err := functionDefinition(tool)
		if err != nil {
			return nil, err
		}
		definition["type"] = "function"
		out = append(out, definition)
	}
	return out, nil
}

// functionDefinition builds the shared body of a function tool definition.
//
// Bound parameters are stripped: a binding exists so the host, not the model,
// supplies that value, and advertising it would invite the model to fill it
// (spec §7.1.3).
func functionDefinition(tool wire.FunctionToolSpec) (map[string]interface{}, error) {
	definition := map[string]interface{}{"name": tool.Name}
	if tool.Description != "" {
		definition["description"] = tool.Description
	}

	parameters, err := wire.ParametersSchema(tool.ModelVisibleParameters(), tool.Strict)
	if err != nil {
		return nil, err
	}
	definition["parameters"] = parameters

	if tool.Strict {
		definition["strict"] = true
		parameters["additionalProperties"] = false
	}
	return definition, nil
}

func allTextContent(msg model.Message) (string, bool) {
	if len(msg.Parts) == 0 {
		return "", true
	}
	texts := make([]string, 0, len(msg.Parts))
	for _, part := range msg.Parts {
		spec, ok := wire.InspectPart(part)
		if !ok || spec.Kind != "text" {
			return "", false
		}
		texts = append(texts, spec.Value)
	}
	return strings.Join(texts, "\n"), true
}

func textContent(msg model.Message) string {
	text, _ := allTextContent(msg)
	return text
}
