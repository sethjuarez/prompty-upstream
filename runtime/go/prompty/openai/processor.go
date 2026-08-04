package openai

import (
	"encoding/json"
	"strings"

	model "prompty/model"
	wire "prompty/wire"
)

// Processor extracts a clean, typed result from a raw OpenAI response
// (spec §8.1).
//
// The result shape is the loop's continue/stop signal:
//
//	[]model.ToolCall   the model asked for tools; the turn continues
//	string             a final text answer
//	map / slice        a decoded structured answer, when the agent declares outputs
//	[]float64          an embedding vector
//
// The zero value is ready to use.
type Processor struct{}

// NewProcessor returns a processor.
func NewProcessor() *Processor { return &Processor{} }

// Process implements model.Processor.
func (p *Processor) Process(agent model.Prompty, response interface{}) (interface{}, error) {
	raw, ok := asMap(response)
	if !ok {
		return nil, wire.NewProviderError("openai", "process", 0, "", "", errUnexpectedResponse(response))
	}

	switch apiType := wire.APIType(agent); apiType {
	case APITypeChat, APITypeAgent:
		return processChat(agent, raw)
	case APITypeResponses:
		return processResponses(agent, raw)
	case APITypeEmbedding:
		return processEmbedding(raw), nil
	case APITypeImage:
		return processImage(raw), nil
	default:
		return nil, &wire.SchemaError{Message: "unsupported OpenAI apiType: " + apiType}
	}
}

// processChat reads a chat.completion response.
//
// Precedence is tool calls, then a refusal, then content. A refusal outranks
// content because when a model refuses it sets content to null and puts the
// explanation in `refusal`; returning "" there would hide the reason from the
// caller.
func processChat(agent model.Prompty, raw map[string]interface{}) (interface{}, error) {
	message := firstChoiceMessage(raw)
	if message == nil {
		return "", nil
	}

	if calls := toolCallsFromChat(message); len(calls) > 0 {
		return calls, nil
	}

	if refusal, ok := message["refusal"].(string); ok && refusal != "" {
		return refusal, nil
	}

	content, _ := message["content"].(string)
	if hasOutputs(agent) {
		return decodeStructured(content), nil
	}
	return content, nil
}

// processResponses reads a Responses API response.
func processResponses(agent model.Prompty, raw map[string]interface{}) (interface{}, error) {
	output, _ := raw["output"].([]interface{})

	var calls []model.ToolCall
	var text strings.Builder

	for _, item := range output {
		entry, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		switch entry["type"] {
		case "function_call":
			calls = append(calls, model.ToolCall{
				// call_id is the identifier a function_call_output must echo;
				// `id` identifies the output item itself and is not accepted.
				Id:        stringField(entry, "call_id", "id"),
				Name:      stringField(entry, "name"),
				Arguments: stringField(entry, "arguments"),
			})
		case "message":
			content, _ := entry["content"].([]interface{})
			for _, block := range content {
				blockMap, ok := block.(map[string]interface{})
				if !ok {
					continue
				}
				if blockMap["type"] == "output_text" {
					text.WriteString(stringField(blockMap, "text"))
				}
			}
		}
	}

	if len(calls) > 0 {
		return calls, nil
	}

	answer := text.String()
	if answer == "" {
		// output_text is the flattened convenience field; fall back to it when
		// the structured output array yielded nothing.
		answer = stringField(raw, "output_text")
	}
	if hasOutputs(agent) {
		return decodeStructured(answer), nil
	}
	return answer, nil
}

// processEmbedding returns one vector for a single input and a slice of vectors
// for a batch, matching the shape of what was sent.
func processEmbedding(raw map[string]interface{}) interface{} {
	data, _ := raw["data"].([]interface{})
	vectors := make([][]float64, 0, len(data))
	for _, item := range data {
		entry, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		values, _ := entry["embedding"].([]interface{})
		vector := make([]float64, 0, len(values))
		for _, value := range values {
			vector = append(vector, toFloat(value))
		}
		vectors = append(vectors, vector)
	}

	if len(vectors) == 1 {
		return vectors[0]
	}
	return vectors
}

// processImage returns the URL of the generated image, or its base64 payload
// when the request asked for inline data.
func processImage(raw map[string]interface{}) interface{} {
	data, _ := raw["data"].([]interface{})
	if len(data) == 0 {
		return ""
	}
	entry, ok := data[0].(map[string]interface{})
	if !ok {
		return ""
	}
	if url := stringField(entry, "url"); url != "" {
		return url
	}
	return stringField(entry, "b64_json")
}

func firstChoiceMessage(raw map[string]interface{}) map[string]interface{} {
	choices, _ := raw["choices"].([]interface{})
	if len(choices) == 0 {
		return nil
	}
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return nil
	}
	message, _ := choice["message"].(map[string]interface{})
	return message
}

// toolCallsFromChat reads the tool_calls array, preserving request order.
func toolCallsFromChat(message map[string]interface{}) []model.ToolCall {
	rawCalls, _ := message["tool_calls"].([]interface{})
	if len(rawCalls) == 0 {
		return nil
	}
	calls := make([]model.ToolCall, 0, len(rawCalls))
	for _, item := range rawCalls {
		entry, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		function, _ := entry["function"].(map[string]interface{})
		call := model.ToolCall{Id: stringField(entry, "id")}
		if function != nil {
			call.Name = stringField(function, "name")
			call.Arguments = stringField(function, "arguments")
		}
		calls = append(calls, call)
	}
	return calls
}

// UsageOf reads token usage from a raw chat, responses or embedding response.
// Both field spellings are accepted because the two APIs disagree.
func UsageOf(raw map[string]interface{}) (model.InvocationUsage, bool) {
	usage, ok := raw["usage"].(map[string]interface{})
	if !ok {
		return model.InvocationUsage{}, false
	}
	return usageFromMap(usage), true
}

func usageFromMap(usage map[string]interface{}) model.InvocationUsage {
	input := int64(toFloat(firstPresent(usage, "prompt_tokens", "input_tokens")))
	output := int64(toFloat(firstPresent(usage, "completion_tokens", "output_tokens")))
	total := int64(toFloat(firstPresent(usage, "total_tokens")))
	if total == 0 {
		total = input + output
	}
	return model.InvocationUsage{InputTokens: input, OutputTokens: output, TotalTokens: total}
}

// hasOutputs reports whether the agent declares structured outputs.
func hasOutputs(agent model.Prompty) bool {
	for _, spec := range wire.Outputs(agent) {
		if spec.Name != "" {
			return true
		}
	}
	return false
}

// decodeStructured parses a structured answer while keeping raw JSON semantics.
//
// A payload that is not a JSON object or array is returned unchanged. That is
// required by the shared process vector chat_structured_invalid_json: a model
// that answers in prose instead of JSON should surface that prose to the
// caller, not a parse error the caller cannot act on.
func decodeStructured(text string) interface{} {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return text
	}
	switch trimmed[0] {
	case '{', '[':
	default:
		return text
	}

	decoder := json.NewDecoder(strings.NewReader(trimmed))
	decoder.UseNumber()
	var decoded interface{}
	if err := decoder.Decode(&decoded); err != nil {
		return text
	}
	return narrowNumbers(decoded)
}

// narrowNumbers turns json.Number into int64 or float64 so an integer stays an
// integer. Structured output is frequently fed straight back into a template,
// where 72 and 72.0 render differently.
func narrowNumbers(v interface{}) interface{} {
	switch t := v.(type) {
	case json.Number:
		if i, err := t.Int64(); err == nil {
			return i
		}
		if f, err := t.Float64(); err == nil {
			return f
		}
		return t.String()
	case map[string]interface{}:
		for k, value := range t {
			t[k] = narrowNumbers(value)
		}
		return t
	case []interface{}:
		for i, value := range t {
			t[i] = narrowNumbers(value)
		}
		return t
	default:
		return v
	}
}
