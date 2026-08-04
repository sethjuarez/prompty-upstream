package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	model "prompty/model"
	wire "prompty/wire"
)

// Processor extracts a clean, typed result from a raw Anthropic response
// (spec §8.1). The zero value is ready to use.
type Processor struct{}

// NewProcessor returns a processor.
func NewProcessor() *Processor { return &Processor{} }

// Process implements model.Processor.
func (p *Processor) Process(agent model.Prompty, response interface{}) (interface{}, error) {
	raw, ok := asMap(response)
	if !ok {
		return nil, wire.NewProviderError("anthropic", "process", 0, "", "",
			fmt.Errorf("expected a JSON object response, got %T", response))
	}

	content, _ := raw["content"].([]interface{})

	var (
		calls []model.ToolCall
		text  strings.Builder
	)

	for _, item := range content {
		block, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		switch block["type"] {
		case "text":
			// Consecutive text blocks are one continuous answer, so they are
			// concatenated with no separator: the shared process vector
			// anthropic_multiple_text_blocks depends on it.
			text.WriteString(stringField(block, "text"))

		case "tool_use":
			calls = append(calls, model.ToolCall{
				Id:        stringField(block, "id"),
				Name:      stringField(block, "name"),
				Arguments: encodeToolInput(block["input"]),
			})
		}
	}

	if len(calls) > 0 {
		return calls, nil
	}

	answer := text.String()
	if hasOutputs(agent) {
		return decodeStructured(answer), nil
	}
	return answer, nil
}

// ProcessContext is the cancellation-aware form of Process.
func (p *Processor) ProcessContext(ctx context.Context, agent model.Prompty, response interface{}) (interface{}, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return p.Process(agent, response)
}

// encodeToolInput serialises Anthropic's decoded tool input back to the JSON
// string the provider-neutral ToolCall carries. Anthropic sends arguments as an
// object where OpenAI sends a string, and the runtime normalises on the string
// so one dispatcher serves both.
func encodeToolInput(input interface{}) string {
	if input == nil {
		return "{}"
	}
	encoded, err := json.Marshal(input)
	if err != nil {
		return "{}"
	}
	return string(encoded)
}

// UsageOf reads token usage from a raw response.
func UsageOf(raw map[string]interface{}) (model.InvocationUsage, bool) {
	usage, ok := raw["usage"].(map[string]interface{})
	if !ok {
		return model.InvocationUsage{}, false
	}
	return usageFromMap(usage), true
}

func usageFromMap(usage map[string]interface{}) model.InvocationUsage {
	input := int64(toFloat(usage["input_tokens"]))
	output := int64(toFloat(usage["output_tokens"]))
	return model.InvocationUsage{
		InputTokens:  input,
		OutputTokens: output,
		TotalTokens:  input + output,
	}
}

func hasOutputs(agent model.Prompty) bool {
	for _, spec := range wire.Outputs(agent) {
		if spec.Name != "" {
			return true
		}
	}
	return false
}

// decodeStructured parses a structured answer while keeping raw JSON semantics.
// A payload that is not a JSON object or array is returned unchanged, matching
// the OpenAI processor and the shared invalid-JSON vector.
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

// asMap coerces a decoded provider response into a string-keyed map without
// panicking on an unexpected shape.
func asMap(v interface{}) (map[string]interface{}, bool) {
	switch t := v.(type) {
	case nil:
		return nil, false
	case map[string]interface{}:
		return t, true
	case *map[string]interface{}:
		if t == nil {
			return nil, false
		}
		return *t, true
	case []byte:
		var decoded map[string]interface{}
		if err := json.Unmarshal(t, &decoded); err != nil {
			return nil, false
		}
		return decoded, true
	case string:
		var decoded map[string]interface{}
		if err := json.Unmarshal([]byte(t), &decoded); err != nil {
			return nil, false
		}
		return decoded, true
	default:
		return nil, false
	}
}

func stringField(m map[string]interface{}, keys ...string) string {
	for _, key := range keys {
		if v, ok := m[key].(string); ok && v != "" {
			return v
		}
	}
	return ""
}

func toFloat(v interface{}) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case float32:
		return float64(t)
	case int:
		return float64(t)
	case int32:
		return float64(t)
	case int64:
		return float64(t)
	case json.Number:
		f, err := t.Float64()
		if err != nil {
			return 0
		}
		return f
	default:
		return 0
	}
}
