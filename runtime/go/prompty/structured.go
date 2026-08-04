package prompty

import (
	"encoding/json"
	"strings"

	model "prompty/model"
	wire "prompty/wire"
)

// Structured output maps an agent's declared `outputs` onto a provider's JSON
// Schema contract and decodes what comes back (spec §7.1.6, §8.1).
//
// The schema is always strict: additionalProperties is false and every declared
// output is listed in `required`. Outputs the author did not mark required are
// made nullable instead of being omitted from `required`, because the OpenAI
// strict contract has no notion of an absent key — a model that wants to skip
// one answers null.

// OutputSchema returns the JSON Schema for the agent's declared outputs, or nil
// when the agent declares none.
func OutputSchema(agent model.Prompty) (map[string]interface{}, error) {
	schema, err := wire.OutputsSchema(wire.Outputs(agent))
	if err != nil {
		return nil, err
	}
	return schema, nil
}

// HasOutputs reports whether the agent declares structured outputs.
func HasOutputs(agent model.Prompty) bool {
	for _, raw := range agent.Outputs {
		if spec, ok := wire.InspectProperty(raw); ok && spec.Name != "" {
			return true
		}
	}
	return false
}

// StructuredOutputName is the schema name both providers are given. It is fixed
// rather than derived from the agent name so a request body stays stable when
// an agent is renamed.
const StructuredOutputName = wire.StructuredOutputName

// ChatResponseFormat builds the OpenAI chat-completions `response_format`
// value, or nil when the agent declares no outputs.
func ChatResponseFormat(agent model.Prompty) (map[string]interface{}, error) {
	return wire.ChatResponseFormat(wire.Outputs(agent))
}

// ResponsesTextFormat builds the OpenAI Responses `text` value, or nil when the
// agent declares no outputs. The Responses API flattens the schema one level
// and hoists `strict` out of the schema envelope.
func ResponsesTextFormat(agent model.Prompty) (map[string]interface{}, error) {
	return wire.ResponsesTextFormat(wire.Outputs(agent))
}

// AnthropicOutputConfig builds the Anthropic `output_config` value, or nil when
// the agent declares no outputs.
func AnthropicOutputConfig(agent model.Prompty) (map[string]interface{}, error) {
	return wire.AnthropicOutputConfig(wire.Outputs(agent))
}

// DecodeStructured decodes a model's text answer into a structured value.
//
// Raw JSON semantics are preserved: objects stay map[string]interface{},
// arrays stay []interface{}, and integers stay integers rather than collapsing
// to float64, so a value round-tripped back into a prompt renders as it was
// written.
//
// A payload that is not valid JSON is returned unchanged as the original
// string. That is deliberate and is what the shared process vector
// chat_structured_invalid_json requires: a refusal or an apology is more useful
// to a caller than a parse error, and the caller can still tell the two apart by
// type-asserting the result.
func DecodeStructured(text string) interface{} {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return text
	}
	// Only object and array payloads are candidates. Without this guard a plain
	// answer like "42" or "null" would silently become a number or nil.
	switch trimmed[0] {
	case '{', '[':
	default:
		return text
	}
	decoded, err := decodeJSONValue([]byte(trimmed))
	if err != nil {
		return text
	}
	return decoded
}

// DecodeStructuredInto unmarshals a model's text answer into a caller-supplied
// Go value. It is the typed counterpart to DecodeStructured for hosts that have
// a struct to fill, and unlike DecodeStructured it reports a parse failure.
func DecodeStructuredInto(text string, target interface{}) error {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return newValueError("Structured output is empty")
	}
	if err := json.Unmarshal([]byte(trimmed), target); err != nil {
		return &ValueError{
			Message:    "Structured output is not valid JSON for the requested type: " + err.Error(),
			Constraint: "json_schema",
			Err:        err,
		}
	}
	return nil
}

// StructuredResult applies the agent's output contract to a text answer: agents
// with declared outputs get the decoded value, agents without get the text.
func StructuredResult(agent model.Prompty, text string) interface{} {
	if !HasOutputs(agent) {
		return text
	}
	return DecodeStructured(text)
}
