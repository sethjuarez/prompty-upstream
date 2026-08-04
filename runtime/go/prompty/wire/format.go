package wire

// StructuredOutputName is the schema name both providers are given for an
// agent's declared outputs. It is fixed rather than derived from the agent name
// so a request body stays byte-stable when an agent is renamed.
const StructuredOutputName = "structured_output"

// ChatResponseFormat builds the OpenAI chat-completions `response_format`
// value for a set of declared outputs, or nil when there are none (§7.1.6).
func ChatResponseFormat(outputs []PropertySpec) (map[string]interface{}, error) {
	schema, err := OutputsSchema(outputs)
	if err != nil || schema == nil {
		return nil, err
	}
	return map[string]interface{}{
		"type": "json_schema",
		"json_schema": map[string]interface{}{
			"name":   StructuredOutputName,
			"strict": true,
			"schema": schema,
		},
	}, nil
}

// ResponsesTextFormat builds the OpenAI Responses `text` value, or nil.
//
// The Responses API flattens the envelope one level and hoists `strict` out to
// sit beside `type` rather than inside the schema wrapper.
func ResponsesTextFormat(outputs []PropertySpec) (map[string]interface{}, error) {
	schema, err := OutputsSchema(outputs)
	if err != nil || schema == nil {
		return nil, err
	}
	return map[string]interface{}{
		"format": map[string]interface{}{
			"type":   "json_schema",
			"name":   StructuredOutputName,
			"schema": schema,
			"strict": true,
		},
	}, nil
}

// AnthropicOutputConfig builds the Anthropic `output_config` value, or nil.
//
// Anthropic's schema subset is the plain one, not OpenAI's strict dialect: only
// genuinely required outputs are listed in `required` and optional ones stay
// absent rather than becoming nullable unions. Applying OpenAI's strict rules
// here would force the model to emit an explicit null for every optional field.
func AnthropicOutputConfig(outputs []PropertySpec) (map[string]interface{}, error) {
	if len(outputs) == 0 {
		return nil, nil
	}
	schema, err := ParametersSchema(outputs, false)
	if err != nil {
		return nil, err
	}
	schema["additionalProperties"] = false
	return map[string]interface{}{
		"format": map[string]interface{}{
			"type":   "json_schema",
			"schema": schema,
		},
	}, nil
}
