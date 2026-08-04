package wire

import (
	model "prompty/model"
)

// Option dialects accepted by ApplyOptions. These are the provider keys the
// emitted ModelOptions.ToWire dispatches on; "responses" is a distinct dialect
// from "openai" because the Responses API renames maxOutputTokens and drops the
// penalties entirely.
const (
	DialectOpenAI    = "openai"
	DialectResponses = "responses"
	DialectAnthropic = "anthropic"
)

// ApplyOptions merges the agent's model options into a request body under the
// given dialect.
//
// Two behaviours are load-bearing:
//
//   - Nil-valued entries are dropped. The emitted ModelOptions.Save
//     unconditionally writes stopSequences even when it is nil, so ToWire
//     reports a nil "stop"; sending that to a provider is a 400.
//   - additionalProperties are merged last and never overwrite a key a mapped
//     option already produced, so an escape-hatch value cannot silently shadow
//     a declared option.
func ApplyOptions(body map[string]interface{}, opts *model.ModelOptions, dialect string) {
	if opts == nil {
		return
	}

	for key, value := range opts.ToWire(dialect) {
		if isNilValue(value) {
			continue
		}
		body[key] = value
	}

	for key, value := range opts.AdditionalProperties {
		if _, exists := body[key]; exists {
			continue
		}
		body[key] = value
	}
}

// MergeAdditionalProperties copies only the escape-hatch options into body.
// The embedding and image endpoints accept none of the mapped chat options, so
// they take this path instead of ApplyOptions.
func MergeAdditionalProperties(body map[string]interface{}, opts *model.ModelOptions) {
	if opts == nil {
		return
	}
	for key, value := range opts.AdditionalProperties {
		if _, exists := body[key]; exists {
			continue
		}
		body[key] = value
	}
}

// isNilValue reports whether a decoded option value is absent. Typed nil slices
// arrive here as non-nil interfaces, so the concrete cases are checked too.
func isNilValue(v interface{}) bool {
	switch t := v.(type) {
	case nil:
		return true
	case []string:
		return t == nil
	case []interface{}:
		return t == nil
	default:
		return false
	}
}
