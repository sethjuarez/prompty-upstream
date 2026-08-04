package wire

import (
	"errors"
	"fmt"
)

// ErrSchema is the sentinel behind every schema projection failure, so callers
// can branch with errors.Is without importing a concrete type.
var ErrSchema = errors.New("prompty/wire: cannot project property to JSON Schema")

// SchemaError reports a property that has no valid JSON Schema projection.
type SchemaError struct {
	Message  string
	Property string
}

func (e *SchemaError) Error() string {
	if e.Property == "" {
		return e.Message
	}
	return fmt.Sprintf("%s (property %q)", e.Message, e.Property)
}

func (e *SchemaError) Unwrap() error { return ErrSchema }

func newSchemaError(property, format string, args ...any) *SchemaError {
	return &SchemaError{Message: fmt.Sprintf(format, args...), Property: property}
}

// KindToJSONType maps a Prompty property kind to its JSON Schema type name
// (spec §7.1.3). The second result is false for kinds that have no direct type
// keyword, such as "union", where the composition keyword carries the meaning.
func KindToJSONType(kind string) (string, bool) {
	switch kind {
	case "string":
		return "string", true
	case "integer":
		return "integer", true
	case "float", "number":
		return "number", true
	case "boolean":
		return "boolean", true
	case "array":
		return "array", true
	case "object":
		return "object", true
	default:
		return "", false
	}
}

// PropertySchema projects one property into a JSON Schema fragment.
//
// strict propagates OpenAI strict-mode semantics into nested object and array
// schemas: under strict mode every declared field is listed in `required` and
// optional fields become nullable unions instead, because the OpenAI structured
// output contract has no notion of an absent key.
func PropertySchema(spec PropertySpec, strict bool) (map[string]interface{}, error) {
	schema := map[string]interface{}{}

	if jsonType, ok := KindToJSONType(spec.Kind); ok {
		schema["type"] = jsonType
	}
	if spec.Description != "" {
		schema["description"] = spec.Description
	}
	if len(spec.EnumValues) > 0 {
		schema["enum"] = append([]interface{}(nil), spec.EnumValues...)
	}

	switch spec.Kind {
	case "array":
		// A bare {"type": "array"} is emitted when items is unspecified; that is
		// a legitimate "any element" schema, not an error.
		if spec.Items != nil {
			items, err := PropertySchema(*spec.Items, strict)
			if err != nil {
				return nil, err
			}
			schema["items"] = items
		}

	case "object":
		// Likewise a bare {"type": "object"} when no fields are declared.
		if len(spec.Properties) > 0 {
			nested := map[string]interface{}{}
			required := []interface{}{}
			for _, field := range spec.Properties {
				if field.Name == "" {
					continue
				}
				fieldSchema, err := propertySchemaOptional(field, !field.Required, strict)
				if err != nil {
					return nil, err
				}
				nested[field.Name] = fieldSchema
				if field.Required {
					required = append(required, field.Name)
				}
			}
			schema["properties"] = nested
			if len(required) > 0 {
				schema["required"] = required
			}
			schema["additionalProperties"] = false
		}

	case "union":
		hasOneOf, hasAnyOf := len(spec.OneOf) > 0, len(spec.AnyOf) > 0
		switch {
		case hasOneOf && !hasAnyOf:
			// oneOf demands exactly-one-match validation, which neither the
			// OpenAI nor the Anthropic structured-output subset implements.
			// Failing loudly beats silently downgrading it to anyOf.
			return nil, newSchemaError(spec.Name, "union properties using oneOf are not supported by provider structured output; use anyOf")
		case hasAnyOf && !hasOneOf:
			branches := make([]interface{}, 0, len(spec.AnyOf))
			for _, branch := range spec.AnyOf {
				branchSchema, err := PropertySchema(branch, strict)
				if err != nil {
					return nil, err
				}
				branches = append(branches, branchSchema)
			}
			schema["anyOf"] = branches
		default:
			return nil, newSchemaError(spec.Name, "union property must declare exactly one of oneOf or anyOf with at least one branch")
		}
	}

	if spec.Nullable {
		addNullability(schema)
	}
	return schema, nil
}

// propertySchemaOptional applies strict-mode nullability to an optional field.
func propertySchemaOptional(spec PropertySpec, optional, strict bool) (map[string]interface{}, error) {
	schema, err := PropertySchema(spec, strict)
	if err != nil {
		return nil, err
	}
	if strict && optional && !spec.Nullable {
		addNullability(schema)
	}
	return schema, nil
}

// addNullability widens a schema to also admit null, using whichever spelling
// fits what is already there: a scalar `type` becomes a two-element type array,
// an existing `anyOf` gains a null branch, and anything else is wrapped.
func addNullability(schema map[string]interface{}) {
	if jsonType, ok := schema["type"].(string); ok {
		schema["type"] = []interface{}{jsonType, "null"}
	} else if branches, ok := schema["anyOf"].([]interface{}); ok {
		schema["anyOf"] = append(branches, map[string]interface{}{"type": "null"})
	} else if len(schema) > 0 {
		inner := make(map[string]interface{}, len(schema))
		for k, v := range schema {
			inner[k] = v
			delete(schema, k)
		}
		schema["anyOf"] = []interface{}{inner, map[string]interface{}{"type": "null"}}
	}

	// An enum that does not already admit null would reject the null the type
	// widening just allowed, so keep the two consistent.
	if enumValues, ok := schema["enum"].([]interface{}); ok {
		for _, v := range enumValues {
			if v == nil {
				return
			}
		}
		schema["enum"] = append(enumValues, nil)
	}
}

// ParametersSchema projects a function tool's parameter list into the object
// schema that goes in `function.parameters` (§7.1.3).
//
// Under strict mode every parameter is listed in `required` — optional ones are
// made nullable by propertySchemaOptional instead. additionalProperties is left
// to the caller because the chat and responses dialects attach it in different
// places.
func ParametersSchema(params []PropertySpec, strict bool) (map[string]interface{}, error) {
	properties := map[string]interface{}{}
	required := []interface{}{}

	for _, param := range params {
		if param.Name == "" {
			continue
		}
		paramSchema, err := propertySchemaOptional(param, !param.Required, strict)
		if err != nil {
			return nil, err
		}
		properties[param.Name] = paramSchema
		if strict || param.Required {
			required = append(required, param.Name)
		}
	}

	schema := map[string]interface{}{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema, nil
}

// OutputsSchema projects agent.Outputs into the strict object schema used for
// structured output (§7.1.6).
//
// Structured output is always strict: every declared output is listed in
// `required` and additionalProperties is false. Outputs that are not marked
// required become nullable so the model can still decline to fill them.
// Returns nil when the agent declares no outputs.
func OutputsSchema(outputs []PropertySpec) (map[string]interface{}, error) {
	if len(outputs) == 0 {
		return nil, nil
	}

	properties := map[string]interface{}{}
	required := make([]interface{}, 0, len(outputs))

	for _, output := range outputs {
		if output.Name == "" {
			continue
		}
		outputSchema, err := propertySchemaOptional(output, !output.Required, true)
		if err != nil {
			return nil, err
		}
		properties[output.Name] = outputSchema
		required = append(required, output.Name)
	}
	if len(required) == 0 {
		return nil, nil
	}

	return map[string]interface{}{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}, nil
}
