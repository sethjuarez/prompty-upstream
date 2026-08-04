package wire

import (
	model "prompty/model"
)

// DefaultAPIType is the apiType assumed when an agent does not declare one.
const DefaultAPIType = "chat"

// APIType returns the agent's declared API type as a plain string.
//
// The emitted model.Model.ApiType is a pointer to an unexported named string
// type, so it cannot be named outside the model package. Converting the
// dereferenced value to string is legal and keeps that unexported type out of
// every public signature here.
func APIType(agent model.Prompty) string {
	if agent.Model.ApiType == nil {
		return DefaultAPIType
	}
	if value := string(*agent.Model.ApiType); value != "" {
		return value
	}
	return DefaultAPIType
}

// ModelID returns the agent's model identifier, or fallback when it is unset.
// For Azure deployments the model id doubles as the deployment name.
func ModelID(agent model.Prompty, fallback string) string {
	if agent.Model.Id != "" {
		return agent.Model.Id
	}
	return fallback
}

// Provider returns the agent's provider key, or "" when unset.
func Provider(agent model.Prompty) string {
	if agent.Model.Provider == nil {
		return ""
	}
	return *agent.Model.Provider
}

// FunctionToolSpec is the provider-neutral view of a declared function tool.
// Only function tools reach a provider request; MCP, OpenAPI, custom and
// nested-Prompty tools are host concerns and are filtered out.
type FunctionToolSpec struct {
	Name        string
	Description string
	Parameters  []PropertySpec
	Strict      bool
	// Bindings are the host-supplied parameter injections. Bound parameters are
	// stripped from the wire schema so the model never sees — or fills — them.
	Bindings []model.Binding
}

// BoundNames returns the set of parameter names covered by a binding.
func (t FunctionToolSpec) BoundNames() map[string]bool {
	if len(t.Bindings) == 0 {
		return nil
	}
	out := make(map[string]bool, len(t.Bindings))
	for _, binding := range t.Bindings {
		out[binding.Name] = true
	}
	return out
}

// ModelVisibleParameters returns the parameters the provider should be told
// about: everything except bound parameters (§7.1.3).
func (t FunctionToolSpec) ModelVisibleParameters() []PropertySpec {
	bound := t.BoundNames()
	if len(bound) == 0 {
		return t.Parameters
	}
	out := make([]PropertySpec, 0, len(t.Parameters))
	for _, param := range t.Parameters {
		if !bound[param.Name] {
			out = append(out, param)
		}
	}
	return out
}

// FunctionTools projects the agent's declared function tools. Non-function
// tools and unrecognised entries are skipped rather than erroring, because an
// agent may legitimately mix host-executed tools with model-visible ones.
func FunctionTools(agent model.Prompty) []FunctionToolSpec {
	if len(agent.Tools) == 0 {
		return nil
	}
	out := make([]FunctionToolSpec, 0, len(agent.Tools))
	for _, raw := range agent.Tools {
		if spec, ok := inspectFunctionTool(raw); ok {
			out = append(out, spec)
		}
	}
	return out
}

func inspectFunctionTool(v interface{}) (FunctionToolSpec, bool) {
	switch t := v.(type) {
	case model.FunctionTool:
		spec := FunctionToolSpec{
			Name:       t.Name,
			Parameters: InspectProperties(t.Parameters),
			Bindings:   t.Bindings,
		}
		if t.Description != nil {
			spec.Description = *t.Description
		}
		if t.Strict != nil {
			spec.Strict = *t.Strict
		}
		return spec, true
	case *model.FunctionTool:
		if t == nil {
			return FunctionToolSpec{}, false
		}
		return inspectFunctionTool(*t)
	default:
		return FunctionToolSpec{}, false
	}
}

// Outputs projects the agent's declared output properties.
func Outputs(agent model.Prompty) []PropertySpec {
	return InspectProperties(agent.Outputs)
}

// ConnectionMap flattens the agent's connection into a string-keyed map.
//
// model.Model.Connection is interface{} because the connection is a
// discriminated union; the emitted variants all implement Save, which is the
// only shape-independent way to read them. A connection supplied as a raw map
// (a host that built the agent by hand) is returned as-is.
func ConnectionMap(agent model.Prompty) map[string]interface{} {
	return connectionValueMap(agent.Model.Connection)
}

func connectionValueMap(conn interface{}) map[string]interface{} {
	switch c := conn.(type) {
	case nil:
		return map[string]interface{}{}
	case map[string]interface{}:
		return c
	case interface {
		Save(*model.SaveContext) map[string]interface{}
	}:
		if saved := c.Save(model.NewSaveContext()); saved != nil {
			return saved
		}
		return map[string]interface{}{}
	default:
		return map[string]interface{}{}
	}
}

// ConnectionString reads a string field from a connection map, trying each key
// in order. Connection spellings differ across sources (apiKey vs api_key), so
// callers pass the accepted aliases.
func ConnectionString(conn map[string]interface{}, keys ...string) string {
	for _, key := range keys {
		if v, ok := conn[key].(string); ok && v != "" {
			return v
		}
	}
	return ""
}

// AdditionalProperties returns the agent's model.options.additionalProperties,
// or nil. These are provider-specific escape-hatch keys merged into a request.
func AdditionalProperties(agent model.Prompty) map[string]interface{} {
	if agent.Model.Options == nil {
		return nil
	}
	return agent.Model.Options.AdditionalProperties
}
