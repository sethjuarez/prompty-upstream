package prompty

import "sort"

// Default template configuration when `template` is omitted (spec §2.8).
const (
	DefaultFormatKind = "jinja2"
	DefaultParserKind = "prompty"
)

// normalizeFrontmatter rewrites the authored frontmatter into the exact wire
// shape the emitted loaders in prompty/model expect.
//
// The .prompty format accepts several equivalent spellings for the same thing
// (a dictionary of inputs vs a list of properties, a scalar shorthand vs a full
// Property, a bindings map vs a list of Binding). The emitted loaders only
// understand the canonical list forms, so normalisation happens here rather than
// in generated code.
//
// The input map is mutated in place and also returned for convenience.
func normalizeFrontmatter(data map[string]interface{}) (map[string]interface{}, error) {
	if data == nil {
		data = map[string]interface{}{}
	}

	if v, ok := data["inputs"]; ok && v != nil {
		props, err := normalizeProperties(v, "inputs")
		if err != nil {
			return nil, err
		}
		data["inputs"] = props
	}
	if v, ok := data["outputs"]; ok && v != nil {
		props, err := normalizeProperties(v, "outputs")
		if err != nil {
			return nil, err
		}
		data["outputs"] = props
	}
	if v, ok := data["tools"]; ok && v != nil {
		tools, err := normalizeTools(v)
		if err != nil {
			return nil, err
		}
		data["tools"] = tools
	}

	tmpl, err := normalizeTemplate(data["template"])
	if err != nil {
		return nil, err
	}
	data["template"] = tmpl

	// .prompty files always produce a prompt agent (spec §4.2 step 7).
	data["kind"] = "prompt"
	return data, nil
}

// normalizeTemplate applies the template defaults and rejects the v1 bare-string
// spelling.
//
// spec §2.8 documents `template: jinja2` as shorthand, but the shared load
// vector `template_string_invalid` makes it an error in v2 and the vectors are
// the normative tie-breaker. Both format and parser default independently, so
// `template: {format: {kind: mustache}}` still gets the prompty parser.
func normalizeTemplate(v interface{}) (map[string]interface{}, error) {
	switch t := v.(type) {
	case nil:
		return map[string]interface{}{
			"format": map[string]interface{}{"kind": DefaultFormatKind},
			"parser": map[string]interface{}{"kind": DefaultParserKind},
		}, nil
	case string:
		return nil, &ValueError{
			Message:    "Invalid template format: template must be an object with format and parser, not the string \"" + t + "\"",
			Field:      "template",
			Constraint: "object",
		}
	case map[string]interface{}:
		out := make(map[string]interface{}, len(t)+2)
		for k, val := range t {
			out[k] = val
		}
		out["format"] = normalizeKindConfig(out["format"], DefaultFormatKind)
		out["parser"] = normalizeKindConfig(out["parser"], DefaultParserKind)
		return out, nil
	default:
		return nil, &ValueError{
			Message:    "Invalid template format: template must be an object with format and parser",
			Field:      "template",
			Constraint: "object",
		}
	}
}

// normalizeKindConfig accepts `format: jinja2`, `format: {kind: jinja2}` or a
// missing value and always produces `{kind: ...}`.
func normalizeKindConfig(v interface{}, defaultKind string) map[string]interface{} {
	switch t := v.(type) {
	case string:
		return map[string]interface{}{"kind": t}
	case map[string]interface{}:
		out := make(map[string]interface{}, len(t)+1)
		for k, val := range t {
			out[k] = val
		}
		if kind, ok := out["kind"].(string); !ok || kind == "" {
			out["kind"] = defaultKind
		}
		return out
	default:
		return map[string]interface{}{"kind": defaultKind}
	}
}

// normalizeProperties converts any accepted inputs/outputs spelling into the
// canonical list of property maps.
//
// Accepted forms:
//
//	[{name: x, kind: string}]      canonical list
//	{x: {kind: string}}            dictionary of property objects
//	{x: "Jane"}                    dictionary of scalar shorthands (§2.7)
//	["x", "y"]                     list of bare names
//
// Dictionary forms are emitted in sorted key order so a load is deterministic.
func normalizeProperties(v interface{}, field string) ([]interface{}, error) {
	switch t := v.(type) {
	case []interface{}:
		out := make([]interface{}, 0, len(t))
		for i, item := range t {
			prop, err := normalizePropertyEntry("", item)
			if err != nil {
				return nil, err
			}
			if prop == nil {
				return nil, newValueError("Invalid %s entry at index %d", field, i)
			}
			out = append(out, prop)
		}
		return out, nil
	case map[string]interface{}:
		names := make([]string, 0, len(t))
		for name := range t {
			names = append(names, name)
		}
		sort.Strings(names)
		out := make([]interface{}, 0, len(names))
		for _, name := range names {
			prop, err := normalizePropertyEntry(name, t[name])
			if err != nil {
				return nil, err
			}
			out = append(out, prop)
		}
		return out, nil
	default:
		return nil, newValueError("Invalid %s: expected a list or a mapping", field)
	}
}

// propertySpecFields are the keys that mark a mapping as a Property definition
// rather than a plain object value. This is the disambiguation for the
// dictionary input form: `x: {kind: string}` declares a property, while
// `x: {a: 1}` is an object-kinded property whose default is that object
// (spec §2.7, §4.4).
var propertySpecFields = [...]string{
	"kind", "default", "description", "required", "nullable", "example",
	"enumValues", "items", "properties", "oneOf", "anyOf", "name",
}

func looksLikePropertySpec(m map[string]interface{}) bool {
	for _, key := range propertySpecFields {
		if _, ok := m[key]; ok {
			return true
		}
	}
	return false
}

// normalizePropertyEntry produces one canonical property map. name is the
// dictionary key when the caller came from the mapping form, and empty when the
// entry already carries its own name.
func normalizePropertyEntry(name string, v interface{}) (map[string]interface{}, error) {
	switch t := v.(type) {
	case map[string]interface{}:
		// In the dictionary form a mapping with no Property fields is an object
		// literal, so it becomes the default of an object-kinded property.
		if name != "" && !looksLikePropertySpec(t) {
			return map[string]interface{}{"name": name, "kind": "object", "default": t}, nil
		}
		out := make(map[string]interface{}, len(t)+2)
		for k, val := range t {
			out[k] = val
		}
		if name != "" {
			out["name"] = name
		}
		if kind, ok := out["kind"].(string); !ok || kind == "" {
			// Infer from the default when the author omitted the kind.
			out["kind"] = inferKind(out["default"])
		}
		if err := normalizeNestedProperties(out); err != nil {
			return nil, err
		}
		return out, nil
	case string:
		// A bare string in a list is a property name; in a mapping it is a
		// scalar default (spec §2.7 shorthand).
		if name == "" {
			return map[string]interface{}{"name": t, "kind": "string"}, nil
		}
		return map[string]interface{}{"name": name, "kind": "string", "default": t}, nil
	case nil:
		if name == "" {
			return nil, newValueError("Property entry must be a mapping or a name")
		}
		return map[string]interface{}{"name": name, "kind": "string"}, nil
	default:
		if name == "" {
			return nil, newValueError("Property entry must be a mapping or a name")
		}
		return map[string]interface{}{"name": name, "kind": inferKind(t), "default": t}, nil
	}
}

// normalizeNestedProperties recurses into the composite property kinds so a
// dictionary spelling works at any depth.
func normalizeNestedProperties(prop map[string]interface{}) error {
	if items, ok := prop["items"]; ok && items != nil {
		normalized, err := normalizePropertyEntry("", items)
		if err != nil {
			return err
		}
		prop["items"] = normalized
	}
	for _, key := range []string{"properties", "oneOf", "anyOf"} {
		v, ok := prop[key]
		if !ok || v == nil {
			continue
		}
		normalized, err := normalizeProperties(v, key)
		if err != nil {
			return err
		}
		prop[key] = normalized
	}
	return nil
}

// normalizeTools canonicalises the tool list, including each tool's parameters
// and bindings.
func normalizeTools(v interface{}) ([]interface{}, error) {
	var entries []interface{}
	switch t := v.(type) {
	case []interface{}:
		entries = t
	case map[string]interface{}:
		names := make([]string, 0, len(t))
		for name := range t {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			tool, ok := t[name].(map[string]interface{})
			if !ok {
				return nil, newValueError("Invalid tool '%s': expected a mapping", name)
			}
			clone := make(map[string]interface{}, len(tool)+1)
			for k, val := range tool {
				clone[k] = val
			}
			clone["name"] = name
			entries = append(entries, clone)
		}
	default:
		return nil, newValueError("Invalid tools: expected a list or a mapping")
	}

	out := make([]interface{}, 0, len(entries))
	for i, entry := range entries {
		tool, ok := entry.(map[string]interface{})
		if !ok {
			return nil, newValueError("Invalid tool entry at index %d: expected a mapping", i)
		}
		normalized, err := normalizeTool(tool)
		if err != nil {
			return nil, err
		}
		out = append(out, normalized)
	}
	return out, nil
}

func normalizeTool(tool map[string]interface{}) (map[string]interface{}, error) {
	out := make(map[string]interface{}, len(tool))
	for k, v := range tool {
		out[k] = v
	}
	if kind, ok := out["kind"].(string); !ok || kind == "" {
		// The emitted loader dispatches on kind and falls through to CustomTool
		// for anything unrecognised; give it something to dispatch on.
		out["kind"] = "function"
	}

	if params, ok := out["parameters"]; ok && params != nil {
		normalized, err := normalizeToolParameters(params)
		if err != nil {
			return nil, err
		}
		out["parameters"] = normalized
	}
	if bindings, ok := out["bindings"]; ok && bindings != nil {
		normalized, err := normalizeBindings(bindings)
		if err != nil {
			return nil, err
		}
		out["bindings"] = normalized
	}
	return out, nil
}

// normalizeToolParameters accepts the plain list, the dictionary form, and the
// JSON-Schema-ish `{properties: [...]}` wrapper shown in spec §2.9.1.
func normalizeToolParameters(v interface{}) ([]interface{}, error) {
	if m, ok := v.(map[string]interface{}); ok {
		if nested, ok := m["properties"]; ok {
			if nested == nil {
				return []interface{}{}, nil
			}
			return normalizeProperties(nested, "parameters")
		}
	}
	return normalizeProperties(v, "parameters")
}

// normalizeBindings converts `{param: {input: other}}` and `{param: other}` into
// the emitted `[]Binding` wire shape (spec §2.9.1).
func normalizeBindings(v interface{}) ([]interface{}, error) {
	switch t := v.(type) {
	case []interface{}:
		out := make([]interface{}, 0, len(t))
		for i, item := range t {
			m, ok := item.(map[string]interface{})
			if !ok {
				return nil, newValueError("Invalid binding at index %d: expected a mapping", i)
			}
			out = append(out, m)
		}
		return out, nil
	case map[string]interface{}:
		names := make([]string, 0, len(t))
		for name := range t {
			names = append(names, name)
		}
		sort.Strings(names)
		out := make([]interface{}, 0, len(names))
		for _, name := range names {
			switch b := t[name].(type) {
			case map[string]interface{}:
				clone := make(map[string]interface{}, len(b)+1)
				for k, val := range b {
					clone[k] = val
				}
				clone["name"] = name
				out = append(out, clone)
			case string:
				out = append(out, map[string]interface{}{"name": name, "input": b})
			default:
				return nil, newValueError("Invalid binding '%s': expected a mapping or an input name", name)
			}
		}
		return out, nil
	default:
		return nil, newValueError("Invalid bindings: expected a list or a mapping")
	}
}
