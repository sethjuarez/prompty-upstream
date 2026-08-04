package prompty

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
)

// toStringKey renders an arbitrary YAML mapping key as a string.
func toStringKey(k interface{}) string {
	if s, ok := k.(string); ok {
		return s
	}
	return fmt.Sprint(k)
}

// decodeJSONValue parses JSON into plain Go values while preserving the
// integer/float distinction that encoding/json normally destroys.
//
// This matters for rendering: Jinja2 prints the integer 30 as "30" but the float
// 30.0 as "30.0". A naive json.Unmarshal turns every number into float64, so a
// ${file:config.json} default of 5 would render as "5.0". Decoding with
// UseNumber and then narrowing keeps `5` an integer and `5.0` a float, matching
// the reference Python/Rust runtimes.
func decodeJSONValue(data []byte) (interface{}, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v interface{}
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	return narrowJSONNumbers(v), nil
}

// narrowJSONNumbers converts json.Number values in a decoded tree into int64 or
// float64, recursing through maps and slices.
func narrowJSONNumbers(v interface{}) interface{} {
	switch t := v.(type) {
	case json.Number:
		if i, err := strconv.ParseInt(t.String(), 10, 64); err == nil {
			return i
		}
		if f, err := t.Float64(); err == nil {
			return f
		}
		return t.String()
	case map[string]interface{}:
		out := make(map[string]interface{}, len(t))
		for k, val := range t {
			out[k] = narrowJSONNumbers(val)
		}
		return out
	case []interface{}:
		out := make([]interface{}, len(t))
		for i, val := range t {
			out[i] = narrowJSONNumbers(val)
		}
		return out
	default:
		return v
	}
}

// inferKind maps a scalar frontmatter value to a Property kind (spec §2.7,
// §4.4). Unknown shapes fall back to "string", which the template engine can
// always stringify.
func inferKind(v interface{}) string {
	switch v.(type) {
	case nil:
		return "string"
	case bool:
		return "boolean"
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return "integer"
	case float32, float64:
		return "float"
	case string:
		return "string"
	case []interface{}:
		return "array"
	case map[string]interface{}:
		return "object"
	default:
		return "string"
	}
}

// sortedAttrKeys returns the map keys in sorted order so serialized role marker
// attributes are deterministic.
func sortedAttrKeys(m map[string]interface{}) []string {
	if len(m) == 0 {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
