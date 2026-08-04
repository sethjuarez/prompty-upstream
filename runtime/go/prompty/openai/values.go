package openai

import (
	"encoding/json"
	"fmt"
)

// Shared value-shape helpers. Provider responses arrive as generic decoded
// JSON, so every read has to tolerate a missing or mistyped field without
// panicking — a malformed provider response must become an error, never a
// crash in a host process.

// asMap coerces a decoded provider response into a string-keyed map. It accepts
// the already-decoded map, a []byte or string of JSON, and anything that
// marshals, so an executor may hand over either its parsed body or the raw one.
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

func errUnexpectedResponse(v interface{}) error {
	return fmt.Errorf("expected a JSON object response, got %T", v)
}

// stringField returns the first key that holds a non-empty string.
func stringField(m map[string]interface{}, keys ...string) string {
	for _, key := range keys {
		if v, ok := m[key].(string); ok && v != "" {
			return v
		}
	}
	return ""
}

// firstPresent returns the value of the first key that exists.
func firstPresent(m map[string]interface{}, keys ...string) interface{} {
	for _, key := range keys {
		if v, ok := m[key]; ok {
			return v
		}
	}
	return nil
}

// toFloat widens any numeric shape a JSON decoder may produce. Both
// encoding/json's float64 default and the UseNumber path are covered so callers
// do not have to know which decoder produced the value.
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
