package prompty

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	model "prompty/model"
)

// specRoot walks up from the test's working directory until it finds the
// repository's spec/ directory. Tests run from runtime/go/prompty, but the
// module may be vendored or relocated, so the path is discovered rather than
// hard-coded as ../../../spec.
func specRoot(t *testing.T) string {
	t.Helper()

	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		candidate := filepath.Join(dir, "spec")
		if isDir(filepath.Join(candidate, "vectors")) && isDir(filepath.Join(candidate, "fixtures")) {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skipf("shared spec/ directory not found above %s; skipping vector tests", mustGetwd(t))
			return ""
		}
		dir = parent
	}
}

func isDir(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func mustGetwd(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	return dir
}

// loadVectorFile reads a shared vector file. Numbers are narrowed to int64 or
// float64 so an integer in the JSON stays an integer, which matters because
// Jinja2 prints 30 and 30.0 differently.
func loadVectorFile(t *testing.T, path string) []map[string]interface{} {
	t.Helper()

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	decoded, err := decodeJSONValue(raw)
	if err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	list, ok := decoded.([]interface{})
	if !ok {
		t.Fatalf("%s: expected a JSON array at the top level", path)
	}

	out := make([]map[string]interface{}, 0, len(list))
	for i, item := range list {
		m, ok := item.(map[string]interface{})
		if !ok {
			t.Fatalf("%s: vector %d is not an object", path, i)
		}
		out = append(out, m)
	}
	return out
}

func vectorString(m map[string]interface{}, key string) string {
	s, _ := m[key].(string)
	return s
}

func vectorMap(m map[string]interface{}, key string) map[string]interface{} {
	v, _ := m[key].(map[string]interface{})
	return v
}

func vectorList(m map[string]interface{}, key string) []interface{} {
	v, _ := m[key].([]interface{})
	return v
}

// envLookupFrom builds a LoadOptions.LookupEnv over a fixed table. Vectors carry
// their own environment, and tests must never mutate the real process
// environment: that leaks across parallel tests and across packages.
func envLookupFrom(env map[string]interface{}) func(string) (string, bool) {
	table := make(map[string]string, len(env))
	for k, v := range env {
		table[k] = fmt.Sprint(v)
	}
	return func(name string) (string, bool) {
		v, ok := table[name]
		return v, ok
	}
}

// subsetMatch reports whether actual satisfies the expectation described by
// expected.
//
// Only the keys named in expected are checked, so a vector can assert on a few
// fields without spelling out an entire agent. A nil expectation means "not
// meaningfully present": absent, nil, or a value whose every leaf is empty. That
// last case exists because the emitted Save always writes `model` and
// `enumValues`, even when the source document had neither.
func subsetMatch(expected, actual interface{}) (bool, string) {
	if expected == nil {
		if isEmptyish(actual) {
			return true, ""
		}
		return false, fmt.Sprintf("expected empty, got %#v", actual)
	}

	switch exp := expected.(type) {
	case map[string]interface{}:
		act, ok := actual.(map[string]interface{})
		if !ok {
			return false, fmt.Sprintf("expected an object, got %#v", actual)
		}
		for key, expValue := range exp {
			actValue, present := act[key]
			if !present && expValue != nil {
				return false, fmt.Sprintf("missing key %q", key)
			}
			if ok, why := subsetMatch(expValue, actValue); !ok {
				return false, fmt.Sprintf("%s: %s", key, why)
			}
		}
		return true, ""

	case []interface{}:
		act, ok := actual.([]interface{})
		if !ok {
			return false, fmt.Sprintf("expected an array, got %#v", actual)
		}
		if len(exp) != len(act) {
			return false, fmt.Sprintf("expected %d elements, got %d", len(exp), len(act))
		}
		for i := range exp {
			if ok, why := subsetMatch(exp[i], act[i]); !ok {
				return false, fmt.Sprintf("[%d]: %s", i, why)
			}
		}
		return true, ""

	default:
		if scalarEqual(expected, actual) {
			return true, ""
		}
		return false, fmt.Sprintf("expected %#v, got %#v", expected, actual)
	}
}

// scalarEqual compares scalars across the numeric representations that YAML,
// JSON and Go each prefer (int, int64, float32, float64).
//
// The tolerance is relative and sized for float32: several emitted fields
// (temperature, topP) are *float32, so a YAML 0.7 round-trips as 0.69999999.
func scalarEqual(a, b interface{}) bool {
	if af, aok := asFloat(a); aok {
		if bf, bok := asFloat(b); bok {
			return math.Abs(af-bf) <= 1e-6*math.Max(1, math.Abs(af))
		}
		return false
	}
	return reflect.DeepEqual(a, b)
}

func asFloat(v interface{}) (float64, bool) {
	switch t := v.(type) {
	case int:
		return float64(t), true
	case int32:
		return float64(t), true
	case int64:
		return float64(t), true
	case uint:
		return float64(t), true
	case uint64:
		return float64(t), true
	case float32:
		return float64(t), true
	case float64:
		return t, true
	default:
		return 0, false
	}
}

func isEmptyish(v interface{}) bool {
	switch t := v.(type) {
	case nil:
		return true
	case string:
		return t == ""
	case bool:
		return !t
	case []interface{}:
		return len(t) == 0
	case map[string]interface{}:
		for _, val := range t {
			if !isEmptyish(val) {
				return false
			}
		}
		return true
	default:
		if f, ok := asFloat(v); ok {
			return f == 0
		}
		return false
	}
}

// saveAgent projects an agent through the emitted serializer so vector
// expectations can be matched against the canonical wire shape rather than
// against Go struct internals.
func saveAgent(agent model.Prompty) map[string]interface{} {
	return agent.Save(model.NewSaveContext())
}

// messageText concatenates the text parts of a message.
func messageText(msg model.Message) string {
	out := ""
	for _, part := range msg.Parts {
		switch p := part.(type) {
		case model.TextPart:
			out += p.Value
		case *model.TextPart:
			out += p.Value
		}
	}
	return out
}

// expectedMessageText flattens a vector's expected content list into text.
func expectedMessageText(expected map[string]interface{}) string {
	out := ""
	for _, part := range vectorList(expected, "content") {
		m, ok := part.(map[string]interface{})
		if !ok {
			continue
		}
		if vectorString(m, "kind") == "text" {
			out += vectorString(m, "value")
		}
	}
	return out
}
