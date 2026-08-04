package anthropic_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	prompty "prompty"
	model "prompty/model"
)

// specRoot walks up from the test's working directory to the repository's
// spec/ directory, so the shared vectors are found wherever the module sits.
func specRoot(t *testing.T) string {
	t.Helper()

	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		candidate := filepath.Join(dir, "spec")
		if info, err := os.Stat(filepath.Join(candidate, "vectors")); err == nil && info.IsDir() {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skip("shared spec/ directory not found; skipping shared-vector tests")
			return ""
		}
		dir = parent
	}
}

// loadVectors reads a shared vector file that holds a top-level JSON array.
func loadVectors(t *testing.T, relative string) []map[string]interface{} {
	t.Helper()

	path := filepath.Join(specRoot(t), "vectors", filepath.FromSlash(relative))
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var decoded []map[string]interface{}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return decoded
}

// agentFromVector builds a real agent through the runtime's own loader, so the
// vectors exercise the same normalisation and hydration path a .prompty file
// takes rather than a test-only shortcut.
func agentFromVector(t *testing.T, input map[string]interface{}) model.Prompty {
	t.Helper()

	modelSpec := map[string]interface{}{}
	if id, ok := input["model_id"].(string); ok {
		modelSpec["id"] = id
	}
	if provider, ok := input["provider"].(string); ok {
		modelSpec["provider"] = provider
	}
	if apiType, ok := input["apiType"].(string); ok {
		modelSpec["apiType"] = apiType
	}
	if options, ok := input["options"].(map[string]interface{}); ok && len(options) > 0 {
		modelSpec["options"] = options
	}
	if connection, ok := input["connection"]; ok {
		modelSpec["connection"] = connection
	}

	data := map[string]interface{}{
		"name":         "vector",
		"instructions": "",
		"model":        modelSpec,
	}
	if tools, ok := input["tools"].([]interface{}); ok && len(tools) > 0 {
		data["tools"] = tools
	}
	if outputs, ok := input["outputs"].([]interface{}); ok && len(outputs) > 0 {
		data["outputs"] = outputs
	}

	agent, err := prompty.LoadFrontmatter(data, t.TempDir(), prompty.LoadOptions{})
	if err != nil {
		t.Fatalf("load agent from vector: %v", err)
	}
	return agent
}

// messagesFromVector converts the vector message spelling into emitted
// messages. The vectors name a non-text part's payload `value`, whereas the
// emitted parts call it `source`, so the mapping is explicit here rather than
// going through the emitted loader.
func messagesFromVector(t *testing.T, raw interface{}) []model.Message {
	t.Helper()

	list, ok := raw.([]interface{})
	if !ok {
		return nil
	}
	messages := make([]model.Message, 0, len(list))
	for _, item := range list {
		entry, ok := item.(map[string]interface{})
		if !ok {
			t.Fatalf("vector message is not an object: %T", item)
		}
		role, _ := entry["role"].(string)
		messages = append(messages, model.Message{
			Role:     model.Role(role),
			Parts:    partsFromVector(t, entry["content"]),
			Metadata: mapOrNil(entry["metadata"]),
		})
	}
	return messages
}

func partsFromVector(t *testing.T, raw interface{}) []interface{} {
	t.Helper()

	switch content := raw.(type) {
	case string:
		return []interface{}{model.TextPart{Kind: "text", Value: content}}
	case []interface{}:
		parts := make([]interface{}, 0, len(content))
		for _, item := range content {
			entry, ok := item.(map[string]interface{})
			if !ok {
				t.Fatalf("vector content part is not an object: %T", item)
			}
			kind, _ := entry["kind"].(string)
			value, _ := entry["value"].(string)
			mediaType := optionalString(entry, "mediaType")
			switch kind {
			case "text":
				parts = append(parts, model.TextPart{Kind: "text", Value: value})
			case "image":
				parts = append(parts, model.ImagePart{
					Kind: "image", Source: value,
					MediaType: mediaType, Detail: optionalString(entry, "detail"),
				})
			case "audio":
				parts = append(parts, model.AudioPart{Kind: "audio", Source: value, MediaType: mediaType})
			case "file":
				parts = append(parts, model.FilePart{Kind: "file", Source: value, MediaType: mediaType})
			default:
				t.Fatalf("unknown vector content kind %q", kind)
			}
		}
		return parts
	default:
		return nil
	}
}

func optionalString(m map[string]interface{}, key string) *string {
	if v, ok := m[key].(string); ok && v != "" {
		return &v
	}
	return nil
}

func mapOrNil(v interface{}) map[string]interface{} {
	if m, ok := v.(map[string]interface{}); ok {
		return m
	}
	return nil
}

// normalizeJSON round-trips a value through JSON so both sides of a comparison
// share one numeric representation. Without it an int32 option and the float64
// a JSON vector decodes to would never compare equal even though the bytes on
// the wire are identical.
func normalizeJSON(t *testing.T, v interface{}) interface{} {
	t.Helper()

	encoded, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal for comparison: %v", err)
	}
	var decoded interface{}
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("unmarshal for comparison: %v", err)
	}
	return decoded
}

func assertJSONEqual(t *testing.T, name string, got, want interface{}) {
	t.Helper()

	normalizedGot := normalizeJSON(t, got)
	normalizedWant := normalizeJSON(t, want)
	if reflect.DeepEqual(normalizedGot, normalizedWant) {
		return
	}
	gotPretty, _ := json.MarshalIndent(normalizedGot, "", "  ")
	wantPretty, _ := json.MarshalIndent(normalizedWant, "", "  ")
	t.Errorf("%s mismatch\n--- got ---\n%s\n--- want ---\n%s", name, gotPretty, wantPretty)
}
