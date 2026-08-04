package prompty

import (
	"context"
	"path/filepath"
	"regexp"
	"testing"

	model "prompty/model"
)

// TestRenderVectors drives every case in spec/vectors/render/render_vectors.json
// through the real Render pipeline: registry lookup, input validation, rich-kind
// nonce substitution and the registered renderer.
func TestRenderVectors(t *testing.T) {
	spec := specRoot(t)
	vectors := loadVectorFile(t, filepath.Join(spec, "vectors", "render", "render_vectors.json"))
	if len(vectors) == 0 {
		t.Fatal("render vectors file is empty")
	}

	passed := 0
	for _, vec := range vectors {
		name := vectorString(vec, "name")
		t.Run(name, func(t *testing.T) {
			input := vectorMap(vec, "input")
			expected := vectorMap(vec, "expected")

			template := vectorString(input, "template")
			engine := vectorString(input, "engine")
			inputs := vectorMap(input, "inputs")

			agent := buildRenderAgent(t, template, engine, inputs)

			rendered, err := RenderWithContext(context.Background(), agent, inputs)
			if err != nil {
				t.Fatalf("render failed: %v", err)
			}

			if want, ok := expected["rendered"].(string); ok {
				if rendered != want {
					t.Fatalf("rendered mismatch\n want: %q\n  got: %q", want, rendered)
				}
			}
			if pattern, ok := expected["nonce_pattern"].(string); ok {
				re, err := regexp.Compile(pattern)
				if err != nil {
					t.Fatalf("invalid nonce_pattern %q: %v", pattern, err)
				}
				if !re.MatchString(rendered) {
					t.Fatalf("nonce pattern mismatch\n pattern: %s\n     got: %q", pattern, rendered)
				}
			}
		})
		passed++
	}
	t.Logf("render vectors executed: %d", passed)
}

// buildRenderAgent constructs a minimal agent for a render vector. Any input
// whose value carries the `_kind: thread` envelope is declared as a thread
// property so the renderer takes the nonce path.
func buildRenderAgent(t *testing.T, template, engine string, inputs map[string]interface{}) model.Prompty {
	t.Helper()

	var declared []interface{}
	for name, value := range inputs {
		kind := "string"
		if m, ok := value.(map[string]interface{}); ok {
			if k, ok := m["_kind"].(string); ok {
				kind = k
			}
		}
		declared = append(declared, map[string]interface{}{"name": name, "kind": kind})
	}

	data := map[string]interface{}{
		"name":         "render-vector",
		"model":        map[string]interface{}{"id": "test"},
		"instructions": template,
		"template": map[string]interface{}{
			"format": map[string]interface{}{"kind": engine},
			"parser": map[string]interface{}{"kind": "prompty"},
		},
	}
	if declared != nil {
		data["inputs"] = declared
	}

	agent, err := LoadFrontmatter(data, filepath.Join(t.TempDir(), "vector.prompty"), LoadOptions{})
	if err != nil {
		t.Fatalf("building agent: %v", err)
	}
	return agent
}

// TestParseVectors drives every case in spec/vectors/parse/parse_vectors.json.
//
// Vectors carrying `thread_inputs` exercise the production thread-expansion path
// (ParseWithState + expandThreads) rather than a test-local reimplementation, so
// a regression in expansion actually fails this test.
func TestParseVectors(t *testing.T) {
	spec := specRoot(t)
	vectors := loadVectorFile(t, filepath.Join(spec, "vectors", "parse", "parse_vectors.json"))
	if len(vectors) == 0 {
		t.Fatal("parse vectors file is empty")
	}

	agent := parseVectorAgent(t)
	executed := 0

	for _, vec := range vectors {
		name := vectorString(vec, "name")
		t.Run(name, func(t *testing.T) {
			input := vectorMap(vec, "input")
			expected := vectorList(vectorMap(vec, "expected"), "messages")
			rendered := vectorString(input, "rendered")

			state := threadStateFor(rendered, vectorMap(input, "thread_inputs"))

			messages, err := ParseWithState(context.Background(), agent, rendered, state)
			if err != nil {
				t.Fatalf("parse failed: %v", err)
			}

			if len(messages) != len(expected) {
				t.Fatalf("message count mismatch: got %d, want %d\n got: %s",
					len(messages), len(expected), formatMessages(messages))
			}
			for i, raw := range expected {
				exp, ok := raw.(map[string]interface{})
				if !ok {
					t.Fatalf("expected message %d is not an object", i)
				}
				if got, want := string(messages[i].Role), vectorString(exp, "role"); got != want {
					t.Errorf("message %d role: got %q, want %q", i, got, want)
				}
				if got, want := messageText(messages[i]), expectedMessageText(exp); got != want {
					t.Errorf("message %d text: got %q, want %q", i, got, want)
				}
			}
		})
		executed++
	}
	t.Logf("parse vectors executed: %d", executed)
}

// parseVectorAgent returns an agent that only carries the template settings the
// parser needs.
func parseVectorAgent(t *testing.T) model.Prompty {
	t.Helper()
	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":  "parse-vector",
		"model": map[string]interface{}{"id": "test"},
	}, filepath.Join(t.TempDir(), "vector.prompty"), LoadOptions{})
	if err != nil {
		t.Fatalf("building agent: %v", err)
	}
	return agent
}

// threadStateFor rebuilds the render state a real render would have produced:
// it maps each nonce present in the rendered text back to the thread value the
// vector supplies under that nonce's property name.
func threadStateFor(rendered string, threadInputs map[string]interface{}) *RenderState {
	if len(threadInputs) == 0 {
		return nil
	}
	state := newRenderState()
	for _, nonce := range threadNonceRe.FindAllString(rendered, -1) {
		name := noncePropertyName(nonce)
		if value, ok := threadInputs[name]; ok {
			state.put(nonce, value)
		}
	}
	return state
}

// noncePropertyName extracts the property name from
// __PROMPTY_THREAD_<hex>_<name>__.
func noncePropertyName(nonce string) string {
	trimmed := nonce[len(noncePrefix) : len(nonce)-len(nonceSuffix)]
	for i := 0; i < len(trimmed); i++ {
		if trimmed[i] == '_' {
			return trimmed[i+1:]
		}
	}
	return trimmed
}

func formatMessages(messages []model.Message) string {
	out := ""
	for i, msg := range messages {
		out += "\n  [" + string(rune('0'+i)) + "] " + string(msg.Role) + ": " + messageText(msg)
	}
	return out
}
