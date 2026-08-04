package prompty

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	model "prompty/model"
)

// TestLoadVectors drives every case in spec/vectors/load/load_vectors.json.
//
// The vectors describe their input four different ways — a fixture file, a raw
// document, a pre-parsed frontmatter mapping, or a frontmatter mapping plus
// runtime inputs to validate — so the runner dispatches on which key is present.
func TestLoadVectors(t *testing.T) {
	spec := specRoot(t)
	vectors := loadVectorFile(t, filepath.Join(spec, "vectors", "load", "load_vectors.json"))
	if len(vectors) == 0 {
		t.Fatal("load vectors file is empty")
	}

	executed := 0
	for _, vec := range vectors {
		name := vectorString(vec, "name")
		t.Run(name, func(t *testing.T) {
			runLoadVector(t, spec, vec)
		})
		executed++
	}
	t.Logf("load vectors executed: %d", executed)
}

func runLoadVector(t *testing.T, spec string, vec map[string]interface{}) {
	t.Helper()

	input := vectorMap(vec, "input")
	expected := vectorMap(vec, "expected")

	options := LoadOptions{LookupEnv: envLookupFrom(vectorMap(input, "env"))}

	// Vectors that supply `inputs` are validation cases, not load cases.
	if runtimeInputs, ok := input["inputs"].(map[string]interface{}); ok {
		agent, err := LoadFrontmatter(vectorMap(input, "frontmatter"), promptPath(t), options)
		if err != nil {
			t.Fatalf("load failed: %v", err)
		}
		validated, err := ValidateInputs(agent, runtimeInputs)
		assertValidation(t, expected, validated, err)
		return
	}

	agent, err := loadVectorAgent(t, spec, input, options)

	if wantErr, ok := expected["error"].(string); ok {
		assertVectorError(t, wantErr, expected, err)
		return
	}
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	assertAgentMatches(t, expected, agent)
}

func loadVectorAgent(t *testing.T, spec string, input map[string]interface{}, options LoadOptions) (model.Prompty, error) {
	t.Helper()

	switch {
	case input["fixture"] != nil:
		return LoadWithOptions(filepath.Join(spec, "fixtures", vectorString(input, "fixture")), options)

	case input["frontmatter_raw"] != nil:
		return LoadString(vectorString(input, "frontmatter_raw"), promptPath(t), options)

	case input["frontmatter"] != nil:
		base := promptPath(t)
		// Some vectors ship the contents of the files their frontmatter
		// references; materialise them next to the prompt so ${file:...}
		// resolution stays inside the sandbox.
		if files := vectorMap(input, "files"); len(files) > 0 {
			dir := filepath.Dir(base)
			for name, content := range files {
				writeVectorFile(t, filepath.Join(dir, name), content)
			}
		}
		return LoadFrontmatter(vectorMap(input, "frontmatter"), base, options)

	default:
		t.Fatalf("vector input has no fixture, frontmatter_raw or frontmatter key")
		return model.Prompty{}, nil
	}
}

func writeVectorFile(t *testing.T, path string, content interface{}) {
	t.Helper()
	var data []byte
	switch c := content.(type) {
	case string:
		data = []byte(c)
	default:
		encoded, err := json.Marshal(c)
		if err != nil {
			t.Fatalf("encoding vector file %s: %v", path, err)
		}
		data = encoded
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("writing vector file %s: %v", path, err)
	}
}

// promptPath returns a per-test .prompty path inside a temp directory. The file
// itself need not exist — only its directory anchors ${file:...} resolution.
func promptPath(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "vector.prompty")
}

func assertVectorError(t *testing.T, want string, expected map[string]interface{}, err error) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected error %q, got nil", want)
	}

	// The vectors name error *types* in some cases and message fragments in
	// others; accept either, and require the typed check where one applies.
	switch want {
	case "FileNotFoundError":
		if !errors.Is(err, ErrFileNotFound) {
			t.Fatalf("expected a file-not-found error, got %v", err)
		}
	default:
		if !strings.Contains(strings.ToLower(err.Error()), strings.ToLower(want)) {
			t.Fatalf("expected error containing %q, got %q", want, err.Error())
		}
	}

	if field, ok := expected["error_field"].(string); ok {
		var valueErr *ValueError
		if !errors.As(err, &valueErr) {
			t.Fatalf("expected a *ValueError carrying a field, got %T", err)
		}
		if valueErr.Field != field {
			t.Fatalf("error field: got %q, want %q", valueErr.Field, field)
		}
	}
}

func assertValidation(t *testing.T, expected map[string]interface{}, validated map[string]interface{}, err error) {
	t.Helper()

	if wantErr, ok := expected["error"].(string); ok {
		assertVectorError(t, wantErr, expected, err)
		return
	}
	if err != nil {
		t.Fatalf("validate_inputs failed: %v", err)
	}

	want := vectorMap(expected, "validated_inputs")
	if len(validated) != len(want) {
		t.Fatalf("validated inputs: got %#v, want %#v", validated, want)
	}
	for key, wantValue := range want {
		gotValue, ok := validated[key]
		if !ok {
			t.Fatalf("validated inputs missing key %q", key)
		}
		if ok, why := subsetMatch(wantValue, gotValue); !ok {
			t.Fatalf("validated input %q: %s", key, why)
		}
	}
}

// assertAgentMatches compares the loaded agent against the vector's expectation
// by projecting the agent through the emitted serializer.
func assertAgentMatches(t *testing.T, expected map[string]interface{}, agent model.Prompty) {
	t.Helper()

	expected = normalizeExpectedAgent(t, expected)

	// `kind` is a load-algorithm contract (§4.2 step 7), not a field on the
	// emitted Prompty type, so it is asserted separately.
	if kind, ok := expected["kind"]; ok {
		if kind != "prompt" {
			t.Fatalf("unexpected kind expectation %v", kind)
		}
		delete(expected, "kind")
		assertKindInjected(t)
	}

	actual := saveAgent(agent)
	if ok, why := subsetMatch(expected, actual); !ok {
		t.Fatalf("agent mismatch: %s\nactual: %#v", why, actual)
	}
}

// assertKindInjected verifies the loader stamps kind="prompt" onto the
// normalised document. The emitted model.Prompty has no Kind field, so this is
// the only place the contract is observable.
func assertKindInjected(t *testing.T) {
	t.Helper()
	normalized, err := normalizeFrontmatter(map[string]interface{}{"name": "x"})
	if err != nil {
		t.Fatalf("normalize failed: %v", err)
	}
	if normalized["kind"] != "prompt" {
		t.Fatalf("loader did not inject kind=prompt, got %v", normalized["kind"])
	}
}

// normalizeExpectedAgent rewrites the expectation into the same canonical wire
// shape the loader produces. Vectors spell tool bindings and parameters in the
// authoring form; the loaded agent holds the normalised list form.
func normalizeExpectedAgent(t *testing.T, expected map[string]interface{}) map[string]interface{} {
	t.Helper()

	out := make(map[string]interface{}, len(expected))
	for k, v := range expected {
		out[k] = v
	}
	if tools, ok := out["tools"]; ok && tools != nil {
		normalized, err := normalizeTools(tools)
		if err != nil {
			t.Fatalf("normalizing expected tools: %v", err)
		}
		out["tools"] = normalized
	}
	return out
}
