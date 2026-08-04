package prompty

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	model "prompty/model"
)

// promptyAgentFor builds a bare agent with a chosen template format kind.
func promptyAgentFor(t *testing.T, format string) model.Prompty {
	t.Helper()
	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":     "unit",
		"model":    "gpt-4",
		"template": map[string]interface{}{"format": map[string]interface{}{"kind": format}},
	}, filepath.Join(t.TempDir(), "unit.prompty"), LoadOptions{})
	if err != nil {
		t.Fatalf("building agent: %v", err)
	}
	return agent
}

// --- frontmatter splitting (spec §2.2 normative vectors) -------------------

func TestFrontmatterSplitVectors(t *testing.T) {
	cases := []struct {
		name         string
		input        string
		wantName     interface{}
		wantBody     string
		wantNoFields bool
	}{
		{name: "standard", input: "---\nname: test\n---\nHello world", wantName: "test", wantBody: "Hello world"},
		{name: "no frontmatter", input: "Just a prompt with no frontmatter", wantBody: "Just a prompt with no frontmatter", wantNoFields: true},
		{name: "empty frontmatter", input: "---\n---\nBody only", wantBody: "Body only", wantNoFields: true},
		{name: "leading whitespace", input: "  ---\nname: test\n---\nBody", wantName: "test", wantBody: "Body"},
		{name: "plus delimiters", input: "+++\nname: test\n+++\nBody", wantName: "test", wantBody: "Body"},
		{name: "mixed delimiters", input: "---\nname: test\n+++\nBody", wantName: "test", wantBody: "Body"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data, body, err := splitFrontmatter(tc.input)
			if err != nil {
				t.Fatalf("split failed: %v", err)
			}
			if body != tc.wantBody {
				t.Errorf("body: got %q, want %q", body, tc.wantBody)
			}
			if tc.wantNoFields {
				if len(data) != 0 {
					t.Errorf("expected no frontmatter fields, got %#v", data)
				}
				return
			}
			if data["name"] != tc.wantName {
				t.Errorf("name: got %#v, want %#v", data["name"], tc.wantName)
			}
		})
	}
}

func TestFrontmatterMissingClosingDelimiter(t *testing.T) {
	_, _, err := splitFrontmatter("---\nname: test\nno closing delimiter")
	if err == nil {
		t.Fatal("expected an error")
	}
	if !errors.Is(err, ErrValue) {
		t.Fatalf("expected ErrValue, got %v", err)
	}
}

func TestFrontmatterMustBeMapping(t *testing.T) {
	_, _, err := splitFrontmatter("---\n- one\n- two\n---\nBody")
	if err == nil || !strings.Contains(err.Error(), "must be a YAML mapping") {
		t.Fatalf("expected a mapping error, got %v", err)
	}
}

func TestLoadNormalizesCRLF(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, filepath.Join(dir, "crlf.prompty"),
		"---\r\nname: crlf\r\nmodel: gpt-4\r\n---\r\nsystem:\r\nHello\r\n")

	agent, err := Load(path)
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if agent.Instructions == nil || *agent.Instructions != "system:\nHello" {
		t.Fatalf("instructions: got %q", instructionsOf(agent))
	}
}

func TestLoadMissingFile(t *testing.T) {
	_, err := Load(filepath.Join(t.TempDir(), "nope.prompty"))
	if !errors.Is(err, ErrFileNotFound) {
		t.Fatalf("expected ErrFileNotFound, got %v", err)
	}
}

func TestLoadRecordsSourcePath(t *testing.T) {
	dir := t.TempDir()
	path := writeFile(t, filepath.Join(dir, "src.prompty"), "---\nname: src\nmodel: gpt-4\n---\nHi")

	agent, err := Load(path)
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	got, ok := SourcePath(agent)
	if !ok {
		t.Fatal("source path not recorded")
	}
	if !strings.HasSuffix(got, "src.prompty") {
		t.Fatalf("source path: got %q", got)
	}
}

// --- normalization ---------------------------------------------------------

func TestNormalizeDictionaryInputs(t *testing.T) {
	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":  "dict-inputs",
		"model": "gpt-4",
		"inputs": map[string]interface{}{
			"firstName": "Jane",
			"count":     5,
			"ratio":     1.5,
			"active":    true,
			"tags":      []interface{}{"a", "b"},
			"config":    map[string]interface{}{"a": 1},
		},
	}, filepath.Join(t.TempDir(), "p.prompty"), LoadOptions{})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	want := map[string]string{
		"active":    "boolean",
		"config":    "object",
		"count":     "integer",
		"firstName": "string",
		"ratio":     "float",
		"tags":      "array",
	}
	inputs := AgentInputs(agent)
	if len(inputs) != len(want) {
		t.Fatalf("expected %d inputs, got %d", len(want), len(inputs))
	}
	for _, prop := range inputs {
		if prop.Kind != want[prop.Name] {
			t.Errorf("%s: kind %q, want %q", prop.Name, prop.Kind, want[prop.Name])
		}
		if prop.Default == nil {
			t.Errorf("%s: expected the scalar shorthand to become a default", prop.Name)
		}
	}

	// Dictionary order is not defined in YAML, so the loader sorts by name to
	// make loads reproducible.
	if inputs[0].Name != "active" || inputs[len(inputs)-1].Name != "tags" {
		t.Fatalf("expected inputs sorted by name, got %v", inputNames(inputs))
	}
}

func inputNames(props []PropertyView) []string {
	out := make([]string, 0, len(props))
	for _, p := range props {
		out = append(out, p.Name)
	}
	return out
}

func TestNormalizeNestedProperties(t *testing.T) {
	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":  "nested",
		"model": "gpt-4",
		"inputs": []interface{}{
			map[string]interface{}{
				"name": "profile",
				"kind": "object",
				"properties": map[string]interface{}{
					"age":  map[string]interface{}{"kind": "integer"},
					"name": map[string]interface{}{"kind": "string"},
				},
			},
			map[string]interface{}{
				"name":  "scores",
				"kind":  "array",
				"items": map[string]interface{}{"kind": "integer"},
			},
			map[string]interface{}{
				"name":  "either",
				"kind":  "union",
				"anyOf": []interface{}{map[string]interface{}{"kind": "string"}, map[string]interface{}{"kind": "boolean"}},
			},
		},
	}, filepath.Join(t.TempDir(), "p.prompty"), LoadOptions{})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	obj, ok := agent.Inputs[0].(model.ObjectProperty)
	if !ok {
		t.Fatalf("expected an ObjectProperty, got %T", agent.Inputs[0])
	}
	if len(obj.Properties) != 2 {
		t.Fatalf("expected 2 nested properties, got %d", len(obj.Properties))
	}
	// The base emitted loader drops primitive kinds; the runtime hydrates them.
	if child, ok := obj.Properties[0].(model.Property); !ok || child.Kind != "integer" || child.Name != "age" {
		t.Fatalf("nested property not hydrated: %#v", obj.Properties[0])
	}

	arr, ok := agent.Inputs[1].(model.ArrayProperty)
	if !ok {
		t.Fatalf("expected an ArrayProperty, got %T", agent.Inputs[1])
	}
	if item, ok := arr.Items.(model.Property); !ok || item.Kind != "integer" {
		t.Fatalf("array items not hydrated: %#v", arr.Items)
	}

	union, ok := agent.Inputs[2].(model.UnionProperty)
	if !ok {
		t.Fatalf("expected a UnionProperty, got %T", agent.Inputs[2])
	}
	if len(union.AnyOf) != 2 {
		t.Fatalf("expected 2 anyOf branches, got %d", len(union.AnyOf))
	}
	if branch, ok := union.AnyOf[1].(model.Property); !ok || branch.Kind != "boolean" {
		t.Fatalf("union branch not hydrated: %#v", union.AnyOf[1])
	}
}

func TestNormalizeToolBindingsAndParameters(t *testing.T) {
	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":  "tools",
		"model": "gpt-4",
		"tools": []interface{}{
			map[string]interface{}{
				"name": "get_user_orders",
				"kind": "function",
				"parameters": map[string]interface{}{
					"properties": []interface{}{
						map[string]interface{}{"name": "user_id", "kind": "string", "required": true},
						map[string]interface{}{"name": "limit", "kind": "integer", "default": 10},
					},
				},
				"bindings": map[string]interface{}{
					"user_id": "current_user",
					"limit":   map[string]interface{}{"input": "page_size"},
				},
			},
		},
	}, filepath.Join(t.TempDir(), "p.prompty"), LoadOptions{})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	tool, ok := agent.Tools[0].(model.FunctionTool)
	if !ok {
		t.Fatalf("expected a FunctionTool, got %T", agent.Tools[0])
	}

	if len(tool.Parameters) != 2 {
		t.Fatalf("expected 2 parameters, got %d", len(tool.Parameters))
	}
	first, ok := tool.Parameters[0].(model.Property)
	if !ok || first.Name != "user_id" || first.Kind != "string" {
		t.Fatalf("parameter not hydrated: %#v", tool.Parameters[0])
	}
	if first.Required == nil || !*first.Required {
		t.Fatal("expected user_id to be required")
	}

	// The map form of bindings must become the emitted list form, sorted for
	// determinism.
	want := []model.Binding{{Name: "limit", Input: "page_size"}, {Name: "user_id", Input: "current_user"}}
	if len(tool.Bindings) != len(want) {
		t.Fatalf("bindings: got %#v", tool.Bindings)
	}
	for i, b := range want {
		if tool.Bindings[i] != b {
			t.Errorf("binding %d: got %#v, want %#v", i, tool.Bindings[i], b)
		}
	}
}

func TestTemplateDefaultsAreApplied(t *testing.T) {
	t.Run("omitted", func(t *testing.T) {
		agent, err := LoadFrontmatter(map[string]interface{}{"name": "t", "model": "gpt-4"},
			filepath.Join(t.TempDir(), "p.prompty"), LoadOptions{})
		if err != nil {
			t.Fatalf("load failed: %v", err)
		}
		if agent.Template == nil {
			t.Fatal("expected a default template")
		}
		if agent.Template.Format.Kind != "jinja2" || agent.Template.Parser.Kind != "prompty" {
			t.Fatalf("template defaults: %#v", *agent.Template)
		}
	})

	t.Run("partial", func(t *testing.T) {
		agent, err := LoadFrontmatter(map[string]interface{}{
			"name":     "t",
			"model":    "gpt-4",
			"template": map[string]interface{}{"format": map[string]interface{}{"kind": "mustache"}},
		}, filepath.Join(t.TempDir(), "p.prompty"), LoadOptions{})
		if err != nil {
			t.Fatalf("load failed: %v", err)
		}
		if agent.Template.Format.Kind != "mustache" {
			t.Fatalf("format kind: got %q", agent.Template.Format.Kind)
		}
		if agent.Template.Parser.Kind != "prompty" {
			t.Fatalf("parser should default independently, got %q", agent.Template.Parser.Kind)
		}
	})

	t.Run("bare string rejected", func(t *testing.T) {
		_, err := LoadFrontmatter(map[string]interface{}{
			"name": "t", "model": "gpt-4", "template": "jinja2",
		}, filepath.Join(t.TempDir(), "p.prompty"), LoadOptions{})
		if err == nil || !strings.Contains(err.Error(), "Invalid template format") {
			t.Fatalf("expected an invalid-template error, got %v", err)
		}
	})
}

// --- input validation ------------------------------------------------------

func TestValidateInputsPassesThroughUndeclared(t *testing.T) {
	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":   "v",
		"model":  "gpt-4",
		"inputs": []interface{}{map[string]interface{}{"name": "declared", "kind": "string", "default": "d"}},
	}, filepath.Join(t.TempDir(), "p.prompty"), LoadOptions{})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	validated, err := ValidateInputs(agent, map[string]interface{}{"extra": 42})
	if err != nil {
		t.Fatalf("validate failed: %v", err)
	}
	if validated["extra"] != 42 {
		t.Fatalf("undeclared input dropped: %#v", validated)
	}
	if validated["declared"] != "d" {
		t.Fatalf("default not filled: %#v", validated)
	}
}

func TestValidateInputsDoesNotMutateCaller(t *testing.T) {
	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":   "v",
		"model":  "gpt-4",
		"inputs": []interface{}{map[string]interface{}{"name": "topic", "kind": "string", "default": "science"}},
	}, filepath.Join(t.TempDir(), "p.prompty"), LoadOptions{})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	caller := map[string]interface{}{}
	if _, err := ValidateInputs(agent, caller); err != nil {
		t.Fatalf("validate failed: %v", err)
	}
	if len(caller) != 0 {
		t.Fatalf("ValidateInputs mutated the caller's map: %#v", caller)
	}
}

func TestValidateInputsPrefersSuppliedValueOverDefault(t *testing.T) {
	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":   "v",
		"model":  "gpt-4",
		"inputs": []interface{}{map[string]interface{}{"name": "topic", "kind": "string", "default": "science"}},
	}, filepath.Join(t.TempDir(), "p.prompty"), LoadOptions{})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	validated, err := ValidateInputs(agent, map[string]interface{}{"topic": "history"})
	if err != nil {
		t.Fatalf("validate failed: %v", err)
	}
	if validated["topic"] != "history" {
		t.Fatalf("supplied value overridden by default: %#v", validated)
	}
}

// --- registries ------------------------------------------------------------

type stubRenderer struct{ out string }

func (s stubRenderer) Render(model.Prompty, string, map[string]interface{}) (string, error) {
	return s.out, nil
}

func TestRegistryLifecycle(t *testing.T) {
	t.Cleanup(ClearCache)

	if !HasRenderer("jinja2") || !HasRenderer("nunjucks") || !HasRenderer("mustache") {
		t.Fatalf("expected built-in renderers, got %v", RendererKeys())
	}

	if !HasParser("prompty") {
		t.Fatalf("expected the built-in prompty parser, got %v", ParserKeys())
	}
	// Executors and processors are provider packages; nothing is preregistered.
	if len(ExecutorKeys()) != 0 || len(ProcessorKeys()) != 0 {
		t.Fatalf("expected no built-in executors/processors, got %v / %v", ExecutorKeys(), ProcessorKeys())
	}

	RegisterRenderer("stub", stubRenderer{out: "stubbed"})
	got, err := GetRenderer("stub")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}
	if out, _ := got.Render(model.Prompty{}, "", nil); out != "stubbed" {
		t.Fatalf("wrong renderer returned: %q", out)
	}

	UnregisterRenderer("stub")
	if HasRenderer("stub") {
		t.Fatal("unregister did not take effect")
	}

	RegisterRenderer("stub", stubRenderer{})
	ClearCache()
	if HasRenderer("stub") {
		t.Fatal("ClearCache did not reset the registry")
	}
	if !HasRenderer("jinja2") {
		t.Fatal("ClearCache dropped the built-in renderers")
	}
}

func TestNormalizeNullToolPropertiesAsEmpty(t *testing.T) {
	parameters, err := normalizeToolParameters(map[string]interface{}{"properties": nil})
	if err != nil {
		t.Fatalf("normalizeToolParameters returned an error: %v", err)
	}
	if len(parameters) != 0 {
		t.Fatalf("normalizeToolParameters returned %d parameters, want 0", len(parameters))
	}
}

func TestRegistryMissReturnsInvokerError(t *testing.T) {
	_, err := GetRenderer("no-such-engine")
	if !errors.Is(err, ErrInvoker) {
		t.Fatalf("expected ErrInvoker, got %v", err)
	}

	var invokerErr *InvokerError
	if !errors.As(err, &invokerErr) {
		t.Fatalf("expected an *InvokerError, got %T", err)
	}
	if invokerErr.Component != "renderer" || invokerErr.Key != "no-such-engine" {
		t.Fatalf("unexpected error payload: %#v", invokerErr)
	}
	if want := "No renderer registered for key: no-such-engine"; err.Error() != want {
		t.Fatalf("message: got %q, want %q", err.Error(), want)
	}
}

// TestRegistryIsConcurrencySafe is meaningful under -race: it interleaves
// registration, lookup and reset from many goroutines.
func TestRegistryIsConcurrencySafe(t *testing.T) {
	t.Cleanup(ClearCache)

	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(3)
		go func() { defer wg.Done(); RegisterRenderer("concurrent", stubRenderer{}) }()
		go func() { defer wg.Done(); _, _ = GetRenderer("jinja2") }()
		go func() { defer wg.Done(); _ = RendererKeys() }()
	}
	wg.Wait()
}

func TestRenderUnknownEngineIsInvokerError(t *testing.T) {
	agent := promptyAgentFor(t, "handlebars")
	_, err := Render(agent, nil)
	if !errors.Is(err, ErrInvoker) {
		t.Fatalf("expected ErrInvoker, got %v", err)
	}
}

// --- cancellation ----------------------------------------------------------

func TestPipelineHonoursCancellation(t *testing.T) {
	agent := promptyAgentFor(t, "jinja2")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := RenderWithContext(ctx, agent, nil); !errors.Is(err, context.Canceled) {
		t.Errorf("render: expected context.Canceled, got %v", err)
	}
	if _, err := PrepareWithContext(ctx, agent, nil); !errors.Is(err, context.Canceled) {
		t.Errorf("prepare: expected context.Canceled, got %v", err)
	}
	if _, err := ParseWithState(ctx, agent, "system:\nhi", nil); !errors.Is(err, context.Canceled) {
		t.Errorf("parse: expected context.Canceled, got %v", err)
	}
	if _, err := LoadWithContext(ctx, "irrelevant.prompty", LoadOptions{}); !errors.Is(err, context.Canceled) {
		t.Errorf("load: expected context.Canceled, got %v", err)
	}
}
