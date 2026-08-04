package prompty_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	prompty "prompty"
	model "prompty/model"
	openai "prompty/openai"
)

func agentWithOutputs(t *testing.T, outputs []interface{}) model.Prompty {
	t.Helper()

	data := map[string]interface{}{
		"name": "structured", "instructions": "",
		"model": map[string]interface{}{"id": "gpt-4o", "provider": "openai", "apiType": "chat"},
	}
	if outputs != nil {
		data["outputs"] = outputs
	}
	agent, err := prompty.LoadFrontmatter(data, t.TempDir(), prompty.LoadOptions{})
	if err != nil {
		t.Fatalf("load agent: %v", err)
	}
	return agent
}

func assertJSON(t *testing.T, label string, got interface{}, wantJSON string) {
	t.Helper()

	var want interface{}
	if err := json.Unmarshal([]byte(wantJSON), &want); err != nil {
		t.Fatalf("%s: bad expectation: %v", label, err)
	}
	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("%s: marshal: %v", label, err)
	}
	var normalized interface{}
	if err := json.Unmarshal(encoded, &normalized); err != nil {
		t.Fatalf("%s: unmarshal: %v", label, err)
	}
	if !reflect.DeepEqual(normalized, want) {
		gotPretty, _ := json.MarshalIndent(normalized, "", "  ")
		wantPretty, _ := json.MarshalIndent(want, "", "  ")
		t.Errorf("%s mismatch\n--- got ---\n%s\n--- want ---\n%s", label, gotPretty, wantPretty)
	}
}

func TestHasOutputsAndSchema(t *testing.T) {
	plain := agentWithOutputs(t, nil)
	if prompty.HasOutputs(plain) {
		t.Error("an agent with no outputs reported HasOutputs")
	}
	schema, err := prompty.OutputSchema(plain)
	if err != nil || schema != nil {
		t.Errorf("OutputSchema = %v, %v; want nil, nil", schema, err)
	}

	structured := agentWithOutputs(t, []interface{}{
		map[string]interface{}{"name": "city", "kind": "string", "required": true},
	})
	if !prompty.HasOutputs(structured) {
		t.Error("an agent with outputs did not report HasOutputs")
	}
}

// TestNestedStructuredSchema covers the composite shapes an author can declare:
// nested objects, arrays of objects, enums and optional fields together.
func TestNestedStructuredSchema(t *testing.T) {
	agent := agentWithOutputs(t, []interface{}{
		map[string]interface{}{
			"name": "order", "kind": "object", "required": true,
			"properties": []interface{}{
				map[string]interface{}{"name": "id", "kind": "string", "required": true},
				map[string]interface{}{
					"name": "lines", "kind": "array", "required": true,
					"items": map[string]interface{}{
						"kind": "object",
						"properties": []interface{}{
							map[string]interface{}{"name": "sku", "kind": "string", "required": true},
							map[string]interface{}{"name": "qty", "kind": "integer", "required": true},
						},
					},
				},
				map[string]interface{}{
					"name": "status", "kind": "string",
					"enumValues": []interface{}{"open", "closed"},
				},
			},
		},
	})

	format, err := prompty.ChatResponseFormat(agent)
	if err != nil {
		t.Fatalf("ChatResponseFormat: %v", err)
	}
	assertJSON(t, "chat response_format", format, `{
		"type": "json_schema",
		"json_schema": {
			"name": "structured_output",
			"strict": true,
			"schema": {
				"type": "object",
				"properties": {
					"order": {
						"type": "object",
						"properties": {
							"id": {"type": "string"},
							"lines": {
								"type": "array",
								"items": {
									"type": "object",
									"properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
									"required": ["sku", "qty"],
									"additionalProperties": false
								}
							},
							"status": {"type": ["string", "null"], "enum": ["open", "closed", null]}
						},
						"required": ["id", "lines"],
						"additionalProperties": false
					}
				},
				"required": ["order"],
				"additionalProperties": false
			}
		}
	}`)
}

// TestUnionStructuredSchema covers the anyOf branch shape.
func TestUnionStructuredSchema(t *testing.T) {
	agent := agentWithOutputs(t, []interface{}{
		map[string]interface{}{
			"name": "value", "kind": "union", "required": true,
			"anyOf": []interface{}{
				map[string]interface{}{"kind": "string"},
				map[string]interface{}{"kind": "integer"},
			},
		},
	})

	schema, err := prompty.OutputSchema(agent)
	if err != nil {
		t.Fatalf("OutputSchema: %v", err)
	}
	assertJSON(t, "union output", schema, `{
		"type": "object",
		"properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
		"required": ["value"],
		"additionalProperties": false
	}`)
}

// TestOneOfStructuredSchemaIsRejected: an unsupported composition must surface
// as an error at request-build time, not as a provider 400 later.
func TestOneOfStructuredSchemaIsRejected(t *testing.T) {
	agent := agentWithOutputs(t, []interface{}{
		map[string]interface{}{
			"name": "value", "kind": "union", "required": true,
			"oneOf": []interface{}{
				map[string]interface{}{"kind": "string"},
				map[string]interface{}{"kind": "integer"},
			},
		},
	})

	if _, err := prompty.OutputSchema(agent); err == nil {
		t.Fatal("expected oneOf to be rejected")
	}
	if _, err := prompty.ChatResponseFormat(agent); err == nil {
		t.Error("ChatResponseFormat swallowed the schema error")
	}
	if _, err := prompty.ResponsesTextFormat(agent); err == nil {
		t.Error("ResponsesTextFormat swallowed the schema error")
	}
	if _, err := openai.BuildChatRequest(agent, nil); err == nil {
		t.Error("BuildChatRequest swallowed the schema error")
	}
}

func TestResponsesTextFormatShape(t *testing.T) {
	agent := agentWithOutputs(t, []interface{}{
		map[string]interface{}{"name": "city", "kind": "string"},
	})

	format, err := prompty.ResponsesTextFormat(agent)
	if err != nil {
		t.Fatalf("ResponsesTextFormat: %v", err)
	}
	// The Responses dialect hoists strict out of the schema envelope.
	assertJSON(t, "responses text", format, `{
		"format": {
			"type": "json_schema",
			"name": "structured_output",
			"strict": true,
			"schema": {
				"type": "object",
				"properties": {"city": {"type": ["string", "null"]}},
				"required": ["city"],
				"additionalProperties": false
			}
		}
	}`)
}

// TestDecodeStructuredPreservesRawJSONSemantics is the decoding contract:
// integers stay integers, nesting survives, and a non-JSON answer comes back as
// the original string rather than an error the caller cannot act on.
func TestDecodeStructuredPreservesRawJSONSemantics(t *testing.T) {
	decoded := prompty.DecodeStructured(`{"n":7,"f":1.5,"s":"x","b":true,"z":null,
		"nested":{"a":[1,2,{"deep":3}]}}`)

	object, ok := decoded.(map[string]interface{})
	if !ok {
		t.Fatalf("decoded is %T, want a map", decoded)
	}
	if n, ok := object["n"].(int64); !ok || n != 7 {
		t.Errorf("n = %#v, want int64(7)", object["n"])
	}
	if f, ok := object["f"].(float64); !ok || f != 1.5 {
		t.Errorf("f = %#v, want float64(1.5)", object["f"])
	}
	if object["s"] != "x" || object["b"] != true {
		t.Errorf("scalars = %#v, %#v", object["s"], object["b"])
	}
	if value, present := object["z"]; !present || value != nil {
		t.Errorf("z = %#v, want an explicit nil", value)
	}

	nested, _ := object["nested"].(map[string]interface{})
	list, _ := nested["a"].([]interface{})
	if len(list) != 3 {
		t.Fatalf("nested array = %#v", nested["a"])
	}
	if first, ok := list[0].(int64); !ok || first != 1 {
		t.Errorf("nested[0] = %#v, want int64(1)", list[0])
	}
	deep, _ := list[2].(map[string]interface{})
	if value, ok := deep["deep"].(int64); !ok || value != 3 {
		t.Errorf("deep = %#v, want int64(3)", deep["deep"])
	}
}

func TestDecodeStructuredArrayPayload(t *testing.T) {
	decoded := prompty.DecodeStructured(`[{"a":1},{"a":2}]`)
	list, ok := decoded.([]interface{})
	if !ok || len(list) != 2 {
		t.Fatalf("decoded = %#v, want a two-element array", decoded)
	}
}

// TestDecodeStructuredLeavesNonJSONAlone covers the invalid-JSON contract and
// the scalar guard: "42" is an answer, not a number to silently coerce.
func TestDecodeStructuredLeavesNonJSONAlone(t *testing.T) {
	for _, text := range []string{
		"not valid json", "", "   ", "42", "true", "null",
		`{"broken":`, "I cannot help with that",
	} {
		if got := prompty.DecodeStructured(text); got != text {
			t.Errorf("DecodeStructured(%q) = %#v, want the original string", text, got)
		}
	}
}

func TestDecodeStructuredInto(t *testing.T) {
	var target struct {
		City string `json:"city"`
		Temp int    `json:"temp"`
	}

	if err := prompty.DecodeStructuredInto(`{"city":"Paris","temp":21}`, &target); err != nil {
		t.Fatalf("DecodeStructuredInto: %v", err)
	}
	if target.City != "Paris" || target.Temp != 21 {
		t.Errorf("target = %+v", target)
	}

	// Unlike DecodeStructured, the typed form reports a parse failure.
	if err := prompty.DecodeStructuredInto("not json", &target); err == nil {
		t.Error("expected an error for a non-JSON payload")
	} else if !errors.Is(err, prompty.ErrValue) {
		t.Errorf("error does not wrap ErrValue: %v", err)
	}
	if err := prompty.DecodeStructuredInto("", &target); err == nil {
		t.Error("expected an error for an empty payload")
	}
}

func TestStructuredResultAppliesTheAgentContract(t *testing.T) {
	plain := agentWithOutputs(t, nil)
	if got := prompty.StructuredResult(plain, `{"a":1}`); got != `{"a":1}` {
		t.Errorf("an agent with no outputs should get the raw text, got %#v", got)
	}

	structured := agentWithOutputs(t, []interface{}{
		map[string]interface{}{"name": "a", "kind": "integer"},
	})
	decoded := prompty.StructuredResult(structured, `{"a":1}`)
	if _, ok := decoded.(map[string]interface{}); !ok {
		t.Errorf("an agent with outputs should get a decoded value, got %T", decoded)
	}
}

// ---------------------------------------------------------------------------
// Composition APIs
// ---------------------------------------------------------------------------

// TestExecuteProcessThroughTheRegistry proves the public composition entry
// points resolve their invokers from the agent's provider key.
func TestExecuteProcessThroughTheRegistry(t *testing.T) {
	t.Cleanup(prompty.ClearCache)
	prompty.ClearCache()

	agent := agentWithOutputs(t, nil)
	response := map[string]interface{}{
		"choices": []interface{}{map[string]interface{}{
			"message": map[string]interface{}{"role": "assistant", "content": "hello"},
		}},
	}

	// Nothing is registered yet, so both entry points must report the miss.
	if _, err := prompty.Execute(context.Background(), agent, nil); !errors.Is(err, prompty.ErrInvoker) {
		t.Errorf("Execute error = %v, want ErrInvoker", err)
	}
	if _, err := prompty.Process(context.Background(), agent, response); !errors.Is(err, prompty.ErrInvoker) {
		t.Errorf("Process error = %v, want ErrInvoker", err)
	}

	prompty.RegisterExecutor("openai", &scriptedExecutor{responses: []interface{}{response}})
	prompty.RegisterProcessor("openai", openai.NewProcessor())

	raw, err := prompty.Execute(context.Background(), agent,
		[]model.Message{{Role: model.RoleUser, Parts: []interface{}{model.TextPart{Kind: "text", Value: "hi"}}}})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	result, err := prompty.Process(context.Background(), agent, raw)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if result != "hello" {
		t.Errorf("result = %#v", result)
	}
}

// TestExecuteHonoursCancellation guards the boundary check for executors that
// do not implement the context-aware extension.
func TestExecuteHonoursCancellation(t *testing.T) {
	t.Cleanup(prompty.ClearCache)
	prompty.ClearCache()

	agent := agentWithOutputs(t, nil)
	prompty.RegisterExecutor("openai", &scriptedExecutor{responses: []interface{}{
		map[string]interface{}{"choices": []interface{}{}},
	}})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := prompty.Execute(ctx, agent, nil); !errors.Is(err, context.Canceled) {
		t.Errorf("Execute error = %v, want context.Canceled", err)
	}
}

// TestProviderRegistrationIsExplicit: importing a provider package must not
// register anything, so a host never links a client it did not ask for.
func TestProviderRegistrationIsExplicit(t *testing.T) {
	t.Cleanup(prompty.ClearCache)
	prompty.ClearCache()

	if prompty.HasExecutor("openai") || prompty.HasProcessor("openai") {
		t.Fatal("importing prompty/openai registered a provider as a side effect")
	}

	openai.Register()
	if !prompty.HasExecutor("openai") || !prompty.HasProcessor("openai") {
		t.Error("openai.Register did not install the provider")
	}

	openai.RegisterFoundry()
	for _, key := range []string{"foundry", "azure"} {
		if !prompty.HasExecutor(key) || !prompty.HasProcessor(key) {
			t.Errorf("RegisterFoundry did not install %q", key)
		}
	}
}
