package openai_test

import (
	"context"
	"testing"

	openai "prompty/openai"
)

// TestProcessVectors drives every OpenAI-tagged case in the shared process
// vector file through the processor.
func TestProcessVectors(t *testing.T) {
	vectors := loadVectors(t, "process/process_vectors.json")
	processor := openai.NewProcessor()

	var executed, skipped int
	for _, vector := range vectors {
		name, _ := vector["name"].(string)
		input, ok := vector["input"].(map[string]interface{})
		if !ok {
			t.Fatalf("%s: vector has no input object", name)
		}
		if provider, _ := input["provider"].(string); provider != "openai" {
			skipped++
			continue
		}

		executed++
		t.Run(name, func(t *testing.T) {
			agentInput := map[string]interface{}{
				"model_id": "vector-model",
				"provider": "openai",
				"apiType":  input["apiType"],
			}
			// has_outputs is the vector's way of saying "this agent declares
			// structured outputs"; the exact schema is irrelevant to
			// processing, only its presence is.
			if hasOutputs, _ := input["has_outputs"].(bool); hasOutputs {
				agentInput["outputs"] = []interface{}{
					map[string]interface{}{"name": "value", "kind": "string"},
				}
			}
			agent := agentFromVector(t, agentInput)

			result, err := processor.ProcessContext(context.Background(), agent, input["response"])
			if err != nil {
				t.Fatalf("Process: %v", err)
			}

			expected, ok := vector["expected"].(map[string]interface{})
			if !ok {
				t.Fatalf("vector has no expected object")
			}
			assertJSONEqual(t, "result", result, expected["result"])
		})
	}

	t.Logf("openai process vectors: %d executed, %d skipped (non-openai provider)", executed, skipped)
	if executed == 0 {
		t.Fatal("no openai process vectors executed; the harness is not selecting cases")
	}
}

// TestProcessRejectsNonObject proves a malformed provider payload becomes an
// error rather than a panic in a host process.
func TestProcessRejectsNonObject(t *testing.T) {
	processor := openai.NewProcessor()
	agent := agentFromVector(t, map[string]interface{}{
		"model_id": "gpt-4", "provider": "openai", "apiType": "chat",
	})

	for _, bad := range []interface{}{nil, 42, []interface{}{1, 2}, "not json"} {
		if _, err := processor.Process(agent, bad); err == nil {
			t.Errorf("Process(%#v) returned no error", bad)
		}
	}
}

// TestProcessStructuredPreservesIntegers guards the number-narrowing path: a
// structured answer fed back into a template must render 72, not 72.
func TestProcessStructuredPreservesIntegers(t *testing.T) {
	processor := openai.NewProcessor()
	agent := agentFromVector(t, map[string]interface{}{
		"model_id": "gpt-4", "provider": "openai", "apiType": "chat",
		"outputs": []interface{}{
			map[string]interface{}{"name": "temp", "kind": "integer"},
		},
	})

	response := map[string]interface{}{
		"choices": []interface{}{
			map[string]interface{}{
				"message": map[string]interface{}{
					"role": "assistant", "content": `{"temp":72,"ratio":0.5}`,
				},
			},
		},
	}

	result, err := processor.Process(agent, response)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	decoded, ok := result.(map[string]interface{})
	if !ok {
		t.Fatalf("expected a decoded object, got %T", result)
	}
	if temp, ok := decoded["temp"].(int64); !ok || temp != 72 {
		t.Errorf("temp = %#v, want int64(72)", decoded["temp"])
	}
	if ratio, ok := decoded["ratio"].(float64); !ok || ratio != 0.5 {
		t.Errorf("ratio = %#v, want float64(0.5)", decoded["ratio"])
	}
}
