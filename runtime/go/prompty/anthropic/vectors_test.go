package anthropic_test

import (
	"context"
	"testing"

	anthropic "prompty/anthropic"
)

// TestWireVectors drives every Anthropic-tagged case in the shared wire vector
// file through the deterministic request builder.
func TestWireVectors(t *testing.T) {
	vectors := loadVectors(t, "wire/wire_vectors.json")

	var executed, skipped int
	for _, vector := range vectors {
		name, _ := vector["name"].(string)
		input, ok := vector["input"].(map[string]interface{})
		if !ok {
			t.Fatalf("%s: vector has no input object", name)
		}
		if provider, _ := input["provider"].(string); provider != "anthropic" {
			skipped++
			continue
		}

		executed++
		t.Run(name, func(t *testing.T) {
			agent := agentFromVector(t, input)
			messages := messagesFromVector(t, input["messages"])

			body, err := anthropic.BuildChatRequest(agent, messages)
			if err != nil {
				t.Fatalf("BuildChatRequest: %v", err)
			}

			expected, ok := vector["expected"].(map[string]interface{})
			if !ok {
				t.Fatalf("vector has no expected object")
			}
			assertJSONEqual(t, "request_body", body, expected["request_body"])
		})
	}

	t.Logf("anthropic wire vectors: %d executed, %d skipped (non-anthropic provider)", executed, skipped)
	if executed == 0 {
		t.Fatal("no anthropic wire vectors executed; the harness is not selecting cases")
	}
}

// TestWireVectorsAreDeterministic re-runs each case and requires identical
// output, which catches any place map iteration order leaked into a slice.
func TestWireVectorsAreDeterministic(t *testing.T) {
	vectors := loadVectors(t, "wire/wire_vectors.json")

	for _, vector := range vectors {
		name, _ := vector["name"].(string)
		input, ok := vector["input"].(map[string]interface{})
		if !ok {
			continue
		}
		if provider, _ := input["provider"].(string); provider != "anthropic" {
			continue
		}

		t.Run(name, func(t *testing.T) {
			agent := agentFromVector(t, input)
			messages := messagesFromVector(t, input["messages"])

			first, err := anthropic.BuildChatRequest(agent, messages)
			if err != nil {
				t.Fatalf("BuildChatRequest: %v", err)
			}
			for i := 0; i < 8; i++ {
				again, err := anthropic.BuildChatRequest(agent, messages)
				if err != nil {
					t.Fatalf("BuildChatRequest repeat %d: %v", i, err)
				}
				assertJSONEqual(t, "repeat", again, first)
			}
		})
	}
}

// TestProcessVectors drives every Anthropic-tagged case in the shared process
// vector file through the processor.
func TestProcessVectors(t *testing.T) {
	vectors := loadVectors(t, "process/process_vectors.json")
	processor := anthropic.NewProcessor()

	var executed, skipped int
	for _, vector := range vectors {
		name, _ := vector["name"].(string)
		input, ok := vector["input"].(map[string]interface{})
		if !ok {
			t.Fatalf("%s: vector has no input object", name)
		}
		if provider, _ := input["provider"].(string); provider != "anthropic" {
			skipped++
			continue
		}

		executed++
		t.Run(name, func(t *testing.T) {
			agentInput := map[string]interface{}{
				"model_id": "claude-sonnet-4-20250514",
				"provider": "anthropic",
				"apiType":  "chat",
			}
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

	t.Logf("anthropic process vectors: %d executed, %d skipped (non-anthropic provider)", executed, skipped)
	if executed == 0 {
		t.Fatal("no anthropic process vectors executed; the harness is not selecting cases")
	}
}

// TestProcessRejectsNonObject proves a malformed provider payload becomes an
// error rather than a panic in a host process.
func TestProcessRejectsNonObject(t *testing.T) {
	processor := anthropic.NewProcessor()
	agent := agentFromVector(t, map[string]interface{}{
		"model_id": "claude-3", "provider": "anthropic", "apiType": "chat",
	})

	for _, bad := range []interface{}{nil, 42, []interface{}{1, 2}, "not json"} {
		if _, err := processor.Process(agent, bad); err == nil {
			t.Errorf("Process(%#v) returned no error", bad)
		}
	}
}
