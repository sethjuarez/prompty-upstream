package openai_test

import (
	"testing"

	openai "prompty/openai"
)

// TestWireVectors drives every OpenAI-tagged case in the shared wire vector
// file through the deterministic request builders. No network, no executor —
// this is the request-shape contract on its own.
func TestWireVectors(t *testing.T) {
	vectors := loadVectors(t, "wire/wire_vectors.json")

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
			agent := agentFromVector(t, input)
			messages := messagesFromVector(t, input["messages"])

			body, err := openai.BuildRequest(agent, messages)
			if err != nil {
				t.Fatalf("BuildRequest: %v", err)
			}

			expected, ok := vector["expected"].(map[string]interface{})
			if !ok {
				t.Fatalf("vector has no expected object")
			}
			assertJSONEqual(t, "request_body", body, expected["request_body"])
		})
	}

	t.Logf("openai wire vectors: %d executed, %d skipped (non-openai provider)", executed, skipped)
	if executed == 0 {
		t.Fatal("no openai wire vectors executed; the harness is not selecting cases")
	}
}

// TestWireVectorsAreDeterministic re-runs each case and requires byte-identical
// output. Map iteration order in Go is randomised, so a builder that leaked map
// ordering into a slice would fail here and nowhere else.
func TestWireVectorsAreDeterministic(t *testing.T) {
	vectors := loadVectors(t, "wire/wire_vectors.json")

	for _, vector := range vectors {
		name, _ := vector["name"].(string)
		input, ok := vector["input"].(map[string]interface{})
		if !ok {
			continue
		}
		if provider, _ := input["provider"].(string); provider != "openai" {
			continue
		}

		t.Run(name, func(t *testing.T) {
			agent := agentFromVector(t, input)
			messages := messagesFromVector(t, input["messages"])

			first, err := openai.BuildRequest(agent, messages)
			if err != nil {
				t.Fatalf("BuildRequest: %v", err)
			}
			for i := 0; i < 8; i++ {
				again, err := openai.BuildRequest(agent, messages)
				if err != nil {
					t.Fatalf("BuildRequest repeat %d: %v", i, err)
				}
				assertJSONEqual(t, "repeat", again, first)
			}
		})
	}
}
