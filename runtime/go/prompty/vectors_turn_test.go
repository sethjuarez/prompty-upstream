package prompty_test

import (
	"context"
	"errors"
	"testing"

	prompty "prompty"
	model "prompty/model"
	openai "prompty/openai"
)

// turnVectorFile is the shared engine turn vector file.
type turnVectorFile struct {
	Version string       `json:"version"`
	Cases   []turnVector `json:"cases"`
}

type turnVector struct {
	Name            string `json:"name"`
	CancelBeforeRun bool   `json:"cancelBeforeRun"`
	Messages        []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	Model []struct {
		Assistant string `json:"assistant"`
		Output    string `json:"output"`
		Tools     []struct {
			ID        string                 `json:"id"`
			Name      string                 `json:"name"`
			Arguments map[string]interface{} `json:"arguments"`
		} `json:"tools"`
		NextPortability string `json:"nextPortability"`
	} `json:"model"`
	ToolOutputs map[string]string `json:"toolOutputs"`
	DenyTools   []string          `json:"denyTools"`
	Expected    struct {
		Status          string   `json:"status"`
		Output          string   `json:"output"`
		Iterations      int      `json:"iterations"`
		ToolResults     int      `json:"toolResults"`
		ToolResultOrder []string `json:"toolResultOrder"`
	} `json:"expected"`
}

// unsupportedTurnVectors names the engine cases outside this slice.
var unsupportedTurnVectors = map[string]string{
	"delegated_provider_state": "delegated provider state, snapshots and checkpoint portability are the emitted ReferenceTurnRunner's protocol, not the practical provider/tool pipeline",
}

// TestTurnVectors drives the feasible subset of the shared engine turn vectors
// through the practical turn loop.
//
// The vectors also pin snapshot counts, checkpoint creation, commit portability
// and the low-level engine event kinds. Those belong to the emitted
// ReferenceTurnRunner, which this slice deliberately does not duplicate, so
// only the observable turn outcome is asserted here: status, output, iteration
// count, and the number and order of tool results.
func TestTurnVectors(t *testing.T) {
	var file turnVectorFile
	readVectorFile(t, "engine/turn_vectors.json", &file)

	var executed, skipped int
	for _, vector := range file.Cases {
		if reason, ok := unsupportedTurnVectors[vector.Name]; ok {
			skipped++
			t.Run(vector.Name, func(t *testing.T) { t.Skip(reason) })
			continue
		}

		executed++
		t.Run(vector.Name, func(t *testing.T) { runTurnVector(t, vector) })
	}

	t.Logf("engine turn vectors: %d executed, %d skipped with explicit reasons, %d total",
		executed, skipped, len(file.Cases))
	if executed == 0 {
		t.Fatal("no turn vectors executed; the harness is not selecting cases")
	}
}

func runTurnVector(t *testing.T, vector turnVector) {
	t.Helper()

	agent := agentForTurnVector(t, vector)

	messages := make([]model.Message, 0, len(vector.Messages))
	for _, entry := range vector.Messages {
		messages = append(messages, model.Message{
			Role:  model.Role(entry.Role),
			Parts: []interface{}{model.TextPart{Kind: "text", Value: entry.Content}},
		})
	}

	responses := make([]interface{}, 0, len(vector.Model))
	for _, step := range vector.Model {
		responses = append(responses, chatResponseFor(step.Assistant, step.Output, step.Tools))
	}

	registry := prompty.NewToolRegistry()
	registry.RegisterText("echo", func(_ context.Context, args map[string]interface{}) (string, error) {
		if value, ok := args["value"].(string); ok {
			return value, nil
		}
		return "", nil
	})
	registry.RegisterText("protected", func(context.Context, map[string]interface{}) (string, error) {
		t.Error("a denied tool must never execute")
		return "", nil
	})

	denied := map[string]bool{}
	for _, name := range vector.DenyTools {
		denied[name] = true
	}

	options := prompty.RunOptions{
		Tools:     registry,
		Executor:  &scriptedExecutor{responses: responses},
		Processor: openai.NewProcessor(),
		Permit: func(_ context.Context, call model.ToolCall, _ map[string]interface{}) prompty.PermissionDecision {
			if denied[call.Name] {
				return prompty.Deny("permission denied")
			}
			return prompty.Allow()
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if vector.CancelBeforeRun {
		cancel()
	}

	result, err := prompty.RunMessages(ctx, agent, messages, options)

	if vector.Expected.Status == "cancelled" {
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected cancellation, got err=%v", err)
		}
	} else if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Iterations != vector.Expected.Iterations {
		t.Errorf("iterations = %d, want %d", result.Iterations, vector.Expected.Iterations)
	}
	if vector.Expected.Output != "" && result.Text() != vector.Expected.Output {
		t.Errorf("output = %q, want %q", result.Text(), vector.Expected.Output)
	}
	if len(result.Dispatches) != vector.Expected.ToolResults {
		t.Errorf("tool results = %d, want %d", len(result.Dispatches), vector.Expected.ToolResults)
	}

	if vector.Expected.ToolResultOrder != nil {
		order := make([]string, 0, len(result.Dispatches))
		for _, dispatch := range result.Dispatches {
			order = append(order, dispatch.Call.Id)
		}
		if !equalStrings(order, vector.Expected.ToolResultOrder) {
			t.Errorf("tool result order = %v, want %v", order, vector.Expected.ToolResultOrder)
		}
	}

	// A denied tool must still come back to the model with something, or the
	// provider rejects the following turn for an unanswered tool call.
	for _, dispatch := range result.Dispatches {
		if dispatch.Denied && dispatch.Text() == "" {
			t.Errorf("denied tool %q produced no model-visible result", dispatch.Call.Name)
		}
	}
}

func agentForTurnVector(t *testing.T, vector turnVector) model.Prompty {
	t.Helper()

	agent, err := prompty.LoadFrontmatter(map[string]interface{}{
		"name":         vector.Name,
		"instructions": "",
		"model": map[string]interface{}{
			"id": "gpt-4o-mini", "provider": "openai", "apiType": "chat",
		},
		"tools": []interface{}{
			map[string]interface{}{
				"name": "echo", "kind": "function", "description": "Echo a value",
				"parameters": []interface{}{
					map[string]interface{}{"name": "value", "kind": "string", "required": true},
				},
			},
			map[string]interface{}{
				"name": "protected", "kind": "function", "description": "A protected resource",
				"parameters": []interface{}{},
			},
		},
	}, t.TempDir(), prompty.LoadOptions{})
	if err != nil {
		t.Fatalf("load agent: %v", err)
	}
	return agent
}

// chatResponseFor renders a turn-vector model step as an OpenAI chat response,
// so the real processor decodes it rather than a test-only shortcut.
func chatResponseFor(assistant, output string, tools []struct {
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}) map[string]interface{} {
	message := map[string]interface{}{"role": "assistant"}

	if len(tools) > 0 {
		calls := make([]interface{}, 0, len(tools))
		for _, tool := range tools {
			arguments, err := prompty.EncodeToolArguments(tool.Arguments)
			if err != nil {
				arguments = "{}"
			}
			calls = append(calls, map[string]interface{}{
				"id": tool.ID, "type": "function",
				"function": map[string]interface{}{"name": tool.Name, "arguments": arguments},
			})
		}
		message["tool_calls"] = calls
		message["content"] = assistant
	} else {
		message["content"] = output
	}

	return map[string]interface{}{
		"object":  "chat.completion",
		"choices": []interface{}{map[string]interface{}{"index": 0, "message": message}},
	}
}
