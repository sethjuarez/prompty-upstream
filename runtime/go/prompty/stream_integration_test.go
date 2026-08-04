package prompty_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	prompty "prompty"
	model "prompty/model"
	openai "prompty/openai"
)

// TestStreamMessagesThroughTheRegistry exercises the streaming composition
// entry point end to end: registry resolution, the executor's streaming call,
// the processor's SSE decode and the chunk stream a host consumes.
func TestStreamMessagesThroughTheRegistry(t *testing.T) {
	t.Cleanup(prompty.ClearCache)
	prompty.ClearCache()

	const body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n" +
		"data: [DONE]\n\n"

	var gotAccept string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAccept = r.Header.Get("Accept")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, body)
	}))
	defer server.Close()

	agent, err := prompty.LoadFrontmatter(map[string]interface{}{
		"name": "streaming", "instructions": "",
		"model": map[string]interface{}{
			"id": "gpt-4o-mini", "provider": "openai", "apiType": "chat",
			"connection": map[string]interface{}{
				"kind": "key", "endpoint": server.URL, "apiKey": "sk-test",
			},
		},
	}, t.TempDir(), prompty.LoadOptions{})
	if err != nil {
		t.Fatalf("load agent: %v", err)
	}

	openai.RegisterAs("openai",
		&openai.Executor{
			Env:    func(string) (string, bool) { return "", false },
			Client: server.Client(),
		},
		openai.NewProcessor())

	stream, err := prompty.StreamMessages(context.Background(), agent,
		[]model.Message{{Role: model.RoleUser, Parts: []interface{}{model.TextPart{Kind: "text", Value: "hi"}}}})
	if err != nil {
		t.Fatalf("StreamMessages: %v", err)
	}

	result, err := prompty.Collect(context.Background(), stream)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if result.Text != "Hello" {
		t.Errorf("text = %q, want \"Hello\"", result.Text)
	}
	if result.Usage.TotalTokens != 6 {
		t.Errorf("usage = %+v", result.Usage)
	}
	if gotAccept != "text/event-stream" {
		t.Errorf("Accept = %q, want text/event-stream", gotAccept)
	}
}

// TestStreamRunPreparesThePrompt proves the higher-level entry point renders
// and parses before streaming, so a host can go straight from inputs to chunks.
func TestStreamRunPreparesThePrompt(t *testing.T) {
	t.Cleanup(prompty.ClearCache)
	prompty.ClearCache()

	var gotMessages []interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]interface{}
		_ = decodeJSONBody(r, &request)
		gotMessages, _ = request["messages"].([]interface{})

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()

	agent, err := prompty.LoadFrontmatter(map[string]interface{}{
		"name":         "greeter",
		"instructions": "system:\nBe brief.\nuser:\nHello {{name}}",
		"inputs": map[string]interface{}{
			"name": map[string]interface{}{"kind": "string", "required": true},
		},
		"model": map[string]interface{}{
			"id": "gpt-4o-mini", "provider": "openai", "apiType": "chat",
			"connection": map[string]interface{}{
				"kind": "key", "endpoint": server.URL, "apiKey": "sk-test",
			},
		},
	}, t.TempDir(), prompty.LoadOptions{})
	if err != nil {
		t.Fatalf("load agent: %v", err)
	}

	openai.RegisterAs("openai",
		&openai.Executor{
			Env:    func(string) (string, bool) { return "", false },
			Client: server.Client(),
		},
		openai.NewProcessor())

	stream, err := prompty.StreamRun(context.Background(), agent, map[string]interface{}{"name": "Ada"})
	if err != nil {
		t.Fatalf("StreamRun: %v", err)
	}
	result, err := prompty.Collect(context.Background(), stream)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if result.Text != "ok" {
		t.Errorf("text = %q", result.Text)
	}

	if len(gotMessages) != 2 {
		t.Fatalf("provider saw %d messages, want the parsed system + user pair", len(gotMessages))
	}
	user, _ := gotMessages[1].(map[string]interface{})
	if user["role"] != "user" || user["content"] != "Hello Ada" {
		t.Errorf("user message = %v, want the rendered input", user)
	}
}

// TestStreamRunMissingRequiredInputFailsBeforeAnyCall proves input validation
// happens before the provider is contacted.
func TestStreamRunMissingRequiredInputFailsBeforeAnyCall(t *testing.T) {
	t.Cleanup(prompty.ClearCache)
	prompty.ClearCache()

	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("the provider must not be called when an input is missing")
	}))
	defer server.Close()

	agent, err := prompty.LoadFrontmatter(map[string]interface{}{
		"name": "greeter", "instructions": "user:\nHello {{name}}",
		"inputs": map[string]interface{}{
			"name": map[string]interface{}{"kind": "string", "required": true},
		},
		"model": map[string]interface{}{
			"id": "gpt-4o-mini", "provider": "openai", "apiType": "chat",
			"connection": map[string]interface{}{"kind": "key", "endpoint": server.URL, "apiKey": "sk"},
		},
	}, t.TempDir(), prompty.LoadOptions{})
	if err != nil {
		t.Fatalf("load agent: %v", err)
	}

	openai.RegisterAs("openai",
		&openai.Executor{Env: func(string) (string, bool) { return "", false }, Client: server.Client()},
		openai.NewProcessor())

	if _, err := prompty.StreamRun(context.Background(), agent, nil); err == nil {
		t.Fatal("expected a validation error for the missing input")
	}
}

func decodeJSONBody(r *http.Request, target interface{}) error {
	defer r.Body.Close()
	return json.NewDecoder(r.Body).Decode(target)
}
