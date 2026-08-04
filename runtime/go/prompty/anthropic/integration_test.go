package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	prompty "prompty"
	anthropic "prompty/anthropic"
	model "prompty/model"
	wire "prompty/wire"
)

func staticEnv(pairs map[string]string) anthropic.Env {
	return func(key string) (string, bool) {
		value, ok := pairs[key]
		return value, ok
	}
}

func testAgent(t *testing.T, connection map[string]interface{}, extra map[string]interface{}) model.Prompty {
	t.Helper()

	modelSpec := map[string]interface{}{
		"id": "claude-sonnet-4-20250514", "provider": "anthropic", "apiType": "chat",
	}
	if connection != nil {
		modelSpec["connection"] = connection
	}
	data := map[string]interface{}{"name": "integration", "instructions": "", "model": modelSpec}
	for key, value := range extra {
		if key == "options" {
			modelSpec["options"] = value
			continue
		}
		data[key] = value
	}

	agent, err := prompty.LoadFrontmatter(data, t.TempDir(), prompty.LoadOptions{})
	if err != nil {
		t.Fatalf("load agent: %v", err)
	}
	return agent
}

func userMessage(text string) []model.Message {
	return []model.Message{{
		Role:  model.RoleUser,
		Parts: []interface{}{model.TextPart{Kind: "text", Value: text}},
	}}
}

// ---------------------------------------------------------------------------
// Endpoint and header construction
// ---------------------------------------------------------------------------

func TestEndpointAndHeaders(t *testing.T) {
	cases := []struct {
		name       string
		connection map[string]interface{}
		env        map[string]string
		wantURL    string
		wantKey    string
		wantVer    string
	}{
		{
			name:    "default endpoint with env credential",
			env:     map[string]string{anthropic.EnvAPIKey: "sk-ant-env"},
			wantURL: "https://api.anthropic.com/v1/messages",
			wantKey: "sk-ant-env",
			wantVer: anthropic.Version,
		},
		{
			name:       "connection endpoint and credential win over the environment",
			connection: map[string]interface{}{"kind": "key", "endpoint": "https://proxy.example.com", "apiKey": "sk-ant-conn"},
			env:        map[string]string{anthropic.EnvAPIKey: "sk-ant-env", anthropic.EnvBaseURL: "https://ignored"},
			wantURL:    "https://proxy.example.com/v1/messages",
			wantKey:    "sk-ant-conn",
			wantVer:    anthropic.Version,
		},
		{
			name:       "trailing slash is trimmed",
			connection: map[string]interface{}{"kind": "key", "endpoint": "https://proxy.example.com/", "apiKey": "sk"},
			wantURL:    "https://proxy.example.com/v1/messages",
			wantKey:    "sk",
			wantVer:    anthropic.Version,
		},
		{
			name:       "version override from the environment",
			connection: map[string]interface{}{"kind": "key", "endpoint": "https://api.anthropic.com", "apiKey": "sk"},
			env:        map[string]string{anthropic.EnvVersion: "2099-01-01"},
			wantURL:    "https://api.anthropic.com/v1/messages",
			wantKey:    "sk",
			wantVer:    "2099-01-01",
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			agent := testAgent(t, testCase.connection, nil)
			request, err := anthropic.BuildProviderRequest(
				agent, userMessage("hi"), staticEnv(testCase.env), false)
			if err != nil {
				t.Fatalf("BuildProviderRequest: %v", err)
			}
			if request.URL != testCase.wantURL {
				t.Errorf("URL = %q, want %q", request.URL, testCase.wantURL)
			}
			if got := request.Header["x-api-key"]; got != testCase.wantKey {
				t.Errorf("x-api-key = %q, want %q", got, testCase.wantKey)
			}
			if got := request.Header["anthropic-version"]; got != testCase.wantVer {
				t.Errorf("anthropic-version = %q, want %q", got, testCase.wantVer)
			}
		})
	}
}

func TestCredentialNeverReachesDiagnostics(t *testing.T) {
	const secret = "sk-ant-super-secret"

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": "https://api.anthropic.com", "apiKey": secret,
	}, nil)

	request, err := anthropic.BuildProviderRequest(agent, userMessage("hi"), staticEnv(nil), false)
	if err != nil {
		t.Fatalf("BuildProviderRequest: %v", err)
	}
	if request.Header["x-api-key"] != secret {
		t.Fatal("precondition failed: the credential should be in the x-api-key header")
	}
	if got := request.String(); strings.Contains(got, secret) {
		t.Errorf("Request.String leaked the credential: %s", got)
	}
	for key, value := range request.RedactedHeader() {
		if strings.Contains(value, secret) {
			t.Errorf("RedactedHeader leaked the credential in %s", key)
		}
	}
	if got := request.RedactedHeader()["x-api-key"]; got != wire.Redacted {
		t.Errorf("RedactedHeader x-api-key = %q, want %q", got, wire.Redacted)
	}
}

func TestMissingCredentialIsAnError(t *testing.T) {
	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": "https://api.anthropic.com",
	}, nil)

	_, err := anthropic.BuildProviderRequest(agent, userMessage("hi"), staticEnv(nil), false)
	if err == nil {
		t.Fatal("expected an error when no credential is configured")
	}
	if !strings.Contains(err.Error(), anthropic.EnvAPIKey) {
		t.Errorf("error should name the environment variable to set: %v", err)
	}
}

func TestUnsupportedAPITypeIsRejected(t *testing.T) {
	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": "https://api.anthropic.com", "apiKey": "sk",
	}, nil)
	// Rebuild with an apiType Anthropic does not serve.
	agent.Model.Id = "claude-3"
	embedding, err := prompty.LoadFrontmatter(map[string]interface{}{
		"name": "e", "instructions": "",
		"model": map[string]interface{}{
			"id": "claude-3", "provider": "anthropic", "apiType": "embedding",
			"connection": map[string]interface{}{"kind": "key", "apiKey": "sk"},
		},
	}, t.TempDir(), prompty.LoadOptions{})
	if err != nil {
		t.Fatalf("load agent: %v", err)
	}
	if _, err := anthropic.BuildProviderRequest(embedding, userMessage("hi"), staticEnv(nil), false); err == nil {
		t.Error("expected an error for an unsupported apiType")
	}
}

// ---------------------------------------------------------------------------
// HTTP execution
// ---------------------------------------------------------------------------

func TestExecuteAgainstTestServer(t *testing.T) {
	var (
		gotPath string
		gotKey  string
		gotVer  string
		gotBody map[string]interface{}
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotKey = r.Header.Get("x-api-key")
		gotVer = r.Header.Get("anthropic-version")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant",
			"content":[{"type":"text","text":"pong"}],"stop_reason":"end_turn",
			"usage":{"input_tokens":3,"output_tokens":1}}`)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-ant",
	}, nil)

	executor := &anthropic.Executor{Env: staticEnv(nil), Client: server.Client()}
	raw, err := executor.ExecuteContext(context.Background(), agent, userMessage("ping"))
	if err != nil {
		t.Fatalf("ExecuteContext: %v", err)
	}

	if gotPath != "/v1/messages" {
		t.Errorf("path = %q, want /v1/messages", gotPath)
	}
	if gotKey != "sk-ant" {
		t.Errorf("x-api-key = %q", gotKey)
	}
	if gotVer != anthropic.Version {
		t.Errorf("anthropic-version = %q", gotVer)
	}
	// max_tokens is mandatory for Anthropic and must be present even though the
	// agent declared no options.
	if gotBody["max_tokens"] != float64(anthropic.DefaultMaxTokens) {
		t.Errorf("max_tokens = %v, want %d", gotBody["max_tokens"], anthropic.DefaultMaxTokens)
	}

	result, err := anthropic.NewProcessor().Process(agent, raw)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if result != "pong" {
		t.Errorf("result = %#v, want \"pong\"", result)
	}

	usage, ok := anthropic.UsageOf(raw.(map[string]interface{}))
	if !ok || usage.TotalTokens != 4 {
		t.Errorf("usage = %+v, ok=%v", usage, ok)
	}
}

func TestExecutePropagatesHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"bad model"}}`)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-ant",
	}, nil)

	executor := &anthropic.Executor{Env: staticEnv(nil), Client: server.Client()}
	_, err := executor.ExecuteContext(context.Background(), agent, userMessage("ping"))
	if err == nil {
		t.Fatal("expected an error for HTTP 400")
	}

	var providerErr *wire.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("error is %T, want *wire.ProviderError", err)
	}
	if providerErr.Status != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", providerErr.Status)
	}
	if !strings.Contains(providerErr.Error(), "bad model") {
		t.Errorf("error does not carry the provider message: %s", providerErr.Error())
	}
}

func TestExecuteRespectsContextCancellation(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer server.Close()
	defer close(release)

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-ant",
	}, nil)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	executor := &anthropic.Executor{Env: staticEnv(nil), Client: server.Client()}
	if _, err := executor.ExecuteContext(ctx, agent, userMessage("ping")); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

const textStreamBody = "event: message_start\n" +
	"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":9,\"output_tokens\":0}}}\n\n" +
	"event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n" +
	"event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n" +
	"event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n" +
	"event: message_delta\n" +
	"data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":4}}\n\n" +
	"event: message_stop\n" +
	"data: {\"type\":\"message_stop\"}\n\n"

func TestStreamingText(t *testing.T) {
	var gotBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, textStreamBody)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-ant",
	}, nil)

	executor := &anthropic.Executor{Env: staticEnv(nil), Client: server.Client()}
	raw, err := executor.ExecuteStreamContext(context.Background(), agent, userMessage("hi"))
	if err != nil {
		t.Fatalf("ExecuteStreamContext: %v", err)
	}
	if gotBody["stream"] != true {
		t.Errorf("stream = %v, want true", gotBody["stream"])
	}

	stream, err := anthropic.NewProcessor().ProcessStreamContext(context.Background(), raw)
	if err != nil {
		t.Fatalf("ProcessStreamContext: %v", err)
	}
	result, err := prompty.Collect(context.Background(), stream.(*prompty.Stream))
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	if result.Text != "Hello" {
		t.Errorf("text = %q, want \"Hello\"", result.Text)
	}
	if result.Thinking != "hmm" {
		t.Errorf("thinking = %q", result.Thinking)
	}
	// input_tokens only ever arrives on message_start and output_tokens only on
	// message_delta, so the two must be merged rather than the later replacing
	// the earlier.
	if result.Usage.InputTokens != 9 || result.Usage.OutputTokens != 4 || result.Usage.TotalTokens != 13 {
		t.Errorf("usage = %+v, want input 9 / output 4 / total 13", result.Usage)
	}
}

const toolStreamBody = "event: content_block_start\n" +
	"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_a\",\"name\":\"get_weather\"}}\n\n" +
	"event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n" +
	"event: content_block_delta\n" +
	"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Paris\\\"}\"}}\n\n" +
	"event: content_block_start\n" +
	"data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_b\",\"name\":\"get_time\"}}\n\n" +
	"event: message_stop\n" +
	"data: {\"type\":\"message_stop\"}\n\n"

func TestStreamingToolCallsPreserveOrder(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, toolStreamBody)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-ant",
	}, nil)

	executor := &anthropic.Executor{Env: staticEnv(nil), Client: server.Client()}
	raw, err := executor.ExecuteStreamContext(context.Background(), agent, userMessage("hi"))
	if err != nil {
		t.Fatalf("ExecuteStreamContext: %v", err)
	}
	stream := anthropic.DecodeStream(context.Background(), raw.(anthropic.RawStream))

	result, err := prompty.Collect(context.Background(), stream)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if len(result.ToolCalls) != 2 {
		t.Fatalf("tool calls = %d, want 2: %+v", len(result.ToolCalls), result.ToolCalls)
	}
	if result.ToolCalls[0].Id != "toolu_a" || result.ToolCalls[1].Id != "toolu_b" {
		t.Errorf("tool call order = %v", result.ToolCalls)
	}
	if result.ToolCalls[0].Arguments != `{"city":"Paris"}` {
		t.Errorf("reassembled arguments = %q", result.ToolCalls[0].Arguments)
	}
	// A tool_use block with no argument fragments still needs valid JSON.
	if result.ToolCalls[1].Arguments != "{}" {
		t.Errorf("empty arguments = %q, want {}", result.ToolCalls[1].Arguments)
	}
}

func TestStreamingInBandErrorBecomesErrorChunk(t *testing.T) {
	const body = "event: error\n" +
		"data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"overloaded\"}}\n\n"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, body)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-ant",
	}, nil)

	executor := &anthropic.Executor{Env: staticEnv(nil), Client: server.Client()}
	raw, err := executor.ExecuteStreamContext(context.Background(), agent, userMessage("hi"))
	if err != nil {
		t.Fatalf("ExecuteStreamContext: %v", err)
	}

	result, err := prompty.Collect(context.Background(),
		anthropic.DecodeStream(context.Background(), raw.(anthropic.RawStream)))
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if len(result.Errors) != 1 || !strings.Contains(result.Errors[0], "overloaded") {
		t.Errorf("errors = %v, want one overloaded message", result.Errors)
	}
}

func TestStreamCloseTerminatesProducer(t *testing.T) {
	released := make(chan struct{})
	served := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w,
			"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"first\"}}\n\n")
		w.(http.Flusher).Flush()
		close(served)
		select {
		case <-r.Context().Done():
		case <-released:
		}
	}))
	defer server.Close()
	defer close(released)

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-ant",
	}, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	executor := &anthropic.Executor{Env: staticEnv(nil), Client: server.Client()}
	raw, err := executor.ExecuteStreamContext(ctx, agent, userMessage("hi"))
	if err != nil {
		t.Fatalf("ExecuteStreamContext: %v", err)
	}
	stream := anthropic.DecodeStream(ctx, raw.(anthropic.RawStream))

	if _, ok := stream.Next(); !ok {
		t.Fatal("expected at least one chunk")
	}
	<-served

	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = stream.Close()
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Stream.Close did not return; the producer goroutine leaked")
	}
}

// ---------------------------------------------------------------------------
// Tool turns end to end
// ---------------------------------------------------------------------------

// TestToolTurnOverHTTP proves the Anthropic tool exchange replays the exact
// content blocks the provider produced, so the tool_use ids the tool_result
// blocks refer to still resolve.
func TestToolTurnOverHTTP(t *testing.T) {
	var requests []map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&body)
		requests = append(requests, body)

		w.Header().Set("Content-Type", "application/json")
		if len(requests) == 1 {
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant",
				"content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}],
				"stop_reason":"tool_use"}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"msg_2","type":"message","role":"assistant",
			"content":[{"type":"text","text":"Paris is sunny."}],"stop_reason":"end_turn"}`)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-ant",
	}, map[string]interface{}{
		"tools": []interface{}{
			map[string]interface{}{
				"name": "get_weather", "kind": "function", "description": "Get weather",
				"parameters": []interface{}{
					map[string]interface{}{"name": "city", "kind": "string", "required": true},
				},
			},
		},
	})

	registry := prompty.NewToolRegistry()
	registry.RegisterText("get_weather", func(_ context.Context, args map[string]interface{}) (string, error) {
		if args["city"] != "Paris" {
			t.Errorf("tool received city=%v", args["city"])
		}
		return "72F sunny", nil
	})

	executor := &anthropic.Executor{Env: staticEnv(nil), Client: server.Client()}
	result, err := prompty.RunMessages(context.Background(), agent, userMessage("weather in Paris?"),
		prompty.RunOptions{Tools: registry, Executor: executor, Processor: anthropic.NewProcessor()})
	if err != nil {
		t.Fatalf("RunMessages: %v", err)
	}

	if result.Text() != "Paris is sunny." {
		t.Errorf("result = %q", result.Text())
	}
	if len(requests) != 2 {
		t.Fatalf("made %d provider calls, want 2", len(requests))
	}

	// The tool definition must use input_schema, not OpenAI's nested shape.
	tools, _ := requests[0]["tools"].([]interface{})
	if len(tools) != 1 {
		t.Fatalf("first request carried %d tools, want 1", len(tools))
	}
	definition, _ := tools[0].(map[string]interface{})
	if definition["input_schema"] == nil || definition["parameters"] != nil {
		t.Errorf("tool definition is not in Anthropic shape: %v", definition)
	}

	second, _ := requests[1]["messages"].([]interface{})
	if len(second) != 3 {
		t.Fatalf("second request carried %d messages, want 3", len(second))
	}

	assistant, _ := second[1].(map[string]interface{})
	if assistant["role"] != "assistant" {
		t.Errorf("message[1] role = %v", assistant["role"])
	}
	blocks, _ := assistant["content"].([]interface{})
	if len(blocks) != 1 {
		t.Fatalf("assistant replay carried %d blocks, want the provider's 1", len(blocks))
	}
	block, _ := blocks[0].(map[string]interface{})
	if block["type"] != "tool_use" || block["id"] != "toolu_1" {
		t.Errorf("assistant replay lost the tool_use block: %v", block)
	}

	toolMessage, _ := second[2].(map[string]interface{})
	if toolMessage["role"] != "user" {
		t.Errorf("tool results must be sent as a user message, got %v", toolMessage["role"])
	}
	results, _ := toolMessage["content"].([]interface{})
	if len(results) != 1 {
		t.Fatalf("tool result message carried %d blocks, want 1", len(results))
	}
	toolResult, _ := results[0].(map[string]interface{})
	if toolResult["type"] != "tool_result" || toolResult["tool_use_id"] != "toolu_1" || toolResult["content"] != "72F sunny" {
		t.Errorf("tool_result block = %v", toolResult)
	}
}
