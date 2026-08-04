package openai_test

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
	model "prompty/model"
	openai "prompty/openai"
	wire "prompty/wire"
)

// staticEnv is an injected environment so tests never mutate the real one and
// never accidentally pick up a developer's live credentials.
func staticEnv(pairs map[string]string) openai.Env {
	return func(key string) (string, bool) {
		value, ok := pairs[key]
		return value, ok
	}
}

func testAgent(t *testing.T, connection map[string]interface{}, extra map[string]interface{}) model.Prompty {
	t.Helper()

	modelSpec := map[string]interface{}{
		"id": "gpt-4o-mini", "provider": "openai", "apiType": "chat",
	}
	if connection != nil {
		modelSpec["connection"] = connection
	}
	data := map[string]interface{}{
		"name": "integration", "instructions": "", "model": modelSpec,
	}
	for key, value := range extra {
		if key == "options" {
			modelSpec["options"] = value
			continue
		}
		if key == "apiType" {
			modelSpec["apiType"] = value
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

func TestEndpointAndHeaderConstruction(t *testing.T) {
	cases := []struct {
		name       string
		dialect    openai.Dialect
		connection map[string]interface{}
		env        map[string]string
		apiType    string
		wantURL    string
		wantHeader string
		wantValue  string
	}{
		{
			name:       "openai default endpoint from env credential",
			connection: nil,
			env:        map[string]string{openai.EnvOpenAIAPIKey: "sk-test"},
			wantURL:    "https://api.openai.com/v1/chat/completions",
			wantHeader: "Authorization",
			wantValue:  "Bearer sk-test",
		},
		{
			name:       "openai connection endpoint wins over env base url",
			connection: map[string]interface{}{"kind": "key", "endpoint": "https://proxy.example.com", "apiKey": "sk-conn"},
			env:        map[string]string{openai.EnvOpenAIBaseURL: "https://ignored.example.com"},
			wantURL:    "https://proxy.example.com/v1/chat/completions",
			wantHeader: "Authorization",
			wantValue:  "Bearer sk-conn",
		},
		{
			name:       "openai base already carrying v1 is not doubled",
			connection: map[string]interface{}{"kind": "key", "endpoint": "https://gateway.example.com/openai/v1", "apiKey": "sk-conn"},
			wantURL:    "https://gateway.example.com/openai/v1/chat/completions",
			wantHeader: "Authorization",
			wantValue:  "Bearer sk-conn",
		},
		{
			name:       "openai trailing slash is trimmed",
			connection: map[string]interface{}{"kind": "key", "endpoint": "https://proxy.example.com/", "apiKey": "sk-conn"},
			wantURL:    "https://proxy.example.com/v1/chat/completions",
			wantHeader: "Authorization",
			wantValue:  "Bearer sk-conn",
		},
		{
			name:       "azure inferred from hostname uses deployment path and api-key header",
			connection: map[string]interface{}{"kind": "key", "endpoint": "https://myresource.openai.azure.com", "apiKey": "azure-key"},
			wantURL:    "https://myresource.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=" + openai.DefaultAzureAPIVersion,
			wantHeader: "api-key",
			wantValue:  "azure-key",
		},
		{
			name:    "azure api version override from model options",
			dialect: openai.DialectAzure,
			connection: map[string]interface{}{
				"kind": "key", "endpoint": "https://myresource.openai.azure.com", "apiKey": "azure-key",
			},
			wantURL:    "https://myresource.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2099-01-01",
			wantHeader: "api-key",
			wantValue:  "azure-key",
		},
		{
			name: "foundry project endpoint is rewritten to the inference endpoint",
			connection: map[string]interface{}{
				"kind":     "foundry",
				"endpoint": "https://myresource.services.ai.azure.com/api/projects/my-project",
			},
			env:        map[string]string{openai.EnvInferenceCredential: "foundry-token"},
			wantURL:    "https://myresource.openai.azure.com/openai/v1/chat/completions",
			wantHeader: "Authorization",
			wantValue:  "Bearer foundry-token",
		},
		{
			name: "foundry inference endpoint is idempotent",
			connection: map[string]interface{}{
				"kind": "foundry", "endpoint": "https://myresource.openai.azure.com/openai/v1",
			},
			env:        map[string]string{openai.EnvInferenceCredential: "foundry-token"},
			wantURL:    "https://myresource.openai.azure.com/openai/v1/chat/completions",
			wantHeader: "Authorization",
			wantValue:  "Bearer foundry-token",
		},
		{
			name:       "responses api path",
			apiType:    "responses",
			connection: map[string]interface{}{"kind": "key", "endpoint": "https://api.openai.com", "apiKey": "sk-test"},
			wantURL:    "https://api.openai.com/v1/responses",
			wantHeader: "Authorization",
			wantValue:  "Bearer sk-test",
		},
		{
			name:       "embedding api path",
			apiType:    "embedding",
			connection: map[string]interface{}{"kind": "key", "endpoint": "https://api.openai.com", "apiKey": "sk-test"},
			wantURL:    "https://api.openai.com/v1/embeddings",
			wantHeader: "Authorization",
			wantValue:  "Bearer sk-test",
		},
		{
			name:       "image api path",
			apiType:    "image",
			connection: map[string]interface{}{"kind": "key", "endpoint": "https://api.openai.com", "apiKey": "sk-test"},
			wantURL:    "https://api.openai.com/v1/images/generations",
			wantHeader: "Authorization",
			wantValue:  "Bearer sk-test",
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			extra := map[string]interface{}{}
			if testCase.apiType != "" {
				extra["apiType"] = testCase.apiType
			}
			if strings.Contains(testCase.name, "api version override") {
				extra["options"] = map[string]interface{}{
					"additionalProperties": map[string]interface{}{"apiVersion": "2099-01-01"},
				}
			}

			agent := testAgent(t, testCase.connection, extra)
			request, err := openai.BuildProviderRequest(
				agent, userMessage("hi"), testCase.dialect, staticEnv(testCase.env), false)
			if err != nil {
				t.Fatalf("BuildProviderRequest: %v", err)
			}

			if request.URL != testCase.wantURL {
				t.Errorf("URL = %q, want %q", request.URL, testCase.wantURL)
			}
			if got := request.Header[testCase.wantHeader]; got != testCase.wantValue {
				t.Errorf("header %s = %q, want %q", testCase.wantHeader, got, testCase.wantValue)
			}
			if request.AuthHeader != testCase.wantHeader {
				t.Errorf("AuthHeader = %q, want %q", request.AuthHeader, testCase.wantHeader)
			}
		})
	}
}

// TestCredentialsNeverReachDiagnostics is the security assertion: no rendering
// path that a host might log may contain the key.
func TestCredentialsNeverReachDiagnostics(t *testing.T) {
	const secret = "sk-super-secret-value"

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": "https://api.openai.com", "apiKey": secret,
	}, nil)

	request, err := openai.BuildProviderRequest(agent, userMessage("hi"), "", staticEnv(nil), false)
	if err != nil {
		t.Fatalf("BuildProviderRequest: %v", err)
	}

	if !strings.Contains(request.Header["Authorization"], secret) {
		t.Fatal("precondition failed: the credential should be in the Authorization header")
	}
	if got := request.String(); strings.Contains(got, secret) {
		t.Errorf("Request.String leaked the credential: %s", got)
	}
	for key, value := range request.RedactedHeader() {
		if strings.Contains(value, secret) {
			t.Errorf("RedactedHeader leaked the credential in %s", key)
		}
	}
	if got := request.RedactedHeader()["Authorization"]; got != wire.Redacted {
		t.Errorf("RedactedHeader Authorization = %q, want %q", got, wire.Redacted)
	}

	// An Azure key placed in the query string is masked too.
	azureURL := "https://r.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-01-01&api-key=" + secret
	if redacted := wire.RedactURL(azureURL); strings.Contains(redacted, secret) {
		t.Errorf("RedactURL leaked a query credential: %s", redacted)
	}
}

// TestErrorBodyIsRedactedAndBounded proves a provider error body cannot smuggle
// a credential or an unbounded blob into a host's logs.
func TestErrorBodyIsRedactedAndBounded(t *testing.T) {
	const secret = "sk-leaked-in-body"
	body := `{"error":{"message":"bad"},"api_key":"` + secret + `","padding":"` + strings.Repeat("x", 8192) + `"}`

	err := wire.NewProviderError("openai", "execute", 401, "https://api.openai.com/v1/chat/completions", body, nil)
	rendered := err.Error()

	if strings.Contains(rendered, secret) {
		t.Errorf("provider error leaked a credential: %s", rendered)
	}
	if len(rendered) > 4096 {
		t.Errorf("provider error is %d bytes; diagnostic bodies must be bounded", len(rendered))
	}
	if !errors.Is(err, wire.ErrProvider) {
		t.Error("provider error does not wrap wire.ErrProvider")
	}
}

// ---------------------------------------------------------------------------
// HTTP execution
// ---------------------------------------------------------------------------

func TestExecuteAgainstTestServer(t *testing.T) {
	var (
		gotPath   string
		gotAuth   string
		gotBody   map[string]interface{}
		gotAccept string
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotAccept = r.Header.Get("Accept")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"pong"}}],
			"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-test",
	}, nil)

	executor := &openai.Executor{Env: staticEnv(nil), Client: server.Client()}
	raw, err := executor.ExecuteContext(context.Background(), agent, userMessage("ping"))
	if err != nil {
		t.Fatalf("ExecuteContext: %v", err)
	}

	if gotPath != "/v1/chat/completions" {
		t.Errorf("path = %q, want /v1/chat/completions", gotPath)
	}
	if gotAuth != "Bearer sk-test" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	if gotAccept != "" {
		t.Errorf("non-streaming request should not ask for SSE, got Accept=%q", gotAccept)
	}
	if gotBody["model"] != "gpt-4o-mini" {
		t.Errorf("request model = %v", gotBody["model"])
	}
	if _, streaming := gotBody["stream"]; streaming {
		t.Error("non-streaming request must not set stream")
	}

	result, err := openai.NewProcessor().Process(agent, raw)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if result != "pong" {
		t.Errorf("result = %#v, want \"pong\"", result)
	}

	usage, ok := openai.UsageOf(raw.(map[string]interface{}))
	if !ok || usage.TotalTokens != 4 {
		t.Errorf("usage = %+v, ok=%v", usage, ok)
	}
}

func TestExecutePropagatesHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"message":"rate limited","type":"rate_limit_error"}}`)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-test",
	}, nil)

	executor := &openai.Executor{Env: staticEnv(nil), Client: server.Client()}
	_, err := executor.ExecuteContext(context.Background(), agent, userMessage("ping"))
	if err == nil {
		t.Fatal("expected an error for HTTP 429")
	}

	var providerErr *wire.ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("error is %T, want *wire.ProviderError", err)
	}
	if providerErr.Status != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429", providerErr.Status)
	}
	if !strings.Contains(providerErr.Error(), "rate limited") {
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
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-test",
	}, nil)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	executor := &openai.Executor{Env: staticEnv(nil), Client: server.Client()}
	_, err := executor.ExecuteContext(ctx, agent, userMessage("ping"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
}

func TestMissingCredentialIsAnError(t *testing.T) {
	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": "https://api.openai.com",
	}, nil)

	_, err := openai.BuildProviderRequest(agent, userMessage("hi"), "", staticEnv(nil), false)
	if err == nil {
		t.Fatal("expected an error when no credential is configured")
	}
	if !strings.Contains(err.Error(), openai.EnvOpenAIAPIKey) {
		t.Errorf("error should name the environment variable to set: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Structured output over HTTP
// ---------------------------------------------------------------------------

func TestStructuredOutputRoundTrip(t *testing.T) {
	var gotBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"index":0,"message":{"role":"assistant",
			"content":"{\"city\":\"Paris\",\"temp\":21,\"note\":null}"}}]}`)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-test",
	}, map[string]interface{}{
		"outputs": []interface{}{
			map[string]interface{}{"name": "city", "kind": "string", "required": true},
			map[string]interface{}{"name": "temp", "kind": "integer", "required": true},
			map[string]interface{}{"name": "note", "kind": "string"},
		},
	})

	executor := &openai.Executor{Env: staticEnv(nil), Client: server.Client()}
	raw, err := executor.ExecuteContext(context.Background(), agent, userMessage("weather"))
	if err != nil {
		t.Fatalf("ExecuteContext: %v", err)
	}

	// The request must carry a strict json_schema with every output required
	// and the optional one widened to admit null.
	format, _ := gotBody["response_format"].(map[string]interface{})
	schemaEnvelope, _ := format["json_schema"].(map[string]interface{})
	if schemaEnvelope["strict"] != true {
		t.Errorf("json_schema.strict = %v, want true", schemaEnvelope["strict"])
	}
	schema, _ := schemaEnvelope["schema"].(map[string]interface{})
	if schema["additionalProperties"] != false {
		t.Errorf("additionalProperties = %v, want false", schema["additionalProperties"])
	}
	required, _ := schema["required"].([]interface{})
	if len(required) != 3 {
		t.Errorf("required = %v, want all three outputs", required)
	}
	properties, _ := schema["properties"].(map[string]interface{})
	note, _ := properties["note"].(map[string]interface{})
	assertJSONEqual(t, "optional output type", note["type"], []interface{}{"string", "null"})

	result, err := openai.NewProcessor().Process(agent, raw)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	decoded, ok := result.(map[string]interface{})
	if !ok {
		t.Fatalf("result is %T, want a decoded object", result)
	}
	if decoded["city"] != "Paris" {
		t.Errorf("city = %#v", decoded["city"])
	}
	if temp, ok := decoded["temp"].(int64); !ok || temp != 21 {
		t.Errorf("temp = %#v, want int64(21)", decoded["temp"])
	}
	if note, present := decoded["note"]; !present || note != nil {
		t.Errorf("note = %#v, want an explicit nil", note)
	}
}

// ---------------------------------------------------------------------------
// Streaming over HTTP
// ---------------------------------------------------------------------------

const chatStreamBody = "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n" +
	"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n" +
	"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n\n" +
	"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n" +
	"data: [DONE]\n\n"

func TestStreamingChat(t *testing.T) {
	var gotBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, chatStreamBody)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-test",
	}, nil)

	executor := &openai.Executor{Env: staticEnv(nil), Client: server.Client()}
	raw, err := executor.ExecuteStreamContext(context.Background(), agent, userMessage("hi"))
	if err != nil {
		t.Fatalf("ExecuteStreamContext: %v", err)
	}

	if gotBody["stream"] != true {
		t.Errorf("stream = %v, want true", gotBody["stream"])
	}
	options, _ := gotBody["stream_options"].(map[string]interface{})
	if options["include_usage"] != true {
		t.Errorf("stream_options.include_usage = %v, want true", options["include_usage"])
	}

	stream, err := openai.NewProcessor().ProcessStreamContext(context.Background(), raw)
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
	if result.Thinking != "thinking" {
		t.Errorf("thinking = %q", result.Thinking)
	}
	if result.Usage.TotalTokens != 7 {
		t.Errorf("usage = %+v", result.Usage)
	}
}

const toolStreamBody = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n" +
	"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Paris\\\"}\"}}]}}]}\n\n" +
	"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_b\",\"function\":{\"name\":\"get_time\",\"arguments\":\"{}\"}}]}}]}\n\n" +
	"data: [DONE]\n\n"

// TestStreamingToolCallsPreserveOrder proves fragmented tool calls reassemble
// and come back in request order, which is what the next turn depends on.
func TestStreamingToolCallsPreserveOrder(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, toolStreamBody)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-test",
	}, nil)

	executor := &openai.Executor{Env: staticEnv(nil), Client: server.Client()}
	raw, err := executor.ExecuteStreamContext(context.Background(), agent, userMessage("hi"))
	if err != nil {
		t.Fatalf("ExecuteStreamContext: %v", err)
	}
	stream, err := openai.NewProcessor().ProcessStreamContext(context.Background(), raw)
	if err != nil {
		t.Fatalf("ProcessStreamContext: %v", err)
	}

	result, err := prompty.Collect(context.Background(), stream.(*prompty.Stream))
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if len(result.ToolCalls) != 2 {
		t.Fatalf("tool calls = %d, want 2: %+v", len(result.ToolCalls), result.ToolCalls)
	}
	if result.ToolCalls[0].Id != "call_a" || result.ToolCalls[1].Id != "call_b" {
		t.Errorf("tool call order = %v", result.ToolCalls)
	}
	if result.ToolCalls[0].Arguments != `{"city":"Paris"}` {
		t.Errorf("reassembled arguments = %q", result.ToolCalls[0].Arguments)
	}
}

func TestStreamingErrorStatusPropagates(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"message":"invalid key"}}`)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-test",
	}, nil)

	executor := &openai.Executor{Env: staticEnv(nil), Client: server.Client()}
	_, err := executor.ExecuteStreamContext(context.Background(), agent, userMessage("hi"))
	if err == nil {
		t.Fatal("expected an error for HTTP 401")
	}

	var providerErr *wire.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Status != http.StatusUnauthorized {
		t.Fatalf("error = %v, want a 401 provider error", err)
	}
}

// TestStreamCancellationClosesBody proves that abandoning a stream tears down
// the socket and the producer goroutine rather than leaking both.
func TestStreamCancellationClosesBody(t *testing.T) {
	released := make(chan struct{})
	served := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n")
		w.(http.Flusher).Flush()
		close(served)
		// Hold the response open until the client goes away.
		select {
		case <-r.Context().Done():
		case <-released:
		}
	}))
	defer server.Close()
	defer close(released)

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-test",
	}, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	executor := &openai.Executor{Env: staticEnv(nil), Client: server.Client()}
	raw, err := executor.ExecuteStreamContext(ctx, agent, userMessage("hi"))
	if err != nil {
		t.Fatalf("ExecuteStreamContext: %v", err)
	}
	stream := openai.DecodeStream(ctx, raw.(openai.RawStream))

	chunk, ok := stream.Next()
	if !ok {
		t.Fatal("expected at least one chunk")
	}
	if text, _ := prompty.ChunkText(chunk); text != "first" {
		t.Errorf("first chunk = %#v", chunk)
	}
	<-served

	// Close must return promptly: it cancels the producer, unblocks its pending
	// send and waits for the goroutine to exit.
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

// TestToolTurnOverHTTP runs a full two-iteration turn against a test server,
// exercising the registry, the executor, the processor and the message
// formatter together.
func TestToolTurnOverHTTP(t *testing.T) {
	var requests []map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&body)
		requests = append(requests, body)

		w.Header().Set("Content-Type", "application/json")
		if len(requests) == 1 {
			_, _ = io.WriteString(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":null,
				"tool_calls":[{"id":"call_1","type":"function",
				"function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}}]}`)
			return
		}
		_, _ = io.WriteString(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"Paris is sunny."}}]}`)
	}))
	defer server.Close()

	agent := testAgent(t, map[string]interface{}{
		"kind": "key", "endpoint": server.URL, "apiKey": "sk-test",
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

	executor := &openai.Executor{Env: staticEnv(nil), Client: server.Client()}
	result, err := prompty.RunMessages(context.Background(), agent, userMessage("weather in Paris?"),
		prompty.RunOptions{Tools: registry, Executor: executor, Processor: openai.NewProcessor()})
	if err != nil {
		t.Fatalf("RunMessages: %v", err)
	}

	if result.Text() != "Paris is sunny." {
		t.Errorf("result = %q", result.Text())
	}
	if result.Iterations != 2 {
		t.Errorf("iterations = %d, want 2", result.Iterations)
	}
	if len(requests) != 2 {
		t.Fatalf("made %d provider calls, want 2", len(requests))
	}

	// The second request must replay the assistant tool_calls message followed
	// by the tool result, or the provider rejects it. The conversation starts
	// with a single user message, so the round adds messages 1 and 2.
	second, _ := requests[1]["messages"].([]interface{})
	if len(second) != 3 {
		t.Fatalf("second request carried %d messages, want 3: %s", len(second), mustEncodeJSON(t, second))
	}
	assistant, _ := second[1].(map[string]interface{})
	if assistant["role"] != "assistant" || assistant["tool_calls"] == nil {
		t.Errorf("message[1] is not an assistant tool_calls message: %v", assistant)
	}
	tool, _ := second[2].(map[string]interface{})
	if tool["role"] != "tool" || tool["tool_call_id"] != "call_1" || tool["content"] != "72F sunny" {
		t.Errorf("message[2] is not the tool result: %v", tool)
	}
}

func mustEncodeJSON(t *testing.T, v interface{}) string {
	t.Helper()

	encoded, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(encoded)
}
