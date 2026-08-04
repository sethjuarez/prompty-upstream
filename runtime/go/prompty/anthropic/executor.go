package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	model "prompty/model"
	wire "prompty/wire"
)

// DefaultTimeout bounds a non-streaming request. Streaming requests are not
// bounded by it and rely on the caller's context instead.
const DefaultTimeout = 120 * time.Second

// Environment variables consulted when the connection names neither.
const (
	EnvAPIKey   = "ANTHROPIC_API_KEY"
	EnvBaseURL  = "ANTHROPIC_BASE_URL"
	EnvVersion  = "ANTHROPIC_VERSION"
	authHeader  = "x-api-key"
	versionName = "anthropic-version"
)

// Env is the environment lookup used to resolve the endpoint and credential.
// It is injectable so tests never touch the real process environment.
type Env func(key string) (string, bool)

func (e Env) lookup(key string) string {
	if e == nil {
		value, _ := os.LookupEnv(key)
		return value
	}
	value, _ := e(key)
	return value
}

// Request is a fully resolved provider call. Header holds the credential and is
// the only field that does; every diagnostic path reads RedactedHeader instead.
type Request struct {
	Method string
	URL    string
	Header map[string]string
	Body   map[string]interface{}
}

// RedactedHeader returns the headers with the credential masked, for logging.
func (r Request) RedactedHeader() map[string]string {
	out := make(map[string]string, len(r.Header))
	for key, value := range r.Header {
		if strings.EqualFold(key, authHeader) {
			out[key] = wire.Redacted
			continue
		}
		out[key] = value
	}
	return out
}

// String renders the request for diagnostics with the credential removed.
func (r Request) String() string { return r.Method + " " + wire.RedactURL(r.URL) }

// BuildProviderRequest resolves the URL, headers and body for one call. It
// performs no I/O beyond reading the injected environment.
func BuildProviderRequest(agent model.Prompty, messages []model.Message, env Env, stream bool) (Request, error) {
	if apiType := wire.APIType(agent); apiType != "chat" && apiType != "agent" {
		return Request{}, &wire.SchemaError{Message: "unsupported Anthropic apiType: " + apiType}
	}

	body, err := BuildChatRequest(agent, messages)
	if err != nil {
		return Request{}, err
	}
	if stream {
		EnableStreaming(body)
	}

	conn := wire.ConnectionMap(agent)

	base := wire.ConnectionString(conn, "endpoint")
	if base == "" {
		base = env.lookup(EnvBaseURL)
	}
	if base == "" {
		base = DefaultEndpoint
	}

	key := wire.ConnectionString(conn, "apiKey", "api_key", "key")
	if key == "" {
		key = env.lookup(EnvAPIKey)
	}
	if key == "" {
		return Request{}, &wire.ProviderError{
			Provider: "anthropic",
			Op:       "resolve credential",
			Body:     "no credential found; set model.connection.apiKey or " + EnvAPIKey,
		}
	}

	version := env.lookup(EnvVersion)
	if version == "" {
		version = Version
	}

	header := map[string]string{
		"Content-Type": "application/json",
		authHeader:     key,
		versionName:    version,
	}
	if stream {
		header["Accept"] = "text/event-stream"
	}

	return Request{
		Method: "POST",
		URL:    strings.TrimSuffix(base, "/") + MessagesPath,
		Header: header,
		Body:   body,
	}, nil
}

// Executor implements the Prompty Executor contract over the Anthropic Messages
// API. The zero value is usable and reads its credential from the environment.
type Executor struct {
	// Env resolves the endpoint and credential. Nil reads the process
	// environment.
	Env Env
	// Client performs requests. Nil uses a shared client with DefaultTimeout.
	Client *http.Client
	// MaxResponseBytes bounds buffered non-streaming responses. Zero uses
	// wire.DefaultMaxResponseBytes.
	MaxResponseBytes int64
}

// NewExecutor returns an executor with default transport.
func NewExecutor() *Executor { return &Executor{} }

var sharedClient = &http.Client{Timeout: DefaultTimeout}

func (e *Executor) client() *http.Client {
	if e.Client != nil {
		return e.Client
	}
	return sharedClient
}

// streamClient drops the whole-request timeout, which would otherwise abort a
// long generation partway through the stream.
func (e *Executor) streamClient() *http.Client {
	base := e.client()
	if base.Timeout == 0 {
		return base
	}
	clone := *base
	clone.Timeout = 0
	return &clone
}

// Execute implements model.Executor. Prefer ExecuteContext.
func (e *Executor) Execute(agent model.Prompty, messages []model.Message) (interface{}, error) {
	return e.ExecuteContext(context.Background(), agent, messages)
}

// ExecuteStream implements model.Executor. Prefer ExecuteStreamContext.
func (e *Executor) ExecuteStream(agent model.Prompty, messages []model.Message) (interface{}, error) {
	return e.ExecuteStreamContext(context.Background(), agent, messages)
}

// ExecuteContext performs one non-streaming provider call.
func (e *Executor) ExecuteContext(ctx context.Context, agent model.Prompty, messages []model.Message) (interface{}, error) {
	request, err := BuildProviderRequest(agent, messages, e.Env, false)
	if err != nil {
		return nil, err
	}

	response, err := e.send(ctx, e.client(), request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	body, err := wire.ReadResponseBody(response.Body, e.MaxResponseBytes)
	if err != nil {
		return nil, wire.NewProviderError("anthropic", "read response", response.StatusCode, request.URL, "", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, wire.NewProviderError("anthropic", "execute", response.StatusCode, request.URL, string(body), nil)
	}

	var decoded map[string]interface{}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, wire.NewProviderError("anthropic", "decode response", response.StatusCode, request.URL, string(body), err)
	}
	return decoded, nil
}

// ExecuteStreamContext performs one streaming provider call and returns a
// RawStream whose Body the processor closes.
func (e *Executor) ExecuteStreamContext(ctx context.Context, agent model.Prompty, messages []model.Message) (interface{}, error) {
	request, err := BuildProviderRequest(agent, messages, e.Env, true)
	if err != nil {
		return nil, err
	}

	response, err := e.send(ctx, e.streamClient(), request)
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<16))
		response.Body.Close()
		return nil, wire.NewProviderError("anthropic", "execute stream", response.StatusCode, request.URL, string(body), nil)
	}
	return RawStream{Body: response.Body}, nil
}

func (e *Executor) send(ctx context.Context, client *http.Client, request Request) (*http.Response, error) {
	encoded, err := json.Marshal(request.Body)
	if err != nil {
		return nil, wire.NewProviderError("anthropic", "encode request", 0, request.URL, "", err)
	}

	httpRequest, err := http.NewRequestWithContext(ctx, request.Method, request.URL, bytes.NewReader(encoded))
	if err != nil {
		return nil, wire.NewProviderError("anthropic", "build request", 0, request.URL, "", err)
	}
	for key, value := range request.Header {
		httpRequest.Header.Set(key, value)
	}

	response, err := client.Do(httpRequest)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, wire.NewProviderError("anthropic", "transport", 0, request.URL, wire.RedactSecrets(err.Error()), nil)
	}
	return response, nil
}

// FormatToolMessages turns a completed tool round into the messages that
// continue the conversation (spec §9.3).
//
// Anthropic's shape differs from OpenAI's: the assistant turn replays the exact
// content blocks the provider produced — the tool_use ids in them are what the
// following tool_result blocks refer to — and all results are batched into a
// single user message rather than one message per result.
func (e *Executor) FormatToolMessages(
	rawResponse interface{},
	toolCalls []model.ToolCall,
	toolResults []string,
	textContent *string,
) ([]model.Message, error) {
	assistantMetadata := map[string]interface{}{}
	if raw, ok := asMap(rawResponse); ok {
		if content, ok := raw["content"]; ok && content != nil {
			assistantMetadata["content"] = content
		}
	}
	if len(assistantMetadata) == 0 {
		// No raw response to replay: rebuild equivalent tool_use blocks so the
		// following tool_result blocks still have something to refer to.
		blocks := make([]interface{}, 0, len(toolCalls)+1)
		if textContent != nil && *textContent != "" {
			blocks = append(blocks, map[string]interface{}{"type": "text", "text": *textContent})
		}
		for _, call := range toolCalls {
			var input interface{} = map[string]interface{}{}
			if call.Arguments != "" {
				_ = json.Unmarshal([]byte(call.Arguments), &input)
			}
			blocks = append(blocks, map[string]interface{}{
				"type": "tool_use", "id": call.Id, "name": call.Name, "input": input,
			})
		}
		assistantMetadata["content"] = blocks
	}

	assistantText := ""
	if textContent != nil {
		assistantText = *textContent
	}

	results := make([]interface{}, 0, len(toolCalls))
	for i, call := range toolCalls {
		result := ""
		if i < len(toolResults) {
			result = toolResults[i]
		}
		results = append(results, map[string]interface{}{
			"type":        "tool_result",
			"tool_use_id": call.Id,
			"content":     result,
		})
	}

	return []model.Message{
		wire.TextMessage(model.RoleAssistant, assistantText, assistantMetadata),
		wire.TextMessage(model.RoleTool, "", map[string]interface{}{"tool_results": results}),
	}, nil
}
