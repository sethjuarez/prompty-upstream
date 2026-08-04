package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"time"

	model "prompty/model"
	wire "prompty/wire"
)

// DefaultTimeout bounds a non-streaming request. Streaming requests are not
// bounded by it — a long generation is not a hung socket — and rely on the
// caller's context instead.
const DefaultTimeout = 120 * time.Second

// Executor implements the Prompty Executor contract over an OpenAI-compatible
// HTTP endpoint.
//
// The zero value is usable and talks to OpenAI with credentials from the
// process environment. Set Dialect to force Azure or Foundry request shaping,
// Env to supply credentials from somewhere other than the environment, and
// Client to control transport, proxying and timeouts.
type Executor struct {
	// Dialect is the fallback endpoint shape, used only when the agent's
	// connection does not imply one. Empty means plain OpenAI. A connection
	// that names its kind or an Azure host always wins — see ResolveDialect.
	Dialect Dialect
	// Env resolves endpoints and credentials. Nil reads the process
	// environment.
	Env Env
	// Client performs requests. Nil uses a shared client with DefaultTimeout.
	Client *http.Client
	// MaxResponseBytes bounds buffered non-streaming responses. Zero uses
	// wire.DefaultMaxResponseBytes.
	MaxResponseBytes int64
}

// NewExecutor returns an executor with the given fallback dialect. Pass an
// empty dialect to let every agent's connection decide.
func NewExecutor(dialect Dialect) *Executor { return &Executor{Dialect: dialect} }

var sharedClient = &http.Client{Timeout: DefaultTimeout}

func (e *Executor) client() *http.Client {
	if e.Client != nil {
		return e.Client
	}
	return sharedClient
}

// streamClient is the transport used for streaming calls. It reuses the
// caller's client but drops the whole-request timeout, which would otherwise
// abort a long generation mid-stream.
func (e *Executor) streamClient() *http.Client {
	base := e.client()
	if base.Timeout == 0 {
		return base
	}
	clone := *base
	clone.Timeout = 0
	return &clone
}

// Execute implements model.Executor. It exists for the emitted contract;
// prefer ExecuteContext, which can be cancelled.
func (e *Executor) Execute(agent model.Prompty, messages []model.Message) (interface{}, error) {
	return e.ExecuteContext(context.Background(), agent, messages)
}

// ExecuteStream implements model.Executor. Prefer ExecuteStreamContext.
func (e *Executor) ExecuteStream(agent model.Prompty, messages []model.Message) (interface{}, error) {
	return e.ExecuteStreamContext(context.Background(), agent, messages)
}

// ExecuteContext performs one non-streaming provider call.
func (e *Executor) ExecuteContext(ctx context.Context, agent model.Prompty, messages []model.Message) (interface{}, error) {
	request, err := BuildProviderRequest(agent, messages, e.Dialect, e.Env, false)
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
		return nil, wire.NewProviderError("openai", "read response", response.StatusCode, request.URL, "", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, wire.NewProviderError("openai", "execute", response.StatusCode, request.URL, string(body), nil)
	}

	var decoded map[string]interface{}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, wire.NewProviderError("openai", "decode response", response.StatusCode, request.URL, string(body), err)
	}
	return decoded, nil
}

// ExecuteStreamContext performs one streaming provider call and returns a
// RawStream. The caller must either consume it through a Processor or close its
// Body; DecodeStream does the latter automatically.
func (e *Executor) ExecuteStreamContext(ctx context.Context, agent model.Prompty, messages []model.Message) (interface{}, error) {
	request, err := BuildProviderRequest(agent, messages, e.Dialect, e.Env, true)
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
		return nil, wire.NewProviderError("openai", "execute stream", response.StatusCode, request.URL, string(body), nil)
	}

	return RawStream{APIType: wire.APIType(agent), Body: response.Body}, nil
}

func (e *Executor) send(ctx context.Context, client *http.Client, request Request) (*http.Response, error) {
	encoded, err := json.Marshal(request.Body)
	if err != nil {
		return nil, wire.NewProviderError("openai", "encode request", 0, request.URL, "", err)
	}

	httpRequest, err := http.NewRequestWithContext(ctx, request.Method, request.URL, bytes.NewReader(encoded))
	if err != nil {
		return nil, wire.NewProviderError("openai", "build request", 0, request.URL, "", err)
	}
	for key, value := range request.Header {
		httpRequest.Header.Set(key, value)
	}

	response, err := client.Do(httpRequest)
	if err != nil {
		// A cancelled context is the caller's own signal and must surface as
		// such rather than as an opaque transport failure.
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		// http.Client errors embed the request URL, which may carry an
		// api-key query parameter, so the message goes through the redactor.
		return nil, wire.NewProviderError("openai", "transport", 0, request.URL, wire.RedactSecrets(err.Error()), nil)
	}
	return response, nil
}

// FormatToolMessages turns a completed tool round into the messages that
// continue the conversation (spec §9.3).
//
// The shape is one assistant message carrying the tool_calls metadata followed
// by one tool message per result, in request order. Providers reject a
// tool_calls block whose results are missing, reordered or merged.
func (e *Executor) FormatToolMessages(
	rawResponse interface{},
	toolCalls []model.ToolCall,
	toolResults []string,
	textContent *string,
) ([]model.Message, error) {
	if raw, ok := asMap(rawResponse); ok && isResponsesPayload(raw) {
		return formatResponsesToolMessages(raw, toolCalls, toolResults), nil
	}

	messages := make([]model.Message, 0, len(toolCalls)+1)

	wireCalls := make([]interface{}, 0, len(toolCalls))
	for _, call := range toolCalls {
		wireCalls = append(wireCalls, map[string]interface{}{
			"id":   call.Id,
			"type": "function",
			"function": map[string]interface{}{
				"name":      call.Name,
				"arguments": call.Arguments,
			},
		})
	}

	// The assistant message carries the prose the model emitted alongside its
	// tool calls, or an empty string when it emitted none. It is never omitted:
	// the tool messages that follow are only valid as replies to it.
	//
	// The processor returns tool calls rather than text for a tool round, so
	// the caller usually has no prose to pass. Recovering it from the raw
	// response keeps a model that reasons out loud before calling a tool from
	// losing that reasoning on the next turn.
	assistantText := ""
	if textContent != nil {
		assistantText = *textContent
	} else if raw, ok := asMap(rawResponse); ok {
		if message := firstChoiceMessage(raw); message != nil {
			assistantText, _ = message["content"].(string)
		}
	}
	messages = append(messages, wire.TextMessage(model.RoleAssistant, assistantText, map[string]interface{}{
		"tool_calls": wireCalls,
	}))

	for i, call := range toolCalls {
		result := ""
		if i < len(toolResults) {
			result = toolResults[i]
		}
		messages = append(messages, wire.TextMessage(model.RoleTool, result, map[string]interface{}{
			"tool_call_id": call.Id,
			"name":         call.Name,
		}))
	}
	return messages, nil
}

// isResponsesPayload reports whether a raw response came from the Responses
// API. The tool exchange has to be replayed differently there, and the response
// itself is the only reliable signal at this point in the contract — the
// emitted FormatToolMessages signature carries no agent.
func isResponsesPayload(raw map[string]interface{}) bool {
	if raw["object"] == "response" {
		return true
	}
	_, hasOutput := raw["output"]
	_, hasChoices := raw["choices"]
	return hasOutput && !hasChoices
}

// formatResponsesToolMessages replays the provider's own function_call items
// rather than rebuilding them, so a continuation against a previous_response_id
// still matches byte for byte.
func formatResponsesToolMessages(raw map[string]interface{}, toolCalls []model.ToolCall, toolResults []string) []model.Message {
	originals := map[string]interface{}{}
	if output, ok := raw["output"].([]interface{}); ok {
		for _, item := range output {
			entry, ok := item.(map[string]interface{})
			if !ok || entry["type"] != "function_call" {
				continue
			}
			if id := stringField(entry, "call_id", "id"); id != "" {
				originals[id] = entry
			}
		}
	}

	messages := make([]model.Message, 0, len(toolCalls)*2)
	for i, call := range toolCalls {
		item, ok := originals[call.Id]
		if !ok {
			item = map[string]interface{}{
				"type":      "function_call",
				"call_id":   call.Id,
				"name":      call.Name,
				"arguments": call.Arguments,
			}
		}
		messages = append(messages, wire.TextMessage(model.RoleAssistant, "", map[string]interface{}{
			"responses_function_call": item,
		}))

		result := ""
		if i < len(toolResults) {
			result = toolResults[i]
		}
		messages = append(messages, wire.TextMessage(model.RoleTool, result, map[string]interface{}{
			"tool_call_id": call.Id,
			"name":         call.Name,
		}))
	}
	return messages
}
