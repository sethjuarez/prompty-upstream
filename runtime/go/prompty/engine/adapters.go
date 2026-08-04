package engine

import (
	"context"
	"encoding/json"
	"fmt"

	prompty "prompty"
	model "prompty/model"
)

// Adapters that bridge the engine's ports to the existing Prompty runtime, so
// the durable engine reuses the provider executors, processors and tool
// registry rather than growing a second copy of them.

// RegistryToolPort executes engine tool requests through a prompty.ToolRegistry.
//
// It is the bridge between the two tool vocabularies: the engine speaks
// model.ModelToolRequest and model.ModelToolResult, the registry speaks
// model.ToolCall and prompty.ToolDispatch. Nothing else about tool execution
// differs, so the registry's argument decoding, binding injection and error
// handling are reused verbatim.
type RegistryToolPort struct {
	// Registry holds the tool implementations. A nil registry fails any
	// request, matching the practical loop: a tool nobody registered is a
	// capability the model invented.
	Registry *prompty.ToolRegistry
	// Bindings are the per-tool bound arguments, keyed by tool name, as
	// produced by prompty.AgentBindings.
	Bindings map[string]map[string]interface{}
}

// Execute dispatches one tool request.
//
// A tool that runs and fails becomes a failed model.ModelToolResult, because
// the model must be told. A tool that could not be dispatched at all — an
// unregistered name — is returned as an error, which fails the turn: feeding
// the model a synthetic result would teach it the capability exists.
func (p RegistryToolPort) Execute(
	ctx context.Context,
	request model.ModelToolRequest,
) (model.ModelToolResult, error) {
	if p.Registry == nil {
		return model.ModelToolResult{}, fmt.Errorf("engine: no tool registry configured for %q", request.Name)
	}

	arguments, err := encodeArguments(request.Arguments)
	if err != nil {
		return model.ModelToolResult{}, fmt.Errorf("engine: encode arguments for %q: %w", request.Name, err)
	}
	call := model.ToolCall{Id: request.Id, Name: request.Name, Arguments: arguments}

	dispatch, err := p.Registry.Dispatch(ctx, call, p.Bindings[request.Name], nil)
	if err != nil {
		return model.ModelToolResult{}, err
	}

	output := interface{}(dispatch.Text())
	result := model.ModelToolResult{
		RequestId: request.Id,
		Name:      request.Name,
		Outcome:   model.ModelToolOutcomeSuccess,
		Output:    &output,
	}
	if dispatch.Denied {
		errorKind := ErrorKindPermissionDenied
		result.Outcome = model.ModelToolOutcomeFailed
		result.ErrorKind = &errorKind
	}
	return result, nil
}

// encodeArguments renders engine tool arguments as the JSON string the registry
// decodes. Absent arguments become "{}" rather than an empty string, because a
// tool with no parameters still receives an object.
func encodeArguments(arguments *interface{}) (string, error) {
	if arguments == nil {
		return "{}", nil
	}
	if text, ok := (*arguments).(string); ok {
		return text, nil
	}
	encoded, err := json.Marshal(*arguments)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

// PermitPort adapts a prompty.PermissionFunc to the engine's PermissionPort, so
// a host writes one permission policy and uses it for both the practical loop
// and the durable engine.
type PermitPort struct {
	// Permit rules on each call. Nil approves everything.
	Permit prompty.PermissionFunc
}

// Authorize consults the wrapped permission function.
func (p PermitPort) Authorize(
	ctx context.Context,
	request model.ModelToolRequest,
) (model.EnginePermissionDecision, error) {
	if p.Permit == nil {
		return model.EnginePermissionDecision{Approved: true}, nil
	}

	arguments := map[string]interface{}{}
	if request.Arguments != nil {
		if decoded, ok := (*request.Arguments).(map[string]interface{}); ok {
			arguments = decoded
		}
	}
	call := model.ToolCall{Id: request.Id, Name: request.Name}
	decision := p.Permit(ctx, call, arguments)

	engineDecision := model.EnginePermissionDecision{Approved: decision.Allowed}
	if decision.Reason != "" {
		reason := decision.Reason
		engineDecision.Reason = &reason
	}
	return engineDecision, nil
}

// ProviderModelPort invokes a provider through the existing Executor and
// Processor contracts, which is what makes the engine provider-neutral without
// knowing any wire format.
//
// The executor is handed the snapshot's messages and the processor turns the
// raw response into either tool requests or a final output — exactly the split
// the practical loop makes, so a provider that works with prompty.Run works
// with the engine unchanged.
type ProviderModelPort struct {
	// Agent selects the provider and carries its options.
	Agent model.Prompty
	// Executor and Processor override registry lookup. Leaving both nil
	// resolves them from the agent's provider.
	Executor  prompty.Executor
	Processor prompty.Processor
}

// Invoke performs one provider call and normalizes its response.
func (p ProviderModelPort) Invoke(
	ctx context.Context,
	request model.ModelInvocationRequest,
) (model.ModelInvocationResponse, error) {
	executor := p.Executor
	processor := p.Processor

	var err error
	if executor == nil {
		if executor, err = prompty.ExecutorFor(p.Agent); err != nil {
			return model.ModelInvocationResponse{}, err
		}
	}
	if processor == nil {
		if processor, err = prompty.ProcessorFor(p.Agent); err != nil {
			return model.ModelInvocationResponse{}, err
		}
	}
	if err := ctx.Err(); err != nil {
		return model.ModelInvocationResponse{}, err
	}

	raw, err := executor.Execute(p.Agent, request.Context.Messages)
	if err != nil {
		return model.ModelInvocationResponse{}, err
	}
	processed, err := processor.Process(p.Agent, raw)
	if err != nil {
		return model.ModelInvocationResponse{}, err
	}

	response := model.ModelInvocationResponse{
		// The raw provider response is carried forward so the conversation port
		// can format the follow-up messages the provider expects, which needs
		// the original object and not just the processed view.
		Metadata: map[string]interface{}{rawResponseKey: raw},
	}

	toolCalls, hasToolCalls := prompty.AsToolCalls(processed)
	if !hasToolCalls {
		output := processed
		response.Output = &output
		return response, nil
	}

	response.ToolRequests = make([]model.ModelToolRequest, 0, len(toolCalls))
	for _, call := range toolCalls {
		arguments, err := prompty.DecodeToolArguments(call.Arguments)
		if err != nil {
			// A provider that emitted unparseable arguments is a provider
			// failure, not a tool failure: dispatching with a guess would run
			// the tool with something the model did not ask for.
			return model.ModelInvocationResponse{}, fmt.Errorf(
				"engine: decode arguments for tool %q: %w", call.Name, err)
		}
		boxed := interface{}(arguments)
		response.ToolRequests = append(response.ToolRequests, model.ModelToolRequest{
			Id:        call.Id,
			Name:      call.Name,
			Arguments: &boxed,
			Metadata:  map[string]interface{}{toolCallKey: call},
		})
	}
	return response, nil
}

// Metadata keys the provider adapters use to carry provider-shaped values
// across the engine, which is deliberately ignorant of them.
const (
	rawResponseKey = "promptyRawResponse"
	toolCallKey    = "promptyToolCall"
)

// ProviderConversationPort formats a tool exchange through the provider's own
// Executor.FormatToolMessages, so the assistant and tool messages are spelled
// exactly the way that provider requires.
type ProviderConversationPort struct {
	// Agent selects the provider.
	Agent model.Prompty
	// Executor overrides registry lookup. Nil resolves from the agent.
	Executor prompty.Executor
}

// FormatToolExchange renders one completed round as provider-valid messages.
func (p ProviderConversationPort) FormatToolExchange(
	response model.ModelInvocationResponse,
	results []model.ModelToolResult,
) ([]model.Message, error) {
	executor := p.Executor
	if executor == nil {
		var err error
		if executor, err = prompty.ExecutorFor(p.Agent); err != nil {
			return nil, err
		}
	}

	raw := response.Metadata[rawResponseKey]

	calls := make([]model.ToolCall, 0, len(response.ToolRequests))
	for _, request := range response.ToolRequests {
		if call, ok := request.Metadata[toolCallKey].(model.ToolCall); ok {
			calls = append(calls, call)
			continue
		}
		arguments, err := encodeArguments(request.Arguments)
		if err != nil {
			return nil, err
		}
		calls = append(calls, model.ToolCall{Id: request.Id, Name: request.Name, Arguments: arguments})
	}

	// Results are handed back in request order. A provider rejects a tool_calls
	// block whose results do not line up with the calls it made.
	texts := make([]string, 0, len(results))
	for _, result := range results {
		texts = append(texts, resultText(result))
	}

	return executor.FormatToolMessages(raw, calls, texts, nil)
}

func resultText(result model.ModelToolResult) string {
	if result.Output == nil {
		return ""
	}
	if text, ok := (*result.Output).(string); ok {
		return text
	}
	encoded, err := json.Marshal(*result.Output)
	if err != nil {
		return fmt.Sprint(*result.Output)
	}
	return string(encoded)
}
