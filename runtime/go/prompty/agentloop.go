package prompty

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	model "prompty/model"
	wire "prompty/wire"
)

// DefaultMaxIterations bounds an agent turn loop. Each iteration is one model
// call plus the tool round it requests; without a bound a model that keeps
// asking for tools would spin forever (spec §9.3).
const DefaultMaxIterations = 10

// Event kinds emitted by the turn loop, matching the shared agent vectors.
const (
	EventStatus         = "status"
	EventToolCallStart  = "tool_call_start"
	EventToolResult     = "tool_result"
	EventMessagesUpdate = "messages_updated"
	EventDone           = "done"
	EventCancelled      = "cancelled"
	EventError          = "error"
)

// Event is one observation from the turn loop.
type Event struct {
	Type string
	Data map[string]interface{}
}

// EventFunc receives loop events in order. It is called synchronously on the
// loop's goroutine, so a slow handler slows the turn; hosts that need to fan out
// should hand off to their own queue.
type EventFunc func(Event)

// ErrMaxIterations marks a turn that hit its iteration bound.
var ErrMaxIterations = errors.New("prompty: agent loop exceeded max iterations")

// MaxIterationsError reports the bound that was exceeded.
type MaxIterationsError struct{ Max int }

func (e *MaxIterationsError) Error() string {
	return fmt.Sprintf("Agent loop exceeded %d iterations", e.Max)
}

func (e *MaxIterationsError) Unwrap() error { return ErrMaxIterations }

// RunOptions configures a turn loop.
type RunOptions struct {
	// Tools is the dispatch registry. A nil registry means the model may not
	// call tools; if it tries, the turn fails with *ToolNotRegisteredError.
	Tools *ToolRegistry
	// Inputs are the enclosing agent's inputs, used to resolve tool bindings.
	Inputs map[string]interface{}
	// MaxIterations overrides DefaultMaxIterations when positive.
	MaxIterations int
	// Permit rules on each tool call before it runs. A nil Permit allows all.
	Permit PermissionFunc
	// OnEvent observes the loop, may be nil.
	OnEvent EventFunc
	// Executor and Processor override registry lookup. Both must be set
	// together or neither; leaving them nil resolves from the agent's provider.
	Executor  Executor
	Processor Processor

	// ContextBudget trims the conversation to a character budget before each
	// model call (spec §13.3). Zero or negative disables trimming, which is the
	// default: a host that has not measured its budget is better served by the
	// provider's own error than by silently dropping history.
	ContextBudget int
	// Compaction replaces the mechanical summary of trimmed messages with a
	// better one. It is only consulted when ContextBudget actually dropped
	// something, and a failure leaves the mechanical summary in place.
	Compaction CompactionFunc
	// Guardrails are optional host policy hooks (spec §13.4). A nil Guardrails,
	// or any nil field within it, allows everything.
	Guardrails *Guardrails
	// Steering is an optional queue of messages to inject between iterations
	// (spec §13.5). It is drained before every model call after the first, so
	// steering sent during iteration N is seen by iteration N+1.
	Steering *Steering
}

func (o RunOptions) maxIterations() int {
	if o.MaxIterations > 0 {
		return o.MaxIterations
	}
	return DefaultMaxIterations
}

func (o RunOptions) emit(event Event) {
	if o.OnEvent != nil {
		o.OnEvent(event)
	}
}

// RunResult is the outcome of a turn loop.
type RunResult struct {
	// Output is the processed final result: a string for text, or the decoded
	// structured value when the agent declares outputs.
	Output interface{}
	// Messages is the full conversation including everything the loop appended.
	Messages []model.Message
	// Iterations is how many model calls were made.
	Iterations int
	// Dispatches records every tool call in execution order.
	Dispatches []ToolDispatch
	// ContextDropped counts messages removed by context trimming across the
	// whole turn. It is reported rather than swallowed so a host can tell a
	// short answer caused by lost history from a short answer the model meant.
	ContextDropped int
	// SteeringInjected counts messages injected between iterations.
	SteeringInjected int
}

// Text renders the output as a string when it is one, else "".
func (r RunResult) Text() string {
	if text, ok := r.Output.(string); ok {
		return text
	}
	return ""
}

// Run executes the practical Prompty agent pipeline: render and parse the
// agent, then drive model calls and tool rounds until the model answers.
//
// This is the host-facing surface. It is deliberately not the emitted
// ReferenceTurnRunner, which models the low-level checkpoint/permission/journal
// protocol; Run composes the same provider-neutral Executor and Processor
// contracts into the loop an application actually wants.
func Run(ctx context.Context, agent model.Prompty, inputs map[string]interface{}, options RunOptions) (RunResult, error) {
	messages, err := PrepareWithContext(ctx, agent, inputs)
	if err != nil {
		return RunResult{}, err
	}
	if options.Inputs == nil {
		options.Inputs = inputs
	}
	return RunMessages(ctx, agent, messages, options)
}

// RunMessages drives the turn loop over an already-prepared conversation.
//
// Hosts that maintain their own history — a chat session, a resumed thread —
// call this directly instead of Run so the loop does not re-render the prompt.
func RunMessages(ctx context.Context, agent model.Prompty, messages []model.Message, options RunOptions) (RunResult, error) {
	executor, processor, err := resolveInvokers(agent, options)
	if err != nil {
		return RunResult{}, err
	}

	bindings := AgentBindings(agent, options.Inputs)
	maxIterations := options.maxIterations()

	result := RunResult{Messages: append([]model.Message(nil), messages...)}
	announced := false

	for iteration := 1; ; iteration++ {
		// The cancellation check is at the top of the iteration so a context
		// cancelled between rounds stops before another provider call is paid
		// for, which is what the shared cancellation vectors require.
		if err := ctx.Err(); err != nil {
			options.emit(Event{Type: EventCancelled, Data: map[string]interface{}{
				"reason": cancellationReason(iteration),
			}})
			return result, err
		}

		if iteration > maxIterations {
			result.Iterations = iteration
			err := &MaxIterationsError{Max: maxIterations}
			options.emit(Event{Type: EventError, Data: map[string]interface{}{"message": err.Error()}})
			return result, err
		}

		// Steering is drained between iterations, never before the first model
		// call. A queue filled before the turn started is not skipped — it is
		// injected at the next iteration boundary, which is what the shared
		// steering vectors describe (inject_before_iteration: 2). Anything the
		// caller wants in the opening prompt belongs in the prompt, because
		// the first model call has already been built by the time the loop
		// runs.
		if iteration > 1 {
			if injected := options.Steering.Drain(); len(injected) > 0 {
				result.Messages = append(result.Messages, injected...)
				result.SteeringInjected += len(injected)
				options.emit(Event{Type: EventStatus, Data: map[string]interface{}{
					"message": "Injecting steering message",
					"count":   len(injected),
				}})
				options.emit(Event{Type: EventMessagesUpdate, Data: map[string]interface{}{
					"message_count": len(result.Messages),
				}})
			}
		}

		// Trimming runs after steering so the freshly injected instruction is
		// weighed against the budget like any other message, and before the
		// input guardrail so the guardrail sees exactly what the provider will.
		if options.ContextBudget > 0 {
			trimmed, dropped := applyContextPolicy(ctx, result.Messages, options.ContextBudget, options.Compaction)
			result.Messages = trimmed
			result.ContextDropped += dropped
		}

		if verdict := options.Guardrails.CheckInput(ctx, result.Messages, agent); !verdict.Allowed {
			err := &GuardrailError{
				Reason: guardrailReason(verdict, "Input denied"),
				Phase:  GuardrailPhaseInput,
			}
			options.emit(Event{Type: EventError, Data: map[string]interface{}{"message": err.Error()}})
			return result, err
		}

		raw, err := executeWith(ctx, executor, agent, result.Messages)
		if err != nil {
			options.emit(Event{Type: EventError, Data: map[string]interface{}{"message": err.Error()}})
			return result, err
		}
		result.Iterations = iteration

		processed, err := processWith(ctx, processor, agent, raw)
		if err != nil {
			options.emit(Event{Type: EventError, Data: map[string]interface{}{"message": err.Error()}})
			return result, err
		}

		toolCalls, hasToolCalls := AsToolCalls(processed)
		if !hasToolCalls {
			// The output guardrail runs only on a final answer. An
			// intermediate tool-calling round is not something the host
			// promised to review, and checking it would fire the policy on
			// text the user will never see.
			verdict := options.Guardrails.CheckOutput(ctx, processed, agent)
			if !verdict.Allowed {
				err := &GuardrailError{
					Reason: guardrailReason(verdict, "Output denied"),
					Phase:  GuardrailPhaseOutput,
				}
				options.emit(Event{Type: EventError, Data: map[string]interface{}{"message": err.Error()}})
				return result, err
			}
			if verdict.Rewrite != nil {
				processed = *verdict.Rewrite
			}

			result.Output = processed
			// The model's own answer belongs in the conversation the caller
			// gets back: a host that appends the next user turn and calls
			// RunMessages again must not silently drop what the model said.
			result.Messages = append(result.Messages,
				wire.TextMessage(model.RoleAssistant, assistantTextOf(processed), nil))
			options.emit(Event{Type: EventDone, Data: map[string]interface{}{"response": processed}})
			return result, nil
		}

		// "Starting agent loop" is announced once, immediately before the first
		// tool is dispatched — a turn that never calls a tool emits only `done`.
		if !announced {
			announced = true
			options.emit(Event{Type: EventStatus, Data: map[string]interface{}{"message": "Starting agent loop"}})
		}

		dispatches, err := dispatchAll(ctx, agent, options, bindings, toolCalls)
		result.Dispatches = append(result.Dispatches, dispatches...)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				options.emit(Event{Type: EventCancelled, Data: map[string]interface{}{
					"reason": "Cancellation requested after tool execution",
				}})
			} else {
				options.emit(Event{Type: EventError, Data: map[string]interface{}{"message": err.Error()}})
			}
			return result, err
		}

		// Tool results are handed back in the same order the model requested
		// them, regardless of the order they finished in. Providers reject a
		// tool_calls block whose results do not line up.
		texts := make([]string, 0, len(dispatches))
		for _, dispatch := range dispatches {
			texts = append(texts, dispatch.Text())
		}

		followUp, err := executor.FormatToolMessages(raw, toolCalls, texts, textContentOf(processed))
		if err != nil {
			options.emit(Event{Type: EventError, Data: map[string]interface{}{"message": err.Error()}})
			return result, err
		}
		result.Messages = append(result.Messages, followUp...)
		options.emit(Event{Type: EventMessagesUpdate, Data: map[string]interface{}{
			"message_count": len(result.Messages),
		}})
	}
}

// dispatchAll runs a round of tool calls in request order.
func dispatchAll(
	ctx context.Context,
	agent model.Prompty,
	options RunOptions,
	bindings map[string]map[string]interface{},
	calls []model.ToolCall,
) ([]ToolDispatch, error) {
	registry := options.Tools
	if registry == nil {
		// An empty registry gives the same, correct answer as a nil one — the
		// tool is not registered — without a special case at every call site.
		registry = NewToolRegistry()
	}

	// A tool guardrail and the host's Permit are folded into one decision so
	// the dispatcher has a single refusal path, and a denial from either
	// produces the same model-visible tool result.
	permit := guardrailPermission(options.Guardrails, agent, options.Permit)

	out := make([]ToolDispatch, 0, len(calls))
	for _, call := range calls {
		if err := ctx.Err(); err != nil {
			return out, err
		}

		options.emit(Event{Type: EventToolCallStart, Data: map[string]interface{}{
			"id":        call.Id,
			"name":      call.Name,
			"arguments": call.Arguments,
		}})

		dispatch, err := registry.Dispatch(ctx, call, bindings[call.Name], permit)
		if err != nil {
			return out, err
		}
		out = append(out, dispatch)

		options.emit(Event{Type: EventToolResult, Data: map[string]interface{}{
			"id":     call.Id,
			"name":   call.Name,
			"result": dispatch.Text(),
			"denied": dispatch.Denied,
		}})
	}
	return out, nil
}

func cancellationReason(iteration int) string {
	if iteration == 1 {
		return "Cancellation requested before first iteration"
	}
	return fmt.Sprintf("Cancellation requested before iteration %d", iteration)
}

// AsToolCalls reports whether a processed result is a tool-call request.
//
// Processors return []model.ToolCall for a tool round and something else — a
// string, a decoded structured value — for a final answer, so this is the
// loop's continue/stop test.
func AsToolCalls(processed interface{}) ([]model.ToolCall, bool) {
	switch v := processed.(type) {
	case []model.ToolCall:
		return v, len(v) > 0
	case *[]model.ToolCall:
		if v == nil {
			return nil, false
		}
		return *v, len(*v) > 0
	default:
		return nil, false
	}
}

// textContentOf returns the assistant text that accompanied a tool round, or
// nil. Some providers interleave prose with tool calls and need it preserved in
// the follow-up assistant message.
func textContentOf(processed interface{}) *string {
	if text, ok := processed.(string); ok && text != "" {
		return &text
	}
	return nil
}

// assistantTextOf renders a processed final answer back into message text.
//
// A structured answer is re-encoded rather than dropped, so the appended
// assistant message still carries what the model actually said and the
// conversation stays replayable for a follow-up turn.
func assistantTextOf(processed interface{}) string {
	switch value := processed.(type) {
	case nil:
		return ""
	case string:
		return value
	default:
		encoded, err := json.Marshal(value)
		if err != nil {
			return fmt.Sprint(value)
		}
		return string(encoded)
	}
}

func resolveInvokers(agent model.Prompty, options RunOptions) (Executor, Processor, error) {
	executor := options.Executor
	processor := options.Processor

	var err error
	if executor == nil {
		if executor, err = ExecutorFor(agent); err != nil {
			return nil, nil, err
		}
	}
	if processor == nil {
		if processor, err = ProcessorFor(agent); err != nil {
			return nil, nil, err
		}
	}
	return executor, processor, nil
}
