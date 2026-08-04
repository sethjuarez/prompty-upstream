package prompty

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"

	model "prompty/model"
	wire "prompty/wire"
)

// ToolFunc is a host-supplied tool implementation.
//
// args are the decoded call arguments with any declared bindings already
// applied. Returning an error is not fatal to a turn: the dispatcher converts
// it into an error-status ToolResult so the model can see what went wrong and
// react, which is what the shared agent vectors require.
type ToolFunc func(ctx context.Context, args map[string]interface{}) (model.ToolResult, error)

// TextToolFunc is the convenience shape for a tool that returns plain text.
type TextToolFunc func(ctx context.Context, args map[string]interface{}) (string, error)

// ToolRegistry maps tool names to implementations. The zero value is not
// usable; call NewToolRegistry. It is safe for concurrent use.
type ToolRegistry struct {
	mu      sync.RWMutex
	entries map[string]ToolFunc
}

// NewToolRegistry returns an empty registry.
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{entries: map[string]ToolFunc{}}
}

// Register adds or replaces the implementation for name.
func (r *ToolRegistry) Register(name string, fn ToolFunc) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.entries == nil {
		r.entries = map[string]ToolFunc{}
	}
	r.entries[name] = fn
}

// RegisterText adds or replaces a tool that returns plain text.
func (r *ToolRegistry) RegisterText(name string, fn TextToolFunc) {
	r.Register(name, func(ctx context.Context, args map[string]interface{}) (model.ToolResult, error) {
		text, err := fn(ctx, args)
		if err != nil {
			return model.ToolResult{}, err
		}
		return model.NewTextToolResult(text), nil
	})
}

// Unregister removes the implementation for name.
func (r *ToolRegistry) Unregister(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.entries, name)
}

// Has reports whether name is registered.
func (r *ToolRegistry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.entries[name]
	return ok
}

// Names lists the registered tool names in sorted order.
func (r *ToolRegistry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.entries))
	for name := range r.entries {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

func (r *ToolRegistry) lookup(name string) (ToolFunc, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	fn, ok := r.entries[name]
	return fn, ok
}

// ToolDispatch is the outcome of dispatching one tool call.
type ToolDispatch struct {
	// Call is the originating call, echoed so results can be zipped back to
	// calls without relying on positional alignment.
	Call model.ToolCall
	// Result is what the model will be shown. It is always populated, including
	// for denials and execution failures — a tool call the model made must
	// always come back with something, or the provider rejects the next turn.
	Result model.ToolResult
	// Args are the effective arguments after binding injection, nil when the
	// arguments could not be decoded.
	Args map[string]interface{}
	// Denied reports that a permission decision blocked execution.
	Denied bool
}

// Text is the model-visible text of the dispatch result.
func (d ToolDispatch) Text() string {
	result := d.Result
	return result.Text()
}

// PermissionDecision is a host ruling on whether a tool call may run.
type PermissionDecision struct {
	Allowed bool
	// Reason explains a denial. It is surfaced to the model verbatim inside the
	// synthetic tool result, so it must be safe to show.
	Reason string
}

// Allow is the decision that permits a call.
func Allow() PermissionDecision { return PermissionDecision{Allowed: true} }

// Deny is the decision that blocks a call with a model-visible reason.
func Deny(reason string) PermissionDecision {
	return PermissionDecision{Allowed: false, Reason: reason}
}

// PermissionFunc rules on a tool call before it executes.
type PermissionFunc func(ctx context.Context, call model.ToolCall, args map[string]interface{}) PermissionDecision

// ErrToolNotRegistered marks a call to a tool the host never registered.
//
// Unlike an execution failure this is fatal to the turn: the model invented a
// capability that does not exist, and feeding it a synthetic result would teach
// it the call was legitimate. The shared agent vector
// tool_not_registered_error requires the loop to stop.
var ErrToolNotRegistered = errors.New("prompty: tool not registered")

// ToolNotRegisteredError names the missing tool.
type ToolNotRegisteredError struct{ Name string }

func (e *ToolNotRegisteredError) Error() string {
	return "Tool not registered: " + e.Name
}

func (e *ToolNotRegisteredError) Unwrap() []error {
	return []error{ErrToolNotRegistered, ErrValue}
}

// Dispatch runs one tool call.
//
// bindings are the agent's declared parameter injections for this tool, already
// resolved to values; they overwrite whatever the model supplied, because a
// binding exists precisely to keep a value out of the model's control.
//
// A missing tool returns *ToolNotRegisteredError. Every other failure — an
// undecodable argument string, a panicking or erroring implementation — is
// folded into an error-status ToolResult and returned with a nil error, so the
// turn loop can show it to the model and continue.
func (r *ToolRegistry) Dispatch(
	ctx context.Context,
	call model.ToolCall,
	bindings map[string]interface{},
	permit PermissionFunc,
) (ToolDispatch, error) {
	fn, ok := r.lookup(call.Name)
	if !ok {
		return ToolDispatch{Call: call}, &ToolNotRegisteredError{Name: call.Name}
	}

	args, err := DecodeToolArguments(call.Arguments)
	if err != nil {
		return ToolDispatch{
			Call:   call,
			Result: errorToolResult("invalid_arguments", fmt.Sprintf("Error calling '%s': invalid arguments: %v", call.Name, err)),
		}, nil
	}
	for name, value := range bindings {
		args[name] = value
	}

	if permit != nil {
		if decision := permit(ctx, call, args); !decision.Allowed {
			reason := decision.Reason
			if reason == "" {
				reason = "not permitted"
			}
			return ToolDispatch{
				Call:   call,
				Args:   args,
				Denied: true,
				Result: errorToolResult("permission_denied", "Tool denied by guardrail: "+reason),
			}, nil
		}
	}

	if err := ctx.Err(); err != nil {
		return ToolDispatch{Call: call, Args: args}, err
	}

	result, err := invokeTool(ctx, fn, args)
	if err != nil {
		// Cancellation is the caller's business, not the model's: it must
		// propagate rather than become a tool result the model reasons about.
		if ctxErr := ctx.Err(); ctxErr != nil && errors.Is(err, ctxErr) {
			return ToolDispatch{Call: call, Args: args}, err
		}
		return ToolDispatch{
			Call:   call,
			Args:   args,
			Result: errorToolResult("execution_failed", fmt.Sprintf("Error calling '%s': %v", call.Name, err)),
		}, nil
	}

	return ToolDispatch{Call: call, Args: args, Result: result}, nil
}

// invokeTool calls fn behind a recover barrier. A host tool is arbitrary code;
// a panic in it must degrade to a model-visible error, not take down the turn.
func invokeTool(ctx context.Context, fn ToolFunc, args map[string]interface{}) (result model.ToolResult, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("tool panicked: %v", r)
		}
	}()
	return fn(ctx, args)
}

func errorToolResult(kind, message string) model.ToolResult {
	status := model.ToolResultStatusError
	return model.ToolResult{
		Parts:        []interface{}{model.TextPart{Kind: "text", Value: message}},
		Status:       &status,
		ErrorKind:    &kind,
		ErrorMessage: &message,
	}
}

// DecodeToolArguments parses the JSON argument string of a tool call.
//
// An empty or whitespace-only string is an empty argument set, not a failure:
// providers send "" or "{}" for a zero-parameter tool. Numbers are narrowed so
// an integer argument stays an integer instead of becoming a float64, which
// matters for tools that round-trip values back into a prompt.
func DecodeToolArguments(raw string) (map[string]interface{}, error) {
	trimmed := trimSpace(raw)
	if trimmed == "" || trimmed == "null" {
		return map[string]interface{}{}, nil
	}
	decoded, err := decodeJSONValue([]byte(trimmed))
	if err != nil {
		return nil, err
	}
	args, ok := decoded.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("expected a JSON object, got %T", decoded)
	}
	return args, nil
}

// EncodeToolArguments serialises decoded arguments back to the wire string a
// provider expects on a tool call.
func EncodeToolArguments(args map[string]interface{}) (string, error) {
	if len(args) == 0 {
		return "{}", nil
	}
	encoded, err := json.Marshal(args)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

// ResolveBindings evaluates a function tool's declared bindings against the
// parent agent's inputs (spec §2.9.1).
//
// A binding names an input of the enclosing agent whose value is injected into
// every call of that tool. Bindings whose input is absent are skipped rather
// than injected as nil, so an unset optional input leaves the model's own value
// in place instead of overwriting it with null.
func ResolveBindings(tool wire.FunctionToolSpec, inputs map[string]interface{}) map[string]interface{} {
	if len(tool.Bindings) == 0 {
		return nil
	}
	out := make(map[string]interface{}, len(tool.Bindings))
	for _, binding := range tool.Bindings {
		if binding.Name == "" || binding.Input == "" {
			continue
		}
		if value, ok := inputs[binding.Input]; ok {
			out[binding.Name] = value
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// AgentBindings resolves the bindings of every function tool the agent declares,
// keyed by tool name.
func AgentBindings(agent model.Prompty, inputs map[string]interface{}) map[string]map[string]interface{} {
	tools := wire.FunctionTools(agent)
	if len(tools) == 0 {
		return nil
	}
	out := make(map[string]map[string]interface{}, len(tools))
	for _, tool := range tools {
		if resolved := ResolveBindings(tool, inputs); resolved != nil {
			out[tool.Name] = resolved
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func trimSpace(s string) string {
	start, end := 0, len(s)
	for start < end && isSpaceByte(s[start]) {
		start++
	}
	for end > start && isSpaceByte(s[end-1]) {
		end--
	}
	return s[start:end]
}

func isSpaceByte(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}
