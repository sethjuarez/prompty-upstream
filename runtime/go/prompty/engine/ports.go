package engine

import (
	"context"
	"fmt"
	"sync"
	"time"

	model "prompty/model"
)

// The engine's collaborators. Each port is one responsibility the engine does
// not own: the engine decides *when* something happens and in what order, a
// port decides *how*.
//
// Every port takes a context.Context. Cancellation is cooperative — the engine
// checks between steps — but a port that can abandon in-flight work should,
// because the expensive step in a turn is the one the engine cannot interrupt.

// ModelPort performs one normalized model invocation.
//
// A returned error is retryable: the engine emits model_invocation_failed,
// consults the retry policy and tries again up to the attempt budget. A port
// that knows a failure is permanent should say so by wrapping ErrPermanent.
type ModelPort interface {
	Invoke(ctx context.Context, request model.ModelInvocationRequest) (model.ModelInvocationResponse, error)
}

// HostPolicyPort is the host's hook before each model call and before the
// final commit.
//
// BeforeModel receives the conversation the engine is about to send and returns
// the one it should actually send, which is where trimming, steering injection
// and input policy live. BeforeCommit receives the final output and may
// replace it or refuse it.
type HostPolicyPort interface {
	BeforeModel(ctx context.Context, request model.HostPolicyRequest) (model.HostPolicyResult, error)
	BeforeCommit(ctx context.Context, request model.FinalOutputPolicyRequest) (model.FinalOutputPolicyResult, error)
}

// PermissionPort authorizes one tool request.
//
// A refusal is not an error: it produces a model-visible failed tool result, so
// the model learns it was refused and can adapt. An error from this port means
// the *decision* could not be made, which fails the turn.
type PermissionPort interface {
	Authorize(ctx context.Context, request model.ModelToolRequest) (model.EnginePermissionDecision, error)
}

// ToolPort executes one authorized tool request.
//
// A tool that fails should return a model.ModelToolResult with a failed
// outcome, not an error. An error is reserved for a dispatcher that could not
// run the tool at all — an unknown tool, a broken transport — which fails the
// turn, because inventing a result for a tool that was never dispatched would
// teach the model that the capability exists.
type ToolPort interface {
	Execute(ctx context.Context, request model.ModelToolRequest) (model.ModelToolResult, error)
}

// ConversationPort converts one completed model-and-tool round into the
// provider-valid messages that carry it into the next call.
//
// This is the only provider-shaped step in the loop: an assistant message with
// tool calls and the matching tool results have to be spelled the way the
// provider expects, or the next call is rejected.
type ConversationPort interface {
	FormatToolExchange(response model.ModelInvocationResponse, results []model.ModelToolResult) ([]model.Message, error)
}

// DurabilityPort persists the engine's decisions.
//
// AppendWithCheckpoint must be atomic: a journal that recorded events whose
// checkpoint was not saved will replay past a state that never existed. A port
// that cannot be atomic should write the checkpoint first, so a crash loses
// events rather than inventing a resumable state.
type DurabilityPort interface {
	Append(ctx context.Context, event model.EngineEvent) error
	AppendWithCheckpoint(ctx context.Context, events []model.EngineEvent, checkpoint model.EngineCheckpoint) error
}

// PostCommitPort runs side effects after the turn is committed.
//
// It runs after the commit, not before, so an effect that fails cannot undo a
// turn that already succeeded. Its failure is reported on the result rather
// than returned, for the same reason.
type PostCommitPort interface {
	AfterCommit(ctx context.Context, effectId string, commit model.TurnCommit) error
}

// RetryPolicyPort decides how long to wait before retrying a failed model call.
// Returning an error abandons the retry and fails the turn, which is how a
// policy refuses to retry at all.
type RetryPolicyPort interface {
	Backoff(ctx context.Context, request model.RetryPolicyRequest) error
}

// Clock supplies the timestamps and identifiers the engine stamps on events.
// Substituting it is what makes a turn's event stream reproducible.
type Clock interface {
	Now() string
	NextID(prefix string) string
}

// Ports bundles the engine's collaborators. Model is required; every other port
// has a safe default described on the field.
type Ports struct {
	// Model performs model invocations. Required.
	Model ModelPort
	// HostPolicy hooks the conversation and the final output. Nil passes both
	// through unchanged.
	HostPolicy HostPolicyPort
	// Permission authorizes tool requests. Nil approves everything, which is
	// the right default for a host that has not installed a policy — the same
	// default the practical loop uses.
	Permission PermissionPort
	// Tool executes tool requests. Nil fails any turn whose model requests a
	// tool, because there is nothing that could run it.
	Tool ToolPort
	// Conversation formats tool exchanges. It is required for any turn whose
	// model requests a tool: without it the results never reach the next model
	// call, so the engine fails the turn rather than sending an incoherent
	// conversation. A turn that only ever answers directly may leave it nil.
	Conversation ConversationPort
	// Durability persists events and checkpoints. Nil discards them, which
	// makes the engine a non-durable turn loop rather than a failure.
	Durability DurabilityPort
	// PostCommit runs after-commit side effects. Nil makes the phase a no-op.
	PostCommit PostCommitPort
	// Retry decides model retry backoff. Nil retries immediately.
	Retry RetryPolicyPort
	// Clock stamps events. Nil uses wall-clock time and a monotonic counter.
	Clock Clock
}

// SystemClock is the default Clock: UTC RFC3339 timestamps and monotonic
// per-prefix identifiers.
type SystemClock struct {
	mu       sync.Mutex
	sequence int
}

// Now returns the current UTC time in RFC3339.
func (c *SystemClock) Now() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// NextID returns the next "<prefix>-<n>" identifier.
func (c *SystemClock) NextID(prefix string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sequence++
	return fmt.Sprintf("%s-%d", prefix, c.sequence)
}

// FixedClock is a deterministic Clock for tests and replay comparison.
type FixedClock struct {
	// Timestamp is stamped on every event.
	Timestamp string

	mu       sync.Mutex
	sequence int
}

// Now returns the fixed timestamp.
func (c *FixedClock) Now() string { return c.Timestamp }

// NextID returns the next "<prefix>-<n>" identifier.
func (c *FixedClock) NextID(prefix string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sequence++
	return fmt.Sprintf("%s-%d", prefix, c.sequence)
}

// ---------------------------------------------------------------------------
// Default port implementations
// ---------------------------------------------------------------------------

// PassthroughHostPolicy accepts every conversation and every output unchanged.
// It is the policy of a host that has none.
type PassthroughHostPolicy struct{}

// BeforeModel returns the request's conversation unchanged.
func (PassthroughHostPolicy) BeforeModel(
	_ context.Context,
	request model.HostPolicyRequest,
) (model.HostPolicyResult, error) {
	return model.HostPolicyResult{
		Messages:             request.Messages,
		StablePrefixMessages: request.StablePrefixMessages,
	}, nil
}

// BeforeCommit returns the request's output unchanged.
func (PassthroughHostPolicy) BeforeCommit(
	_ context.Context,
	request model.FinalOutputPolicyRequest,
) (model.FinalOutputPolicyResult, error) {
	return model.FinalOutputPolicyResult{Output: request.Output}, nil
}

// AllowAllPermissionPort approves every tool request.
type AllowAllPermissionPort struct{}

// Authorize approves the request.
func (AllowAllPermissionPort) Authorize(
	context.Context,
	model.ModelToolRequest,
) (model.EnginePermissionDecision, error) {
	return model.EnginePermissionDecision{Approved: true}, nil
}

// DenyAllPermissionPort refuses every tool request, so a dry run can observe
// what the model would have done without anything happening.
type DenyAllPermissionPort struct {
	// Reason explains the refusal to the model. Empty uses a generic message.
	Reason string
}

// Authorize refuses the request.
func (p DenyAllPermissionPort) Authorize(
	context.Context,
	model.ModelToolRequest,
) (model.EnginePermissionDecision, error) {
	reason := p.Reason
	if reason == "" {
		reason = "Permission denied"
	}
	return model.EnginePermissionDecision{Approved: false, Reason: &reason}, nil
}

// PermissionFunc adapts a plain function to PermissionPort.
type PermissionFunc func(ctx context.Context, request model.ModelToolRequest) (model.EnginePermissionDecision, error)

// Authorize delegates to the function.
func (f PermissionFunc) Authorize(
	ctx context.Context,
	request model.ModelToolRequest,
) (model.EnginePermissionDecision, error) {
	return f(ctx, request)
}

// ToolFunc adapts a plain function to ToolPort.
type ToolFunc func(ctx context.Context, request model.ModelToolRequest) (model.ModelToolResult, error)

// Execute delegates to the function.
func (f ToolFunc) Execute(ctx context.Context, request model.ModelToolRequest) (model.ModelToolResult, error) {
	return f(ctx, request)
}

// ModelFunc adapts a plain function to ModelPort.
type ModelFunc func(ctx context.Context, request model.ModelInvocationRequest) (model.ModelInvocationResponse, error)

// Invoke delegates to the function.
func (f ModelFunc) Invoke(
	ctx context.Context,
	request model.ModelInvocationRequest,
) (model.ModelInvocationResponse, error) {
	return f(ctx, request)
}

// ConversationFunc adapts a plain function to ConversationPort.
type ConversationFunc func(response model.ModelInvocationResponse, results []model.ModelToolResult) ([]model.Message, error)

// FormatToolExchange delegates to the function.
func (f ConversationFunc) FormatToolExchange(
	response model.ModelInvocationResponse,
	results []model.ModelToolResult,
) ([]model.Message, error) {
	return f(response, results)
}

// PostCommitFunc adapts a plain function to PostCommitPort.
type PostCommitFunc func(ctx context.Context, effectId string, commit model.TurnCommit) error

// AfterCommit delegates to the function.
func (f PostCommitFunc) AfterCommit(ctx context.Context, effectId string, commit model.TurnCommit) error {
	return f(ctx, effectId, commit)
}

// RecordingDurabilityPort keeps events and checkpoints in memory. It satisfies
// DurabilityPort and is what a test or an in-process host uses.
//
// AppendWithCheckpoint is atomic under its mutex: an observer never sees the
// events without the checkpoint they were written with.
type RecordingDurabilityPort struct {
	mu          sync.Mutex
	events      []model.EngineEvent
	checkpoints []model.EngineCheckpoint
}

// Append records one event.
func (p *RecordingDurabilityPort) Append(_ context.Context, event model.EngineEvent) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.events = append(p.events, event)
	return nil
}

// AppendWithCheckpoint records events and their checkpoint together.
func (p *RecordingDurabilityPort) AppendWithCheckpoint(
	_ context.Context,
	events []model.EngineEvent,
	checkpoint model.EngineCheckpoint,
) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.events = append(p.events, events...)
	p.checkpoints = append(p.checkpoints, checkpoint)
	return nil
}

// Events returns a copy of the recorded events in order.
func (p *RecordingDurabilityPort) Events() []model.EngineEvent {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]model.EngineEvent(nil), p.events...)
}

// EventKinds returns the recorded event kinds in order, which is the shape the
// shared engine vectors assert.
func (p *RecordingDurabilityPort) EventKinds() []string {
	events := p.Events()
	kinds := make([]string, 0, len(events))
	for _, event := range events {
		kinds = append(kinds, string(event.Kind))
	}
	return kinds
}

// Checkpoints returns a copy of the recorded checkpoints in order.
func (p *RecordingDurabilityPort) Checkpoints() []model.EngineCheckpoint {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]model.EngineCheckpoint(nil), p.checkpoints...)
}
