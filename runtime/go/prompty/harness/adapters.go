package harness

import (
	"fmt"
	"sort"
	"sync"
	"time"

	model "prompty/model"
)

// Value boxes a value into the *interface{} the emitted optional fields use.
// It exists so call sites can write Output: harness.Value("done") instead of
// hoisting a variable for every optional payload.
func Value(v interface{}) *interface{} { return &v }

// String boxes a string into the *string the emitted optional fields use.
func String(v string) *string { return &v }

// Int32 boxes an int32 into the *int32 the emitted optional fields use.
func Int32(v int32) *int32 { return &v }

// Float64 boxes a float64 into the *float64 the emitted optional fields use.
func Float64(v float64) *float64 { return &v }

// ---------------------------------------------------------------------------
// Event sink
// ---------------------------------------------------------------------------

// CollectingEventSink captures emitted turn and session events in memory.
//
// It satisfies model.EventSink and is the sink a test or a UI-less host wants:
// nothing is dropped, ordering is preserved, and the accessors hand back copies
// so a reader can iterate while the turn is still running.
//
// The zero value is ready to use and is safe for concurrent emission.
type CollectingEventSink struct {
	mu            sync.Mutex
	turnEvents    []model.TurnEvent
	sessionEvents []model.SessionEvent
}

// EmitTurn records a turn event.
func (s *CollectingEventSink) EmitTurn(turnEvent model.TurnEvent) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turnEvents = append(s.turnEvents, turnEvent)
	return true, nil
}

// EmitSession records a session event.
func (s *CollectingEventSink) EmitSession(sessionEvent model.SessionEvent) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessionEvents = append(s.sessionEvents, sessionEvent)
	return true, nil
}

// TurnEvents returns a copy of the recorded turn events, in emission order.
func (s *CollectingEventSink) TurnEvents() []model.TurnEvent {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]model.TurnEvent(nil), s.turnEvents...)
}

// SessionEvents returns a copy of the recorded session events, in emission
// order.
func (s *CollectingEventSink) SessionEvents() []model.SessionEvent {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]model.SessionEvent(nil), s.sessionEvents...)
}

// Reset discards everything recorded so far, so one sink can be reused across
// turns without leaking the previous turn's events into the next assertion.
func (s *CollectingEventSink) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turnEvents = nil
	s.sessionEvents = nil
}

// FuncEventSink adapts plain functions to model.EventSink, for a host that
// forwards events to its own bus instead of buffering them.
//
// A nil function accepts and discards its event kind, so a host that only cares
// about turn events supplies only OnTurn.
type FuncEventSink struct {
	OnTurn    func(model.TurnEvent) error
	OnSession func(model.SessionEvent) error
}

// EmitTurn forwards a turn event.
func (s FuncEventSink) EmitTurn(turnEvent model.TurnEvent) (bool, error) {
	if s.OnTurn == nil {
		return true, nil
	}
	if err := s.OnTurn(turnEvent); err != nil {
		return false, err
	}
	return true, nil
}

// EmitSession forwards a session event.
func (s FuncEventSink) EmitSession(sessionEvent model.SessionEvent) (bool, error) {
	if s.OnSession == nil {
		return true, nil
	}
	if err := s.OnSession(sessionEvent); err != nil {
		return false, err
	}
	return true, nil
}

// MultiEventSink fans one event out to several sinks in order.
//
// The first failure stops the fan-out and is returned: an event that some sinks
// saw and others did not is a split-brain trace, and reporting it is better
// than letting the durable sink silently fall behind the console one.
type MultiEventSink []model.EventSink

// EmitTurn forwards to each sink in order.
func (m MultiEventSink) EmitTurn(turnEvent model.TurnEvent) (bool, error) {
	for _, sink := range m {
		if sink == nil {
			continue
		}
		if ok, err := sink.EmitTurn(turnEvent); err != nil || !ok {
			return false, err
		}
	}
	return true, nil
}

// EmitSession forwards to each sink in order.
func (m MultiEventSink) EmitSession(sessionEvent model.SessionEvent) (bool, error) {
	for _, sink := range m {
		if sink == nil {
			continue
		}
		if ok, err := sink.EmitSession(sessionEvent); err != nil || !ok {
			return false, err
		}
	}
	return true, nil
}

// ---------------------------------------------------------------------------
// Checkpoint store
// ---------------------------------------------------------------------------

// InMemoryCheckpointStore stores checkpoints keyed by session and checkpoint
// identifier. It satisfies model.CheckpointStore.
//
// Sessions are isolated: a checkpoint saved under one session is invisible to
// a Load or ListCheckpoints for another, even when the checkpoint identifiers
// collide. Hosts routinely number checkpoints per turn, so collisions across
// sessions are the normal case rather than an edge one.
type InMemoryCheckpointStore struct {
	mu          sync.RWMutex
	checkpoints map[string]model.Checkpoint
}

// NewInMemoryCheckpointStore returns an empty store.
func NewInMemoryCheckpointStore() *InMemoryCheckpointStore {
	return &InMemoryCheckpointStore{checkpoints: map[string]model.Checkpoint{}}
}

// Save persists a checkpoint and returns what was stored.
//
// Both identifiers are required. A checkpoint without them cannot be loaded
// back, so accepting one would produce a store that silently loses writes.
func (s *InMemoryCheckpointStore) Save(checkpoint model.Checkpoint) (model.Checkpoint, error) {
	if checkpoint.SessionId == nil || *checkpoint.SessionId == "" {
		return model.Checkpoint{}, fmt.Errorf("harness: checkpoint sessionId is required")
	}
	if checkpoint.Id == nil || *checkpoint.Id == "" {
		return model.Checkpoint{}, fmt.Errorf("harness: checkpoint id is required")
	}
	stored := cloneCheckpoint(checkpoint)

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.checkpoints == nil {
		s.checkpoints = map[string]model.Checkpoint{}
	}
	s.checkpoints[checkpointKey(*checkpoint.SessionId, *checkpoint.Id)] = stored
	return cloneCheckpoint(stored), nil
}

// Load returns the checkpoint for a session, or nil when there is none. A
// missing checkpoint is not an error: resuming a session that never
// checkpointed is a normal cold start.
func (s *InMemoryCheckpointStore) Load(sessionId string, checkpointId string) (*model.Checkpoint, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	checkpoint, ok := s.checkpoints[checkpointKey(sessionId, checkpointId)]
	if !ok {
		return nil, nil
	}
	copied := cloneCheckpoint(checkpoint)
	return &copied, nil
}

// ListCheckpoints returns a session's checkpoints ordered by checkpoint number,
// falling back to identifier order when numbers are absent or equal.
//
// Ordering is by number rather than by identifier because identifiers are
// strings: "turn-1-checkpoint-10" sorts before "turn-1-checkpoint-2"
// lexicographically, which would present a resumable history out of order.
func (s *InMemoryCheckpointStore) ListCheckpoints(sessionId string) ([]model.Checkpoint, error) {
	s.mu.RLock()
	matched := make([]model.Checkpoint, 0, len(s.checkpoints))
	for _, checkpoint := range s.checkpoints {
		if checkpoint.SessionId != nil && *checkpoint.SessionId == sessionId {
			matched = append(matched, cloneCheckpoint(checkpoint))
		}
	}
	s.mu.RUnlock()

	sort.SliceStable(matched, func(i, j int) bool {
		left, right := matched[i], matched[j]
		leftNumber, rightNumber := checkpointNumber(left), checkpointNumber(right)
		if leftNumber != rightNumber {
			return leftNumber < rightNumber
		}
		return checkpointId(left) < checkpointId(right)
	})
	return matched, nil
}

// Latest returns the highest-numbered checkpoint for a session, which is what
// a resume wants, or nil when the session has none.
func (s *InMemoryCheckpointStore) Latest(sessionId string) (*model.Checkpoint, error) {
	all, err := s.ListCheckpoints(sessionId)
	if err != nil || len(all) == 0 {
		return nil, err
	}
	latest := all[len(all)-1]
	return &latest, nil
}

func checkpointKey(sessionId string, checkpointId string) string {
	// The separator is a NUL byte so it cannot occur inside either identifier
	// and two different (session, checkpoint) pairs cannot collide onto one key.
	return sessionId + "\x00" + checkpointId
}

func checkpointNumber(checkpoint model.Checkpoint) int32 {
	if checkpoint.CheckpointNumber == nil {
		return 0
	}
	return *checkpoint.CheckpointNumber
}

func checkpointId(checkpoint model.Checkpoint) string {
	if checkpoint.Id == nil {
		return ""
	}
	return *checkpoint.Id
}

// cloneCheckpoint copies the mutable parts of a checkpoint so a caller cannot
// reach into the store through a retained map and change what was persisted.
func cloneCheckpoint(checkpoint model.Checkpoint) model.Checkpoint {
	copied := checkpoint
	copied.Id = copyString(checkpoint.Id)
	copied.SessionId = copyString(checkpoint.SessionId)
	copied.TurnId = copyString(checkpoint.TurnId)
	copied.Overview = copyString(checkpoint.Overview)
	copied.Summary = copyString(checkpoint.Summary)
	copied.CreatedAt = copyString(checkpoint.CreatedAt)
	if checkpoint.CheckpointNumber != nil {
		number := *checkpoint.CheckpointNumber
		copied.CheckpointNumber = &number
	}
	copied.State = copyShallowMap(checkpoint.State)
	copied.Metadata = copyShallowMap(checkpoint.Metadata)
	return copied
}

func copyString(value *string) *string {
	if value == nil {
		return nil
	}
	copied := *value
	return &copied
}

// copyShallowMap detaches the top level of a map. Nested values stay shared:
// they are the caller's payload, are treated as immutable everywhere in this
// package, and deep-copying arbitrary interface{} trees on every checkpoint
// would cost more than it protects.
func copyShallowMap(source map[string]interface{}) map[string]interface{} {
	if source == nil {
		return nil
	}
	out := make(map[string]interface{}, len(source))
	for key, value := range source {
		out[key] = value
	}
	return out
}

// ---------------------------------------------------------------------------
// Permission resolvers
// ---------------------------------------------------------------------------

// AllowAllPermissionResolver approves every request. It satisfies
// model.PermissionResolver and is the resolver for a trusted, non-interactive
// host — a batch job with a fixed tool set.
type AllowAllPermissionResolver struct{}

// Request approves the permission.
func (AllowAllPermissionResolver) Request(request model.PermissionRequest) (model.PermissionDecision, error) {
	return decisionFor(request, true, "allow_all"), nil
}

// DenyAllPermissionResolver refuses every request. It satisfies
// model.PermissionResolver and is useful for a dry run: the model still plans
// its tool calls and sees each refusal, but nothing executes.
type DenyAllPermissionResolver struct{}

// Request denies the permission.
func (DenyAllPermissionResolver) Request(request model.PermissionRequest) (model.PermissionDecision, error) {
	return decisionFor(request, false, "deny_all"), nil
}

// PermissionFunc resolves one permission request. It is the shape an
// interactive host implements — prompt the user, consult a policy — without
// having to declare a type.
type PermissionFunc func(request model.PermissionRequest) (model.PermissionDecision, error)

// FuncPermissionResolver adapts a PermissionFunc to model.PermissionResolver.
type FuncPermissionResolver struct{ Resolve PermissionFunc }

// Request delegates to the wrapped function. A nil function approves, matching
// the "no policy installed" default used everywhere else in the runtime.
func (r FuncPermissionResolver) Request(request model.PermissionRequest) (model.PermissionDecision, error) {
	if r.Resolve == nil {
		return decisionFor(request, true, "allow_all"), nil
	}
	return r.Resolve(request)
}

// AllowListPermissionResolver approves only the named targets.
//
// The decision is on PermissionRequest.Target, which the runner sets to the
// tool name, so a host can pin the exact set of tools a session may run without
// writing a resolver.
type AllowListPermissionResolver struct {
	// Allowed is the set of permitted targets.
	Allowed map[string]bool
	// Reason explains a refusal. Empty falls back to a generic message.
	Reason string
}

// Request approves a request whose target is on the list.
func (r AllowListPermissionResolver) Request(request model.PermissionRequest) (model.PermissionDecision, error) {
	target := ""
	if request.Target != nil {
		target = *request.Target
	}
	if r.Allowed[target] {
		return decisionFor(request, true, "allow_list"), nil
	}
	reason := r.Reason
	if reason == "" {
		reason = fmt.Sprintf("Permission denied for %q", target)
	}
	return decisionFor(request, false, reason), nil
}

// decisionFor echoes the request's correlation fields onto a decision. Every
// resolver must do this: a decision that loses the request id cannot be matched
// back to the call it answers, and a journal of them is unreplayable.
func decisionFor(request model.PermissionRequest, approved bool, reason string) model.PermissionDecision {
	return model.PermissionDecision{
		RequestId:  copyString(request.RequestId),
		ToolCallId: copyString(request.ToolCallId),
		Permission: request.Permission,
		Approved:   approved,
		Reason:     String(reason),
	}
}

// ---------------------------------------------------------------------------
// Host tool executor
// ---------------------------------------------------------------------------

// Host tool error kinds. They are stable discriminators, not messages: a host
// switches on them, and a replay compares them across runtimes.
const (
	// ErrorKindNotFound marks a request for a tool nobody registered.
	ErrorKindNotFound = "not_found"
	// ErrorKindException marks a handler that returned an error.
	ErrorKindException = "exception"
	// ErrorKindPermissionDenied marks a call blocked before it ran.
	ErrorKindPermissionDenied = "permission_denied"
	// ErrorKindCancelled marks a call abandoned because the turn was cancelled.
	ErrorKindCancelled = "cancelled"
	// ErrorKindMaxIterations marks a turn stopped by its iteration bound.
	ErrorKindMaxIterations = "max_iterations"
)

// HostToolHandler implements one host tool. The full request is passed
// alongside the decoded arguments so a handler can reach the working directory
// or the correlation identifiers without them being threaded separately.
type HostToolHandler func(arguments map[string]interface{}, request model.HostToolRequest) (interface{}, error)

// FunctionHostToolExecutor dispatches host tool requests to registered Go
// functions. It satisfies model.HostToolExecutor.
//
// A handler's error is never propagated to the caller: it is converted into an
// unsuccessful model.HostToolResult carrying the message. A tool that fails is
// an ordinary event in an agent turn — the model is told and adapts — whereas
// aborting the turn would throw away the work already done.
type FunctionHostToolExecutor struct {
	// Handlers maps tool name to implementation.
	Handlers map[string]HostToolHandler
	// Now measures execution duration. Left nil it is time.Now, which is what
	// production wants; tests set it to make DurationMs deterministic.
	Now func() time.Time
}

// Execute runs the requested tool and reports its completion.
func (e FunctionHostToolExecutor) Execute(request model.HostToolRequest) (model.HostToolResult, error) {
	started := e.now()

	handler, ok := e.Handlers[request.ToolName]
	if !ok {
		return e.failure(request, started, ErrorKindNotFound,
			fmt.Sprintf("No host tool registered for '%s'", request.ToolName)), nil
	}

	arguments := request.Arguments
	if arguments == nil {
		// A handler should never have to distinguish "no arguments" from "nil
		// map"; both mean the model called the tool with nothing.
		arguments = map[string]interface{}{}
	}

	output, err := callHostTool(handler, arguments, request)
	if err != nil {
		return e.failure(request, started, ErrorKindException, err.Error()), nil
	}

	return model.HostToolResult{
		RequestId:  copyString(request.RequestId),
		ToolCallId: copyString(request.ToolCallId),
		ToolName:   request.ToolName,
		Success:    true,
		Result:     Value(output),
		DurationMs: Float64(e.elapsedMs(started)),
	}, nil
}

func callHostTool(
	handler HostToolHandler,
	arguments map[string]interface{},
	request model.HostToolRequest,
) (output interface{}, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("tool panicked: %v", recovered)
		}
	}()
	return handler(arguments, request)
}

func (e FunctionHostToolExecutor) failure(
	request model.HostToolRequest,
	started time.Time,
	errorKind string,
	message string,
) model.HostToolResult {
	return model.HostToolResult{
		RequestId:  copyString(request.RequestId),
		ToolCallId: copyString(request.ToolCallId),
		ToolName:   request.ToolName,
		Success:    false,
		Result:     Value(map[string]interface{}{"message": message}),
		DurationMs: Float64(e.elapsedMs(started)),
		ErrorKind:  String(errorKind),
	}
}

func (e FunctionHostToolExecutor) now() time.Time {
	if e.Now != nil {
		return e.Now()
	}
	return time.Now()
}

func (e FunctionHostToolExecutor) elapsedMs(started time.Time) float64 {
	return float64(e.now().Sub(started).Microseconds()) / 1000
}
