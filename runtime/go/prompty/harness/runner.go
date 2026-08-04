package harness

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	model "prompty/model"
)

// TurnRunner drives one durable agent turn over the emitted harness protocol:
// it calls the model, resolves permissions, executes host tools, emits typed
// turn and session events to a sink and a journal, and checkpoints after every
// model response.
//
// # Why this is not the emitted ReferenceTurnRunner
//
// prompty/model carries a generated ReferenceTurnRunner with the same shape.
// It is regenerated wholesale and is not edited, and it has a defect this
// runner exists to fix: when a permission request is denied it returns the
// synthetic model.HostToolResult to the model but never records a tool_result
// turn event. The refusal is therefore invisible in the journal, so a replay
// cannot tell "the tool was denied" from "the tool was never requested" — and
// a denial is precisely the step an audit of a durable session most needs to
// see. The shared vector spec/vectors/harness/replay_vectors.json requires
// `turn:tool_result:0:add:false:permission_denied`, which the emitted runner
// does not produce.
//
// Behaviour is otherwise deliberately identical, so a host can swap between
// them and compare journals.
//
// # Event order
//
// Per turn: session_start, turn_start. Per model round: llm_start,
// llm_complete, then a checkpoint and its checkpoint_created session event. Per
// requested tool: permission_requested, permission_completed, then — only when
// approved — tool_execution_start and tool_execution_complete, then always
// tool_result. After the round: messages_updated. At the end: error (only when
// the turn failed), turn_end, session_end, and the journal summary.
//
// # Determinism
//
// Now and NextId default to wall-clock time and a monotonic counter. Supplied,
// they make a turn's journal byte-identical across runs.
type TurnRunner struct {
	// EventSink receives every event as it happens. Required.
	EventSink model.EventSink
	// Journal persists every event for replay. Required.
	Journal model.EventJournalWriter
	// CheckpointStore persists resumable state after each model response.
	// Required.
	CheckpointStore model.CheckpointStore
	// PermissionResolver rules on each requested tool. Required whenever the
	// model can request tools.
	PermissionResolver model.PermissionResolver
	// HostToolExecutor runs approved tools. Required whenever the model can
	// request tools.
	HostToolExecutor model.HostToolExecutor
	// InvokeModel produces one model response per iteration. Required.
	InvokeModel TurnModelFunc
	// Now stamps events. Nil means time.Now in UTC, RFC3339.
	Now func() string
	// NextId names events and generated permission requests. Nil means a
	// per-runner monotonic counter, "<prefix>-<n>".
	NextId func(prefix string) string
	// DefaultMaxIterations bounds a turn whose options do not. Zero means 10.
	DefaultMaxIterations int32

	mu       sync.Mutex
	sequence int
}

// TurnModelFunc produces one model response for an iteration. The request
// carries the tool results from the previous round, so a provider that needs
// them in its next call has them without the runner knowing the wire shape.
type TurnModelFunc func(request model.TurnModelRequest) (model.TurnModelResponse, error)

// ContextTurnModelFunc is the cancellation-aware form of TurnModelFunc. A
// runner whose InvokeModel is not context-aware still cancels — the check
// happens between iterations — but a long provider call cannot be interrupted.
type ContextTurnModelFunc func(ctx context.Context, request model.TurnModelRequest) (model.TurnModelResponse, error)

// WithContext adapts a ContextTurnModelFunc so a cancellable model call can be
// installed in TurnRunner.InvokeModel.
func WithContext(ctx context.Context, invoke ContextTurnModelFunc) TurnModelFunc {
	return func(request model.TurnModelRequest) (model.TurnModelResponse, error) {
		return invoke(ctx, request)
	}
}

// defaultMaxIterations bounds a turn whose options say nothing. Without a bound
// a model that keeps requesting tools would spin forever.
const defaultMaxIterations = int32(10)

// maxIterationsMessage is the model-visible text for a turn stopped by its
// iteration bound. It is a constant because a replay compares it.
const maxIterationsMessage = "Maximum turn iterations reached"

// ErrRunnerMisconfigured reports a runner missing a collaborator it needs.
var ErrRunnerMisconfigured = errors.New("harness: turn runner is misconfigured")

// Run executes a turn without cancellation.
func (r *TurnRunner) Run(request model.RunTurnRequest) (model.RunTurnResult, error) {
	return r.RunContext(context.Background(), request)
}

// RunContext executes a turn, honouring cancellation between steps.
//
// Cancellation is checked at the top of each iteration and before each tool,
// which are the points where the turn is about to spend money or cause an
// effect. A cancelled turn still emits turn_end, session_end and a journal
// summary with status cancelled: an abandoned session that leaves no terminal
// record is indistinguishable from one that crashed.
func (r *TurnRunner) RunContext(ctx context.Context, request model.RunTurnRequest) (model.RunTurnResult, error) {
	if err := r.validate(); err != nil {
		return model.RunTurnResult{}, err
	}
	if ctx == nil {
		ctx = context.Background()
	}

	// finalize closes the journal on every path it reaches, but the paths that
	// abort before it — a sink failure while recording the opening events, a
	// collaborator that broke mid-turn — would otherwise leak the journal's
	// file handle, which the caller has no reference to and cannot release.
	// Close is idempotent, so the finalized case is unaffected.
	defer func() { _, _ = r.Journal.Close(nil) }()

	inputs := request.Inputs
	if inputs == nil {
		inputs = map[string]interface{}{}
	}
	options := request.Options
	if options == nil {
		options = &model.TurnOptions{}
	}
	maxIterations := r.maxIterations(options)

	state := &turnState{
		request:       request,
		inputs:        inputs,
		options:       options,
		maxIterations: maxIterations,
	}

	if err := r.recordSession(model.SessionEventTypeSessionStart, request.SessionId, request.TurnId, map[string]interface{}{
		"sessionId":     request.SessionId,
		"schemaVersion": "1",
	}); err != nil {
		return model.RunTurnResult{}, err
	}
	if err := r.recordTurn(model.TurnEventTypeTurnStart, request.TurnId, 0, map[string]interface{}{
		"inputs":        inputs,
		"maxIterations": maxIterations,
	}); err != nil {
		return model.RunTurnResult{}, err
	}

	runErr := r.loop(ctx, state)
	cancelled := errors.Is(runErr, context.Canceled) || errors.Is(runErr, context.DeadlineExceeded)
	// A collaborator failure aborts before the terminal records can be trusted;
	// a cancellation does not, and is finalised below like any other outcome.
	if runErr != nil && !cancelled {
		return model.RunTurnResult{}, runErr
	}
	if cancelled && !state.cancelled {
		// The cancellation surfaced through a collaborator — a permission
		// resolver or tool executor that honoured the context — and the loop's
		// own checks never saw it. Recording it here keeps finalize from
		// misreading an abandoned turn as one that ran out of iterations.
		state.cancelled = true
		state.cancelReason = runErr.Error()
	}

	result, err := r.finalize(state)
	if err != nil {
		return model.RunTurnResult{}, err
	}
	return result, runErr
}

// turnState is the mutable state of one turn, kept in one place so the loop and
// the finaliser cannot disagree about what happened.
type turnState struct {
	request       model.RunTurnRequest
	inputs        map[string]interface{}
	options       *model.TurnOptions
	maxIterations int32

	iterations   int32
	checkpoints  []model.Checkpoint
	allResults   []model.HostToolResult
	pending      []model.HostToolResult
	output       interface{}
	hasOutput    bool
	answered     bool
	cancelled    bool
	cancelReason string
}

func (r *TurnRunner) loop(ctx context.Context, state *turnState) error {
	for iteration := int32(0); iteration < state.maxIterations; iteration++ {
		if err := ctx.Err(); err != nil {
			state.cancelled = true
			state.cancelReason = err.Error()
			return err
		}
		state.iterations = iteration + 1

		if err := r.recordTurn(model.TurnEventTypeLlmStart, state.request.TurnId, int(iteration), map[string]interface{}{
			"attempt": 0,
		}); err != nil {
			return err
		}

		response, err := r.InvokeModel(model.TurnModelRequest{
			SessionId:   state.request.SessionId,
			TurnId:      state.request.TurnId,
			Iteration:   iteration,
			Inputs:      state.inputs,
			Options:     state.options,
			ToolResults: state.pending,
		})
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				state.cancelled = true
				state.cancelReason = err.Error()
			}
			return err
		}

		if err := r.recordTurn(model.TurnEventTypeLlmComplete, state.request.TurnId, int(iteration), map[string]interface{}{}); err != nil {
			return err
		}

		checkpoint, err := r.saveCheckpoint(state.request, int(iteration), response)
		if err != nil {
			return err
		}
		state.checkpoints = append(state.checkpoints, checkpoint)

		if len(response.ToolRequests) == 0 {
			state.answered = true
			if response.Output != nil {
				state.output = *response.Output
				state.hasOutput = true
			}
			return nil
		}

		// The pending results are rebuilt each round: they are what the next
		// model call is owed, not a running log. The running log is allResults.
		state.pending = make([]model.HostToolResult, 0, len(response.ToolRequests))
		for _, toolRequest := range response.ToolRequests {
			if err := ctx.Err(); err != nil {
				state.cancelled = true
				state.cancelReason = err.Error()
				return err
			}
			result, err := r.resolveAndExecute(state.request.TurnId, int(iteration), toolRequest)
			if err != nil {
				return err
			}
			state.pending = append(state.pending, result)
			state.allResults = append(state.allResults, result)
		}

		serialized := make([]map[string]interface{}, 0, len(state.pending))
		for _, result := range state.pending {
			serialized = append(serialized, result.Save(model.NewSaveContext()))
		}
		if err := r.recordTurn(model.TurnEventTypeMessagesUpdated, state.request.TurnId, int(iteration), map[string]interface{}{
			"toolResults": serialized,
		}); err != nil {
			return err
		}
	}
	return nil
}

// finalize emits the terminal records and builds the result.
//
// It runs for every outcome, including cancellation, so a session always ends
// with a turn_end, a session_end and a summary. The journal is closed here and
// only here.
func (r *TurnRunner) finalize(state *turnState) (model.RunTurnResult, error) {
	status := model.RunTurnStatusSuccess
	output := state.output
	hasOutput := state.hasOutput

	switch {
	case state.cancelled:
		status = model.RunTurnStatusCancelled
		if err := r.recordTurn(model.TurnEventTypeCancelled, state.request.TurnId, int(state.iterations), map[string]interface{}{
			"reason": state.cancelReason,
		}); err != nil {
			return model.RunTurnResult{}, err
		}

	case !state.answered:
		// The loop ran out of iterations before the model produced an answer.
		// This is an error, not an empty success: the caller asked a question
		// and is getting no answer, and reporting success would hide that.
		status = model.RunTurnStatusError
		output = map[string]interface{}{"message": maxIterationsMessage}
		hasOutput = true
		if err := r.recordTurn(model.TurnEventTypeError, state.request.TurnId, int(state.iterations), map[string]interface{}{
			"errorKind": ErrorKindMaxIterations,
			"message":   maxIterationsMessage,
		}); err != nil {
			return model.RunTurnResult{}, err
		}
	}

	if err := r.recordTurn(model.TurnEventTypeTurnEnd, state.request.TurnId, int(state.iterations), map[string]interface{}{
		"iterations": state.iterations,
		"status":     string(status),
		"response":   output,
	}); err != nil {
		return model.RunTurnResult{}, err
	}
	if err := r.recordSession(model.SessionEventTypeSessionEnd, state.request.SessionId, state.request.TurnId, map[string]interface{}{
		"sessionId": state.request.SessionId,
		"status":    string(status),
		"reason":    "turn_complete",
	}); err != nil {
		return model.RunTurnResult{}, err
	}

	summaryStatus := model.SessionSummaryStatus(status)
	turns := int32(1)
	checkpointCount := int32(len(state.checkpoints))
	if _, err := r.Journal.Close(&model.SessionSummary{
		SessionId:   state.request.SessionId,
		Status:      &summaryStatus,
		Turns:       &turns,
		Checkpoints: &checkpointCount,
	}); err != nil {
		return model.RunTurnResult{}, fmt.Errorf("harness: close journal: %w", err)
	}

	result := model.RunTurnResult{
		SessionId:   state.request.SessionId,
		TurnId:      state.request.TurnId,
		Status:      status,
		Iterations:  state.iterations,
		ToolResults: state.allResults,
		Checkpoints: state.checkpoints,
	}
	if hasOutput {
		result.Output = Value(output)
	}
	return result, nil
}

// resolveAndExecute takes one tool request through permission and execution.
//
// Every path returns a model.HostToolResult and records a tool_result event —
// including a denial. A tool call the model made must always come back with
// something: a provider rejects the following turn for an unanswered tool call,
// and a journal that omits the refusal cannot be audited.
func (r *TurnRunner) resolveAndExecute(
	turnId string,
	iteration int,
	toolRequest model.HostToolRequest,
) (model.HostToolResult, error) {
	permission := r.permissionRequestFor(toolRequest)
	if err := r.recordTurn(model.TurnEventTypePermissionRequested, turnId, iteration,
		permission.Save(model.NewSaveContext())); err != nil {
		return model.HostToolResult{}, err
	}

	decision, err := r.PermissionResolver.Request(permission)
	if err != nil {
		return model.HostToolResult{}, fmt.Errorf("harness: resolve permission: %w", err)
	}
	if err := r.recordTurn(model.TurnEventTypePermissionCompleted, turnId, iteration,
		decision.Save(model.NewSaveContext())); err != nil {
		return model.HostToolResult{}, err
	}

	if !decision.Approved {
		message := "Permission denied"
		if decision.Reason != nil && *decision.Reason != "" {
			message = *decision.Reason
		}
		denied := model.HostToolResult{
			RequestId:  copyString(toolRequest.RequestId),
			ToolCallId: copyString(toolRequest.ToolCallId),
			ToolName:   toolRequest.ToolName,
			Success:    false,
			Result:     Value(map[string]interface{}{"message": message}),
			ErrorKind:  String(ErrorKindPermissionDenied),
		}
		// No tool_execution_start/complete pair: nothing executed, and
		// synthesising one would put an event in the journal describing work
		// that never happened.
		if err := r.recordTurn(model.TurnEventTypeToolResult, turnId, iteration,
			denied.Save(model.NewSaveContext())); err != nil {
			return model.HostToolResult{}, err
		}
		return denied, nil
	}

	if err := r.recordTurn(model.TurnEventTypeToolExecutionStart, turnId, iteration,
		toolRequest.Save(model.NewSaveContext())); err != nil {
		return model.HostToolResult{}, err
	}

	result, err := r.HostToolExecutor.Execute(toolRequest)
	if err != nil {
		return model.HostToolResult{}, fmt.Errorf("harness: execute host tool %q: %w", toolRequest.ToolName, err)
	}

	saved := result.Save(model.NewSaveContext())
	if err := r.recordTurn(model.TurnEventTypeToolExecutionComplete, turnId, iteration, saved); err != nil {
		return model.HostToolResult{}, err
	}
	if err := r.recordTurn(model.TurnEventTypeToolResult, turnId, iteration, saved); err != nil {
		return model.HostToolResult{}, err
	}
	return result, nil
}

// permissionRequestFor derives a permission request from a tool request.
//
// The request id is derived from the tool request's own id where there is one,
// so a host correlating a prompt back to the call it is about does not need a
// side table. Only a tool request without an id falls back to a generated one.
func (r *TurnRunner) permissionRequestFor(toolRequest model.HostToolRequest) model.PermissionRequest {
	requestId := r.id("permission")
	if toolRequest.RequestId != nil && *toolRequest.RequestId != "" {
		requestId = *toolRequest.RequestId + "-permission"
	}
	toolName := toolRequest.ToolName
	return model.PermissionRequest{
		RequestId:  &requestId,
		ToolCallId: copyString(toolRequest.ToolCallId),
		Permission: "tool.execute",
		Target:     &toolName,
		Details:    toolRequest.Save(model.NewSaveContext()),
	}
}

// saveCheckpoint persists the resumable state after one model response and
// announces it.
//
// The response's own checkpoint state is merged over the runner's, so a
// provider that carries opaque continuation state can round-trip it, while the
// runner's iteration bookkeeping stays authoritative for keys it owns.
func (r *TurnRunner) saveCheckpoint(
	request model.RunTurnRequest,
	iteration int,
	response model.TurnModelResponse,
) (model.Checkpoint, error) {
	checkpointId := fmt.Sprintf("%s-checkpoint-%d", request.TurnId, iteration)
	checkpointNumber := int32(iteration + 1)

	toolRequests := make([]map[string]interface{}, 0, len(response.ToolRequests))
	for _, toolRequest := range response.ToolRequests {
		toolRequests = append(toolRequests, toolRequest.Save(model.NewSaveContext()))
	}

	state := map[string]interface{}{
		"iteration":    iteration,
		"output":       dereference(response.Output),
		"toolRequests": toolRequests,
	}
	for key, value := range response.CheckpointState {
		state[key] = value
	}

	sessionId, turnId := request.SessionId, request.TurnId
	saved, err := r.CheckpointStore.Save(model.Checkpoint{
		Id:               &checkpointId,
		SessionId:        &sessionId,
		TurnId:           &turnId,
		CheckpointNumber: &checkpointNumber,
		Title:            fmt.Sprintf("Turn %s iteration %d", turnId, iteration),
		State:            state,
		CreatedAt:        String(r.timestamp()),
	})
	if err != nil {
		return model.Checkpoint{}, fmt.Errorf("harness: save checkpoint: %w", err)
	}

	if err := r.recordSession(model.SessionEventTypeCheckpointCreated, sessionId, turnId, map[string]interface{}{
		"checkpointId":     stringOr(saved.Id, checkpointId),
		"checkpointNumber": checkpointNumber,
	}); err != nil {
		return model.Checkpoint{}, err
	}
	return saved, nil
}

// recordTurn emits a turn event to the sink and the journal, in that order.
//
// The sink comes first because it is the live observer — a UI that has already
// rendered an event is not harmed by a journal write failing afterwards,
// whereas a journal that records an event no observer saw is a silent divergence.
// Either failure aborts the turn: a partially recorded turn is not replayable,
// and continuing would produce a journal that describes a run that did not happen.
func (r *TurnRunner) recordTurn(
	eventType model.TurnEventType,
	turnId string,
	iteration int,
	payload map[string]interface{},
) error {
	turn := turnId
	event := model.TurnEvent{
		Id:        r.id("turn-event"),
		Type:      eventType,
		Timestamp: r.timestamp(),
		TurnId:    &turn,
		Iteration: Int32(int32(iteration)),
		Payload:   payload,
	}
	if _, err := r.EventSink.EmitTurn(event); err != nil {
		return fmt.Errorf("harness: emit turn event %s: %w", eventType, err)
	}
	if _, err := r.Journal.AppendTurn(event); err != nil {
		return fmt.Errorf("harness: journal turn event %s: %w", eventType, err)
	}
	return nil
}

// recordSession emits a session event to the sink and the journal.
func (r *TurnRunner) recordSession(
	eventType model.SessionEventType,
	sessionId string,
	turnId string,
	payload map[string]interface{},
) error {
	session, turn := sessionId, turnId
	event := model.SessionEvent{
		Id:        r.id("session-event"),
		Type:      eventType,
		Timestamp: r.timestamp(),
		SessionId: &session,
		TurnId:    &turn,
		Payload:   payload,
	}
	if _, err := r.EventSink.EmitSession(event); err != nil {
		return fmt.Errorf("harness: emit session event %s: %w", eventType, err)
	}
	if _, err := r.Journal.AppendSession(event); err != nil {
		return fmt.Errorf("harness: journal session event %s: %w", eventType, err)
	}
	return nil
}

func (r *TurnRunner) validate() error {
	var missing []string
	if r.EventSink == nil {
		missing = append(missing, "EventSink")
	}
	if r.Journal == nil {
		missing = append(missing, "Journal")
	}
	if r.CheckpointStore == nil {
		missing = append(missing, "CheckpointStore")
	}
	if r.InvokeModel == nil {
		missing = append(missing, "InvokeModel")
	}
	if len(missing) == 0 {
		return nil
	}
	return fmt.Errorf("%w: missing %v", ErrRunnerMisconfigured, missing)
}

func (r *TurnRunner) maxIterations(options *model.TurnOptions) int32 {
	if options != nil && options.MaxIterations != nil && *options.MaxIterations >= 0 {
		return *options.MaxIterations
	}
	if r.DefaultMaxIterations > 0 {
		return r.DefaultMaxIterations
	}
	return defaultMaxIterations
}

func (r *TurnRunner) timestamp() string {
	if r.Now != nil {
		return r.Now()
	}
	return time.Now().UTC().Format(time.RFC3339)
}

// id generates the next identifier. The counter is mutex-guarded because a
// host may drive a runner's callbacks from another goroutine, and two events
// sharing an id would break journal correlation.
func (r *TurnRunner) id(prefix string) string {
	if r.NextId != nil {
		return r.NextId(prefix)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.sequence++
	return fmt.Sprintf("%s-%d", prefix, r.sequence)
}

func dereference(value *interface{}) interface{} {
	if value == nil {
		return nil
	}
	return *value
}

// SequentialIDs returns a NextId function producing "<prefix>-<n>" from a
// shared counter starting at 1. It is safe for concurrent use and is what a
// deterministic test installs.
func SequentialIDs() func(prefix string) string {
	var (
		mu      sync.Mutex
		counter int
	)
	return func(prefix string) string {
		mu.Lock()
		defer mu.Unlock()
		counter++
		return fmt.Sprintf("%s-%d", prefix, counter)
	}
}

// FixedClock returns a Now function that always reports the same timestamp,
// so a journal can be compared byte for byte.
func FixedClock(timestamp string) func() string {
	return func() string { return timestamp }
}
