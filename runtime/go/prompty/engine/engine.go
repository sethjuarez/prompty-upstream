package engine

import (
	"context"
	"errors"
	"fmt"

	model "prompty/model"
)

// Engine error kinds. They are stable discriminators a host switches on, and a
// replay compares, so they are constants rather than formatted messages.
const (
	ErrorKindMaxIterations     = "max_iterations"
	ErrorKindModelFailed       = "model_invocation_failed"
	ErrorKindPermissionDenied  = "permission_denied"
	ErrorKindToolDispatch      = "tool_dispatch_failed"
	ErrorKindPolicyFailed      = "host_policy_failed"
	ErrorKindConversation      = "conversation_format_failed"
	ErrorKindCancelled         = "cancelled"
	ErrorKindDurabilityFailure = "durability_failed"
)

// DefaultMaxIterations bounds a turn whose request does not.
const DefaultMaxIterations = 10

// DefaultMaxModelAttempts bounds the retries of one model invocation.
const DefaultMaxModelAttempts = 1

// ErrPermanent marks a model failure the engine must not retry. A ModelPort
// that knows a failure is not transient — a malformed request, a rejected
// credential — wraps it, so the engine does not burn its attempt budget on a
// call that cannot succeed.
var ErrPermanent = errors.New("engine: permanent failure")

// ErrTurnFailed marks a turn the engine ran to a failed commit, so a caller can
// test for the class without matching on a message.
var ErrTurnFailed = errors.New("engine: turn failed")

// TurnFailedError reports a turn that completed with a failed commit.
//
// It is returned alongside the Result rather than swallowed: a turn that ran
// out of iterations, or whose pre-commit policy refused the output, produced no
// answer, and returning a nil error next to a failed commit would let a caller
// that checks only the error treat it as a success.
type TurnFailedError struct {
	ErrorKind string
	Message   string
}

func (e *TurnFailedError) Error() string {
	return fmt.Sprintf("engine: turn failed (%s): %s", e.ErrorKind, e.Message)
}

func (e *TurnFailedError) Unwrap() error { return ErrTurnFailed }

// ErrNoModelPort reports a request with no ModelPort, which cannot run.
var ErrNoModelPort = errors.New("engine: a ModelPort is required")

// Request describes one turn to run.
type Request struct {
	// SessionId and TurnId identify the turn. Both appear on every event and
	// checkpoint, so a durable store can find a turn's records.
	SessionId string
	TurnId    string
	// RunId distinguishes this attempt at the turn. Empty is generated.
	RunId string
	// ParentRunId and DelegationDepth position a delegated sub-turn inside its
	// parent, so a nested agent's events are attributable.
	ParentRunId     *string
	DelegationDepth int32
	// Messages is the prepared conversation the turn starts from.
	Messages []model.Message
	// Inputs are the agent inputs, passed through to the host policy.
	Inputs map[string]interface{}
	// MaxIterations bounds the model rounds. Zero uses DefaultMaxIterations.
	MaxIterations int32
	// MaxModelAttempts bounds retries per model invocation. Zero uses
	// DefaultMaxModelAttempts, i.e. no retry.
	MaxModelAttempts int32
	// ContextState is the turn's starting portability. The zero value is
	// treated as portable: a conversation the engine can fully describe.
	ContextState model.InvocationContextState
	// Ports are the engine's collaborators.
	Ports Ports
}

// Result is a completed turn: the emitted TurnEngineResult plus the events the
// engine emitted, so a caller that supplied no DurabilityPort can still see
// what happened.
type Result struct {
	model.TurnEngineResult
	// Events is every engine event in order.
	Events []model.EngineEvent
	// Checkpoints is every checkpoint the turn produced, in order.
	Checkpoints []model.EngineCheckpoint
}

// Run executes one turn.
//
// It returns a Result for every outcome the engine can describe — success,
// failure, cancellation — and an error only when the turn could not be run at
// all or a collaborator the engine depends on broke. A cancelled turn is
// reported as a cancelled commit *and* a context error, so a caller can both
// see the durable record and propagate the cancellation.
func Run(ctx context.Context, request Request) (Result, error) {
	if request.Ports.Model == nil {
		return Result{}, ErrNoModelPort
	}
	if ctx == nil {
		ctx = context.Background()
	}

	run := newRun(ctx, request)
	return run.execute()
}

// turnRun is the mutable state of one turn. Keeping it in one struct rather
// than threading a dozen parameters is what lets the commit path and the loop
// agree on what happened.
type turnRun struct {
	ctx     context.Context
	request Request
	ports   Ports
	clock   Clock

	runId            string
	maxIterations    int32
	maxModelAttempts int32

	sequence     int64
	events       []model.EngineEvent
	checkpoints  []model.EngineCheckpoint
	snapshots    []model.ModelInvocationContextSnapshot
	toolResults  []model.ModelToolResult
	messages     []model.Message
	stablePrefix int32
	contextState model.InvocationContextState

	iteration      int32
	completed      int32
	output         *interface{}
	reconciliation *model.ModelReconciliationState
}

func newRun(ctx context.Context, request Request) *turnRun {
	ports := request.Ports
	if ports.HostPolicy == nil {
		ports.HostPolicy = PassthroughHostPolicy{}
	}
	if ports.Permission == nil {
		ports.Permission = AllowAllPermissionPort{}
	}
	if ports.Clock == nil {
		ports.Clock = &SystemClock{}
	}

	contextState := request.ContextState
	if contextState.Portability == "" {
		contextState.Portability = model.InvocationContextPortabilityPortable
	}

	maxIterations := request.MaxIterations
	if maxIterations <= 0 {
		maxIterations = DefaultMaxIterations
	}
	maxModelAttempts := request.MaxModelAttempts
	if maxModelAttempts <= 0 {
		maxModelAttempts = DefaultMaxModelAttempts
	}

	runId := request.RunId
	if runId == "" {
		runId = ports.Clock.NextID("run")
	}

	return &turnRun{
		ctx:              ctx,
		request:          request,
		ports:            ports,
		clock:            ports.Clock,
		runId:            runId,
		maxIterations:    maxIterations,
		maxModelAttempts: maxModelAttempts,
		messages:         append([]model.Message(nil), request.Messages...),
		stablePrefix:     int32(len(request.Messages)),
		contextState:     contextState,
	}
}

func (r *turnRun) execute() (Result, error) {
	if err := r.emit(model.EngineEventKindTurnStarted, nil, nil); err != nil {
		return r.result(), err
	}

	// Cancellation is checked before any context is prepared, so a turn
	// cancelled before it started costs nothing and produces exactly two
	// events: it started, and it was abandoned.
	if err := r.ctx.Err(); err != nil {
		return r.cancel(err)
	}

	for r.iteration = 0; r.iteration < r.maxIterations; r.iteration++ {
		done, err := r.runIteration()
		if err != nil {
			return r.fail(err)
		}
		if done {
			return r.commit(model.EngineTurnStatusSuccess, nil)
		}
	}

	// The loop ran out of rounds before the model answered. That is a failure,
	// not an empty success: the caller asked a question and is getting none.
	return r.commitFailed(ErrorKindMaxIterations, "Maximum model iterations reached")
}

// runIteration performs one model round and its tool round. It reports done
// when the model produced a final answer.
func (r *turnRun) runIteration() (bool, error) {
	if err := r.ctx.Err(); err != nil {
		return false, err
	}

	policy, err := r.ports.HostPolicy.BeforeModel(r.ctx, model.HostPolicyRequest{
		SessionId:            r.request.SessionId,
		TurnId:               r.request.TurnId,
		Iteration:            r.iteration,
		Messages:             r.messages,
		StablePrefixMessages: r.stablePrefix,
		Inputs:               boxInputs(r.request.Inputs),
	})
	if err != nil {
		return false, fmt.Errorf("%s: %w", ErrorKindPolicyFailed, err)
	}
	r.messages = policy.Messages
	// A policy may only ever shrink the stable prefix: it can drop or rewrite
	// history, which invalidates a prefix, but it cannot make older messages
	// more stable than they already were.
	r.stablePrefix = clampPrefix(min32(r.stablePrefix, policy.StablePrefixMessages), int32(len(r.messages)))

	invocationId := r.clock.NextID("invocation")
	snapshot := model.ModelInvocationContextSnapshot{
		Id:                   r.clock.NextID("snapshot"),
		SessionId:            r.request.SessionId,
		TurnId:               r.request.TurnId,
		InvocationId:         invocationId,
		Iteration:            r.iteration,
		Messages:             append([]model.Message(nil), r.messages...),
		StablePrefixMessages: r.stablePrefix,
		ContextState:         r.contextState,
		Metadata:             policy.Metadata,
	}
	r.snapshots = append(r.snapshots, snapshot)

	if err := r.emitInvocation(model.EngineEventKindContextPrepared, invocationId, map[string]interface{}{
		"snapshotId":           snapshot.Id,
		"messages":             len(r.messages),
		"stablePrefixMessages": r.stablePrefix,
	}); err != nil {
		return false, err
	}

	response, err := r.invokeModel(invocationId, snapshot)
	if err != nil {
		return false, err
	}

	// The provider may hand back continuation state instead of a transcript.
	// Adopting it here, before the checkpoint, is what makes a resume able to
	// reattach rather than replay.
	if response.NextContextState != nil {
		r.contextState = *response.NextContextState
	}
	r.completed = r.iteration + 1

	if len(response.AssistantMessages) > 0 {
		r.messages = append(r.messages, response.AssistantMessages...)
	}

	if err := r.checkpoint(&response, nil); err != nil {
		return false, err
	}

	if len(response.ToolRequests) == 0 {
		r.output = response.Output
		return true, nil
	}

	results, err := r.runToolRound(response)
	if err != nil {
		return false, err
	}

	for _, result := range results {
		if err := r.emitInvocation(model.EngineEventKindToolResultCommitted, invocationId, map[string]interface{}{
			"requestId": result.RequestId,
			"name":      result.Name,
			"outcome":   string(result.Outcome),
		}); err != nil {
			return false, err
		}
	}

	// Without a conversation port the tool results never reach the next model
	// call. That is not a degraded mode, it is a broken one: the provider sees
	// tool calls it made with no answers, and either rejects the request or
	// invents the results. Failing here is the only honest outcome.
	if r.ports.Conversation == nil {
		return false, fmt.Errorf(
			"%s: the model requested %d tool(s) but no ConversationPort is configured to return their results",
			ErrorKindConversation, len(response.ToolRequests))
	}
	followUp, err := r.ports.Conversation.FormatToolExchange(response, results)
	if err != nil {
		return false, fmt.Errorf("%s: %w", ErrorKindConversation, err)
	}
	r.messages = append(r.messages, followUp...)

	if err := r.emitInvocation(model.EngineEventKindConversationUpdated, invocationId, map[string]interface{}{
		"messages": len(r.messages),
	}); err != nil {
		return false, err
	}
	return false, r.checkpoint(nil, nil)
}

// invokeModel calls the model port, retrying within the attempt budget.
func (r *turnRun) invokeModel(
	invocationId string,
	snapshot model.ModelInvocationContextSnapshot,
) (model.ModelInvocationResponse, error) {
	request := model.ModelInvocationRequest{Context: snapshot}

	var lastErr error
	for attempt := int32(0); attempt < r.maxModelAttempts; attempt++ {
		if err := r.ctx.Err(); err != nil {
			return model.ModelInvocationResponse{}, err
		}
		if err := r.emitInvocation(model.EngineEventKindModelInvocationStarted, invocationId, map[string]interface{}{
			"attempt": attempt,
		}); err != nil {
			return model.ModelInvocationResponse{}, err
		}

		response, err := r.ports.Model.Invoke(r.ctx, request)
		if err == nil {
			if err := r.emitInvocation(model.EngineEventKindModelInvocationCompleted, invocationId, map[string]interface{}{
				"attempt":      attempt,
				"toolRequests": len(response.ToolRequests),
			}); err != nil {
				return model.ModelInvocationResponse{}, err
			}
			return response, nil
		}

		lastErr = err
		if emitErr := r.emitInvocation(model.EngineEventKindModelInvocationFailed, invocationId, map[string]interface{}{
			"attempt": attempt,
			"message": err.Error(),
		}); emitErr != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return model.ModelInvocationResponse{},
					fmt.Errorf("%w (also failed to record model cancellation: %v)", err, emitErr)
			}
			return model.ModelInvocationResponse{}, emitErr
		}

		// A cancelled call is not a transient failure, and neither is one the
		// port marked permanent; retrying either wastes the budget and delays
		// the error the caller needs to see.
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrPermanent) {
			return model.ModelInvocationResponse{}, err
		}
		if attempt+1 >= r.maxModelAttempts {
			break
		}
		if r.ports.Retry != nil {
			if backoffErr := r.ports.Retry.Backoff(r.ctx, model.RetryPolicyRequest{
				FailedAttempts: attempt + 1,
				NextAttempt:    attempt + 2,
				MaxAttempts:    r.maxModelAttempts,
				Reason:         err.Error(),
			}); backoffErr != nil {
				return model.ModelInvocationResponse{}, backoffErr
			}
		}
	}

	// The attempt budget is exhausted. The reconciliation state records the
	// request that could not be completed, so a resume can decide whether the
	// provider may already have acted on it.
	r.reconciliation = &model.ModelReconciliationState{
		InvocationId:  invocationId,
		Request:       request,
		FailedAttempt: r.maxModelAttempts - 1,
		Message:       lastErr.Error(),
	}
	return model.ModelInvocationResponse{}, fmt.Errorf("%s: %w", ErrorKindModelFailed, lastErr)
}

// runToolRound authorizes and executes every requested tool, in request order.
//
// Order is the model's, not completion order: a provider rejects a tool_calls
// block whose results do not line up with the calls it made.
func (r *turnRun) runToolRound(response model.ModelInvocationResponse) ([]model.ModelToolResult, error) {
	results := make([]model.ModelToolResult, 0, len(response.ToolRequests))

	for _, toolRequest := range response.ToolRequests {
		if err := r.ctx.Err(); err != nil {
			return nil, err
		}

		if err := r.emit(model.EngineEventKindPermissionRequested, nil, map[string]interface{}{
			"requestId": toolRequest.Id,
			"name":      toolRequest.Name,
		}); err != nil {
			return nil, err
		}

		decision, err := r.ports.Permission.Authorize(r.ctx, toolRequest)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", ErrorKindPermissionDenied, err)
		}
		if err := r.emit(model.EngineEventKindPermissionResolved, nil, map[string]interface{}{
			"requestId": toolRequest.Id,
			"name":      toolRequest.Name,
			"approved":  decision.Approved,
			"reason":    stringOrNil(decision.Reason),
		}); err != nil {
			return nil, err
		}

		var result model.ModelToolResult
		if decision.Approved {
			if r.ports.Tool == nil {
				return nil, fmt.Errorf("%s: no ToolPort is configured but the model requested %q",
					ErrorKindToolDispatch, toolRequest.Name)
			}
			if err := r.emit(model.EngineEventKindToolExecutionStarted, nil, map[string]interface{}{
				"requestId": toolRequest.Id,
				"name":      toolRequest.Name,
			}); err != nil {
				return nil, err
			}
			result, err = r.ports.Tool.Execute(r.ctx, toolRequest)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", ErrorKindToolDispatch, err)
			}
			if err := r.emit(model.EngineEventKindToolExecutionCompleted, nil, map[string]interface{}{
				"requestId": result.RequestId,
				"name":      result.Name,
				"outcome":   string(result.Outcome),
			}); err != nil {
				return nil, err
			}
		} else {
			// A refusal still produces a result. The model asked, and it must
			// be told the answer is no — otherwise the next call carries an
			// unanswered tool call and the provider rejects it.
			result = deniedResult(toolRequest, decision)
		}

		results = append(results, result)
		r.toolResults = append(r.toolResults, result)

		if err := r.checkpoint(nil, results); err != nil {
			return nil, err
		}
	}
	return results, nil
}

func deniedResult(request model.ModelToolRequest, decision model.EnginePermissionDecision) model.ModelToolResult {
	reason := "Permission denied"
	if decision.Reason != nil && *decision.Reason != "" {
		reason = *decision.Reason
	}
	errorKind := ErrorKindPermissionDenied
	output := interface{}(reason)
	return model.ModelToolResult{
		RequestId: request.Id,
		Name:      request.Name,
		Outcome:   model.ModelToolOutcomeFailed,
		Output:    &output,
		ErrorKind: &errorKind,
	}
}

// ---------------------------------------------------------------------------
// Terminal states
// ---------------------------------------------------------------------------

func (r *turnRun) cancel(cause error) (Result, error) {
	if err := r.emit(model.EngineEventKindTurnCancelled, nil, map[string]interface{}{
		"reason": cause.Error(),
	}); err != nil {
		return r.result(), fmt.Errorf("%w (also failed to record cancellation: %v)", cause, err)
	}
	commit := r.buildCommit(model.EngineTurnStatusCancelled, nil)
	result := r.result()
	result.Commit = commit
	return result, cause
}

func (r *turnRun) fail(cause error) (Result, error) {
	if errors.Is(cause, context.Canceled) || errors.Is(cause, context.DeadlineExceeded) {
		return r.cancel(cause)
	}
	result, err := r.commitFailed(errorKindOf(cause), cause.Error())
	// A durability or sink failure during the failure path is the more
	// important error: it means the record of the failure was not written.
	var failed *TurnFailedError
	if err != nil && !errors.As(err, &failed) {
		return result, err
	}
	return result, cause
}

// commitFailed records a failed turn and returns a *TurnFailedError, so a
// caller that only inspects the error still learns the turn produced no answer.
func (r *turnRun) commitFailed(errorKind string, message string) (Result, error) {
	if err := r.emit(model.EngineEventKindTurnFailed, nil, map[string]interface{}{
		"errorKind": errorKind,
		"message":   message,
	}); err != nil {
		return r.result(), err
	}
	output := interface{}(map[string]interface{}{"errorKind": errorKind, "message": message})
	result, err := r.commit(model.EngineTurnStatusFailed, &output)
	if err != nil {
		return result, err
	}
	return result, &TurnFailedError{ErrorKind: errorKind, Message: message}
}

// commit finalises the turn: the host's pre-commit policy, the commit event and
// record, and then the post-commit phase.
func (r *turnRun) commit(status model.EngineTurnStatus, failureOutput *interface{}) (Result, error) {
	output := r.output
	if failureOutput != nil {
		output = failureOutput
	}

	if status == model.EngineTurnStatusSuccess {
		policy, err := r.ports.HostPolicy.BeforeCommit(r.ctx, model.FinalOutputPolicyRequest{
			SessionId: r.request.SessionId,
			TurnId:    r.request.TurnId,
			Iteration: r.completed,
			Messages:  r.messages,
			Output:    output,
			Inputs:    boxInputs(r.request.Inputs),
		})
		if err != nil {
			// A pre-commit refusal must not silently become a successful turn.
			return r.commitFailed(ErrorKindPolicyFailed, err.Error())
		}
		output = policy.Output
	}
	r.output = output

	// The commit event is emitted before the commit record is built so
	// TurnCommit.LastSequence names the sequence the turn actually ended at,
	// rather than the last event before it.
	if err := r.emit(model.EngineEventKindTurnCommitted, nil, map[string]interface{}{
		"status":     string(status),
		"iterations": r.completed,
	}); err != nil {
		return r.result(), err
	}

	commit := r.buildCommit(status, output)

	postCommitError := r.runPostCommit(commit)

	result := r.result()
	result.Commit = commit
	result.PostCommitError = postCommitError
	return result, nil
}

// runPostCommit runs after-commit side effects and reports a failure rather
// than returning it: the turn is already committed, and unwinding it because a
// notification failed would lose work the caller has been told succeeded.
func (r *turnRun) runPostCommit(commit model.TurnCommit) *string {
	effectId := r.clock.NextID("effect")
	if err := r.emit(model.EngineEventKindPostCommitStarted, nil, map[string]interface{}{
		"effectId": effectId,
	}); err != nil {
		message := err.Error()
		return &message
	}

	if r.ports.PostCommit != nil {
		if err := r.ports.PostCommit.AfterCommit(r.ctx, effectId, commit); err != nil {
			message := err.Error()
			if emitErr := r.emit(model.EngineEventKindPostCommitFailed, nil, map[string]interface{}{
				"effectId": effectId,
				"message":  message,
			}); emitErr != nil {
				combined := fmt.Sprintf("%s; and the failure could not be recorded: %v", message, emitErr)
				return &combined
			}
			return &message
		}
	}

	if err := r.emit(model.EngineEventKindPostCommitCompleted, nil, map[string]interface{}{
		"effectId": effectId,
	}); err != nil {
		message := err.Error()
		return &message
	}
	return nil
}

func (r *turnRun) buildCommit(status model.EngineTurnStatus, output *interface{}) model.TurnCommit {
	return model.TurnCommit{
		SessionId:           r.request.SessionId,
		TurnId:              r.request.TurnId,
		Status:              status,
		Output:              output,
		Messages:            append([]model.Message(nil), r.messages...),
		Iterations:          r.completed,
		LastSequence:        r.sequence,
		ContextState:        r.contextState,
		ModelReconciliation: r.reconciliation,
	}
}

func (r *turnRun) result() Result {
	return Result{
		TurnEngineResult: model.TurnEngineResult{
			Snapshots:   append([]model.ModelInvocationContextSnapshot(nil), r.snapshots...),
			ToolResults: append([]model.ModelToolResult(nil), r.toolResults...),
		},
		Events:      append([]model.EngineEvent(nil), r.events...),
		Checkpoints: append([]model.EngineCheckpoint(nil), r.checkpoints...),
	}
}

// ---------------------------------------------------------------------------
// Durability
// ---------------------------------------------------------------------------

func (r *turnRun) emit(kind model.EngineEventKind, invocationId *string, payload map[string]interface{}) error {
	event := r.newEvent(kind, invocationId, payload)
	r.events = append(r.events, event)
	if r.ports.Durability == nil {
		return nil
	}
	if err := r.ports.Durability.Append(r.ctx, event); err != nil {
		return fmt.Errorf("%s: %w", ErrorKindDurabilityFailure, err)
	}
	return nil
}

func (r *turnRun) emitInvocation(kind model.EngineEventKind, invocationId string, payload map[string]interface{}) error {
	return r.emit(kind, &invocationId, payload)
}

func (r *turnRun) newEvent(
	kind model.EngineEventKind,
	invocationId *string,
	payload map[string]interface{},
) model.EngineEvent {
	r.sequence++
	iteration := r.iteration
	event := model.EngineEvent{
		Sequence:        r.sequence,
		Id:              r.clock.NextID("event"),
		Timestamp:       r.clock.Now(),
		SessionId:       r.request.SessionId,
		TurnId:          r.request.TurnId,
		RunId:           r.runId,
		ParentRunId:     r.request.ParentRunId,
		DelegationDepth: r.request.DelegationDepth,
		InvocationId:    invocationId,
		Iteration:       &iteration,
		Kind:            kind,
	}
	if payload != nil {
		boxed := interface{}(payload)
		event.Payload = &boxed
	}
	return event
}

// checkpoint saves the resumable state and announces it.
//
// The checkpoint_created event and the checkpoint itself are written through
// AppendWithCheckpoint so a durable store can commit them together: a journal
// that announced a checkpoint the store does not have would replay into a state
// that never existed.
func (r *turnRun) checkpoint(response *model.ModelInvocationResponse, pending []model.ModelToolResult) error {
	r.sequence++
	iteration := r.iteration
	event := model.EngineEvent{
		Sequence:        r.sequence,
		Id:              r.clock.NextID("event"),
		Timestamp:       r.clock.Now(),
		SessionId:       r.request.SessionId,
		TurnId:          r.request.TurnId,
		RunId:           r.runId,
		ParentRunId:     r.request.ParentRunId,
		DelegationDepth: r.request.DelegationDepth,
		Iteration:       &iteration,
		Kind:            model.EngineEventKindCheckpointCreated,
	}

	checkpoint := model.EngineCheckpoint{
		Id:                       r.clock.NextID("checkpoint"),
		SessionId:                r.request.SessionId,
		TurnId:                   r.request.TurnId,
		RunId:                    r.runId,
		ParentRunId:              r.request.ParentRunId,
		DelegationDepth:          r.request.DelegationDepth,
		Iteration:                r.iteration,
		LastSequence:             r.sequence,
		Messages:                 append([]model.Message(nil), r.messages...),
		StablePrefixMessages:     r.stablePrefix,
		Inputs:                   boxInputs(r.request.Inputs),
		CompletedToolResults:     append([]model.ModelToolResult(nil), pending...),
		CompletedModelIterations: r.completed,
		ContextState:             r.contextState,
		ModelReconciliation:      r.reconciliation,
	}
	if response != nil {
		checkpoint.PendingToolRequests = response.ToolRequests
		checkpoint.PendingModelResponse = response
		checkpoint.FinalOutputReady = len(response.ToolRequests) == 0
		checkpoint.PendingOutput = response.Output
	}

	payload := interface{}(map[string]interface{}{
		"checkpointId": checkpoint.Id,
		"iteration":    r.iteration,
	})
	event.Payload = &payload

	r.events = append(r.events, event)
	r.checkpoints = append(r.checkpoints, checkpoint)

	if r.ports.Durability == nil {
		return nil
	}
	if err := r.ports.Durability.AppendWithCheckpoint(r.ctx, []model.EngineEvent{event}, checkpoint); err != nil {
		return fmt.Errorf("%s: %w", ErrorKindDurabilityFailure, err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func boxInputs(inputs map[string]interface{}) *interface{} {
	if inputs == nil {
		return nil
	}
	boxed := interface{}(inputs)
	return &boxed
}

func stringOrNil(value *string) interface{} {
	if value == nil {
		return nil
	}
	return *value
}

func min32(left, right int32) int32 {
	if left < right {
		return left
	}
	return right
}

func clampPrefix(prefix int32, length int32) int32 {
	if prefix < 0 {
		return 0
	}
	if prefix > length {
		return length
	}
	return prefix
}

// errorKindOf recovers the stable discriminator an internal failure was tagged
// with, so a turn_failed event names the phase rather than repeating prose.
func errorKindOf(err error) string {
	message := err.Error()
	for _, kind := range []string{
		ErrorKindMaxIterations,
		ErrorKindModelFailed,
		ErrorKindPermissionDenied,
		ErrorKindToolDispatch,
		ErrorKindPolicyFailed,
		ErrorKindConversation,
		ErrorKindDurabilityFailure,
	} {
		if len(message) >= len(kind) && message[:len(kind)] == kind {
			return kind
		}
	}
	return "error"
}
