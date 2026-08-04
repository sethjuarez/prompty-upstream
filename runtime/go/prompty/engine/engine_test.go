package engine_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	engine "prompty/engine"
	model "prompty/model"
)

func answeringModel(output string) engine.ModelFunc {
	return func(context.Context, model.ModelInvocationRequest) (model.ModelInvocationResponse, error) {
		boxed := interface{}(output)
		return model.ModelInvocationResponse{Output: &boxed}, nil
	}
}

func toolRequestingModel(name string) engine.ModelFunc {
	return func(context.Context, model.ModelInvocationRequest) (model.ModelInvocationResponse, error) {
		return model.ModelInvocationResponse{
			ToolRequests: []model.ModelToolRequest{{Id: "call-1", Name: name}},
		}, nil
	}
}

func baseRequest(ports engine.Ports) engine.Request {
	return engine.Request{
		SessionId: "session-1",
		TurnId:    "turn-1",
		Messages:  []model.Message{model.NewUserMessage("hello")},
		Ports:     ports,
	}
}

func TestRunRequiresAModelPort(t *testing.T) {
	if _, err := engine.Run(context.Background(), engine.Request{}); !errors.Is(err, engine.ErrNoModelPort) {
		t.Fatalf("err = %v, want ErrNoModelPort", err)
	}
}

func TestMaxIterationsFailsTheTurnRatherThanReturningNothing(t *testing.T) {
	// A turn that ran out of rounds has no answer. Reporting success with an
	// empty output would hide that from a caller who asked a question.
	durability := &engine.RecordingDurabilityPort{}
	request := baseRequest(engine.Ports{
		Model: toolRequestingModel("spin"),
		Tool: engine.ToolFunc(func(context.Context, model.ModelToolRequest) (model.ModelToolResult, error) {
			return model.ModelToolResult{RequestId: "call-1", Name: "spin", Outcome: model.ModelToolOutcomeSuccess}, nil
		}),
		Conversation: echoConversation(),
		Durability:   durability,
	})
	request.MaxIterations = 2

	result, err := engine.Run(context.Background(), request)
	if !errors.Is(err, engine.ErrTurnFailed) {
		t.Fatalf("err = %v, want ErrTurnFailed — a turn with no answer must not report success", err)
	}
	var failed *engine.TurnFailedError
	if !errors.As(err, &failed) || failed.ErrorKind != engine.ErrorKindMaxIterations {
		t.Errorf("err = %v, want a TurnFailedError naming max_iterations", err)
	}
	if result.Commit.Status != model.EngineTurnStatusFailed {
		t.Errorf("status = %q, want failed", result.Commit.Status)
	}
	if !containsKind(durability.EventKinds(), string(model.EngineEventKindTurnFailed)) {
		t.Errorf("event kinds %v are missing turn_failed", durability.EventKinds())
	}
	payload := failurePayload(t, durability)
	if payload["errorKind"] != engine.ErrorKindMaxIterations {
		t.Errorf("errorKind = %v, want %q", payload["errorKind"], engine.ErrorKindMaxIterations)
	}
	if result.Commit.Iterations != 2 {
		t.Errorf("iterations = %d, want 2", result.Commit.Iterations)
	}
}

func TestModelFailureIsRetriedWithinTheAttemptBudget(t *testing.T) {
	var attempts int
	backoffs := 0

	durability := &engine.RecordingDurabilityPort{}
	request := baseRequest(engine.Ports{
		Model: engine.ModelFunc(func(context.Context, model.ModelInvocationRequest) (model.ModelInvocationResponse, error) {
			attempts++
			if attempts < 3 {
				return model.ModelInvocationResponse{}, errors.New("transient")
			}
			boxed := interface{}("recovered")
			return model.ModelInvocationResponse{Output: &boxed}, nil
		}),
		Retry: retryFunc(func(context.Context, model.RetryPolicyRequest) error {
			backoffs++
			return nil
		}),
		Durability: durability,
	})
	request.MaxModelAttempts = 3

	result, err := engine.Run(context.Background(), request)
	if err != nil {
		t.Fatalf("a recoverable failure must not fail the turn: %v", err)
	}
	if attempts != 3 {
		t.Errorf("model attempts = %d, want 3", attempts)
	}
	if backoffs != 2 {
		t.Errorf("backoffs = %d, want one per retry (2)", backoffs)
	}
	if result.Commit.Status != model.EngineTurnStatusSuccess {
		t.Errorf("status = %q, want success", result.Commit.Status)
	}
	// Every failed attempt must be visible; a retry that leaves no trace makes
	// a slow, flaky provider indistinguishable from a fast one.
	failures := countKind(durability.EventKinds(), string(model.EngineEventKindModelInvocationFailed))
	if failures != 2 {
		t.Errorf("model_invocation_failed events = %d, want 2", failures)
	}
}

func TestPermanentModelFailureIsNotRetried(t *testing.T) {
	var attempts int
	request := baseRequest(engine.Ports{
		Model: engine.ModelFunc(func(context.Context, model.ModelInvocationRequest) (model.ModelInvocationResponse, error) {
			attempts++
			return model.ModelInvocationResponse{}, fmt.Errorf("bad credential: %w", engine.ErrPermanent)
		}),
	})
	request.MaxModelAttempts = 5

	if _, err := engine.Run(context.Background(), request); err == nil {
		t.Fatal("a permanent failure must fail the turn")
	}
	if attempts != 1 {
		t.Errorf("model attempts = %d, want 1 — a permanent failure must not burn the retry budget", attempts)
	}
}

func TestExhaustedRetriesRecordReconciliationState(t *testing.T) {
	// A resume has to know a model call may already have had an effect the
	// engine never saw the result of.
	request := baseRequest(engine.Ports{
		Model: engine.ModelFunc(func(context.Context, model.ModelInvocationRequest) (model.ModelInvocationResponse, error) {
			return model.ModelInvocationResponse{}, errors.New("timeout")
		}),
	})
	request.MaxModelAttempts = 2

	result, _ := engine.Run(context.Background(), request)
	if result.Commit.ModelReconciliation == nil {
		t.Fatal("an exhausted model invocation must record reconciliation state")
	}
	if result.Commit.ModelReconciliation.Message != "timeout" {
		t.Errorf("reconciliation message = %q, want the underlying failure", result.Commit.ModelReconciliation.Message)
	}
}

func TestCancellationDuringAToolRoundStopsBeforeTheNextTool(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	executed := 0

	request := baseRequest(engine.Ports{
		Model: engine.ModelFunc(func(context.Context, model.ModelInvocationRequest) (model.ModelInvocationResponse, error) {
			return model.ModelInvocationResponse{
				ToolRequests: []model.ModelToolRequest{
					{Id: "call-a", Name: "echo"},
					{Id: "call-b", Name: "echo"},
				},
			}, nil
		}),
		Tool: engine.ToolFunc(func(context.Context, model.ModelToolRequest) (model.ModelToolResult, error) {
			executed++
			cancel()
			return model.ModelToolResult{Outcome: model.ModelToolOutcomeSuccess}, nil
		}),
	})

	result, err := engine.Run(ctx, request)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if executed != 1 {
		t.Errorf("executed %d tools, want 1 — cancellation must stop the round", executed)
	}
	if result.Commit.Status != model.EngineTurnStatusCancelled {
		t.Errorf("status = %q, want cancelled", result.Commit.Status)
	}
}

func TestCancellationCauseSurvivesDurabilityFailure(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	request := baseRequest(engine.Ports{
		Model: engine.ModelFunc(func(context.Context, model.ModelInvocationRequest) (model.ModelInvocationResponse, error) {
			cancel()
			return model.ModelInvocationResponse{}, context.Canceled
		}),
		Durability: cancelAwareDurability{},
	})

	_, err := engine.Run(ctx, request)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled to remain discoverable", err)
	}
	if !strings.Contains(err.Error(), "failed to record cancellation") {
		t.Fatalf("err = %v, want the durability failure to remain visible", err)
	}
}

func TestToolDispatchFailureFailsTheTurn(t *testing.T) {
	// A tool that could not be dispatched at all is different from one that
	// ran and failed: inventing a result would teach the model the capability
	// exists.
	request := baseRequest(engine.Ports{
		Model: toolRequestingModel("ghost"),
		Tool: engine.ToolFunc(func(context.Context, model.ModelToolRequest) (model.ModelToolResult, error) {
			return model.ModelToolResult{}, errors.New("no such tool")
		}),
	})
	result, err := engine.Run(context.Background(), request)
	if err == nil {
		t.Fatal("a dispatch failure must fail the turn")
	}
	if result.Commit.Status != model.EngineTurnStatusFailed {
		t.Errorf("status = %q, want failed", result.Commit.Status)
	}
}

func TestAModelRequestingToolsWithNoToolPortFails(t *testing.T) {
	request := baseRequest(engine.Ports{Model: toolRequestingModel("anything")})
	if _, err := engine.Run(context.Background(), request); err == nil {
		t.Fatal("a tool request with no ToolPort must fail rather than be ignored")
	}
}

func TestPreCommitPolicyRefusalDoesNotBecomeASuccessfulTurn(t *testing.T) {
	request := baseRequest(engine.Ports{
		Model:      answeringModel("unsafe"),
		HostPolicy: refusingCommitPolicy{reason: "output policy refused"},
	})
	result, err := engine.Run(context.Background(), request)
	if !errors.Is(err, engine.ErrTurnFailed) {
		t.Fatalf("err = %v, want ErrTurnFailed", err)
	}
	var failed *engine.TurnFailedError
	if !errors.As(err, &failed) || failed.ErrorKind != engine.ErrorKindPolicyFailed {
		t.Errorf("err = %v, want a TurnFailedError naming the policy phase", err)
	}
	if result.Commit.Status != model.EngineTurnStatusFailed {
		t.Errorf("status = %q, want failed", result.Commit.Status)
	}
}

func TestPostCommitFailureIsReportedNotFatal(t *testing.T) {
	// The turn already succeeded. Unwinding it because a notification failed
	// would lose work the caller has been told about.
	durability := &engine.RecordingDurabilityPort{}
	request := baseRequest(engine.Ports{
		Model:      answeringModel("done"),
		Durability: durability,
		PostCommit: engine.PostCommitFunc(func(context.Context, string, model.TurnCommit) error {
			return errors.New("webhook unreachable")
		}),
	})

	result, err := engine.Run(context.Background(), request)
	if err != nil {
		t.Fatalf("a post-commit failure must not fail the turn: %v", err)
	}
	if result.Commit.Status != model.EngineTurnStatusSuccess {
		t.Errorf("status = %q, want success", result.Commit.Status)
	}
	if result.PostCommitError == nil || *result.PostCommitError != "webhook unreachable" {
		t.Errorf("postCommitError = %v, want the effect's message", result.PostCommitError)
	}
	if !containsKind(durability.EventKinds(), string(model.EngineEventKindPostCommitFailed)) {
		t.Errorf("event kinds %v are missing post_commit_failed", durability.EventKinds())
	}
}

func TestDurabilityFailureStopsTheTurn(t *testing.T) {
	// A journal that silently falls behind produces a turn nobody can replay,
	// which is worse than a turn that failed loudly.
	request := baseRequest(engine.Ports{
		Model:      answeringModel("done"),
		Durability: failingDurability{err: errors.New("journal full")},
	})
	if _, err := engine.Run(context.Background(), request); err == nil {
		t.Fatal("a durability failure must stop the turn")
	}
}

func TestCheckpointsCarryTheConversationAndContextState(t *testing.T) {
	durability := &engine.RecordingDurabilityPort{}
	request := baseRequest(engine.Ports{
		Model: engine.ModelFunc(func(context.Context, model.ModelInvocationRequest) (model.ModelInvocationResponse, error) {
			boxed := interface{}("done")
			return model.ModelInvocationResponse{
				Output: &boxed,
				NextContextState: &model.InvocationContextState{
					Portability: model.InvocationContextPortabilityDelegated,
					DelegatedState: []model.DelegatedStateReference{
						{Provider: "openai", Kind: "previous_response", Id: "resp_1"},
					},
				},
			}, nil
		}),
		Durability: durability,
	})

	result, err := engine.Run(context.Background(), request)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	checkpoints := durability.Checkpoints()
	if len(checkpoints) != 1 {
		t.Fatalf("checkpoints = %d, want 1", len(checkpoints))
	}
	checkpoint := checkpoints[0]
	if checkpoint.ContextState.Portability != model.InvocationContextPortabilityDelegated {
		t.Errorf("checkpoint portability = %q, want delegated", checkpoint.ContextState.Portability)
	}
	if len(checkpoint.Messages) != 1 {
		t.Errorf("checkpoint carried %d messages, want the conversation", len(checkpoint.Messages))
	}
	if !checkpoint.FinalOutputReady {
		t.Error("a checkpoint after a final answer must mark the output ready")
	}
	if result.Commit.LastSequence <= checkpoint.LastSequence {
		t.Errorf("commit sequence %d is not after the checkpoint's %d",
			result.Commit.LastSequence, checkpoint.LastSequence)
	}
}

func TestEventSequenceIsStrictlyIncreasing(t *testing.T) {
	// The sequence is what orders a replay. A gap or a repeat makes the
	// journal ambiguous.
	durability := &engine.RecordingDurabilityPort{}
	request := baseRequest(engine.Ports{
		Model: engine.ModelFunc(func(_ context.Context, request model.ModelInvocationRequest) (model.ModelInvocationResponse, error) {
			if request.Context.Iteration == 0 {
				return model.ModelInvocationResponse{
					ToolRequests: []model.ModelToolRequest{{Id: "call-1", Name: "echo"}},
				}, nil
			}
			boxed := interface{}("done")
			return model.ModelInvocationResponse{Output: &boxed}, nil
		}),
		Tool: engine.ToolFunc(func(_ context.Context, request model.ModelToolRequest) (model.ModelToolResult, error) {
			return model.ModelToolResult{RequestId: request.Id, Name: request.Name, Outcome: model.ModelToolOutcomeSuccess}, nil
		}),
		Conversation: echoConversation(),
		Durability:   durability,
	})

	if _, err := engine.Run(context.Background(), request); err != nil {
		t.Fatalf("run: %v", err)
	}
	events := durability.Events()
	for index, event := range events {
		if event.Sequence != int64(index+1) {
			t.Fatalf("event %d has sequence %d, want %d", index, event.Sequence, index+1)
		}
		if event.SessionId != "session-1" || event.TurnId != "turn-1" {
			t.Errorf("event %d is not attributed to the turn: %+v", index, event)
		}
		if event.RunId == "" {
			t.Errorf("event %d has no run id", index)
		}
	}
}

func TestDeterministicClockProducesIdenticalEventIdentifiers(t *testing.T) {
	run := func() []string {
		durability := &engine.RecordingDurabilityPort{}
		request := baseRequest(engine.Ports{
			Model:      answeringModel("done"),
			Durability: durability,
			Clock:      &engine.FixedClock{Timestamp: "2026-06-28T00:00:00Z"},
		})
		if _, err := engine.Run(context.Background(), request); err != nil {
			t.Fatalf("run: %v", err)
		}
		ids := make([]string, 0)
		for _, event := range durability.Events() {
			ids = append(ids, event.Id+"@"+event.Timestamp)
		}
		return ids
	}

	first, second := run(), run()
	if !equalStrings(first, second) {
		t.Errorf("two deterministic runs diverged:\n%v\nvs\n%v", first, second)
	}
}

func TestStablePrefixNeverExceedsTheConversation(t *testing.T) {
	// A host policy that drops history cannot leave a prefix pointing past the
	// end of the conversation it returned.
	request := baseRequest(engine.Ports{
		Model:      answeringModel("done"),
		HostPolicy: shrinkingPolicy{},
	})
	request.Messages = []model.Message{
		model.NewSystemMessage("system"),
		model.NewUserMessage("one"),
		model.NewUserMessage("two"),
	}

	result, err := engine.Run(context.Background(), request)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	snapshot := result.Snapshots[0]
	if int(snapshot.StablePrefixMessages) > len(snapshot.Messages) {
		t.Errorf("stable prefix %d exceeds the %d messages in the snapshot",
			snapshot.StablePrefixMessages, len(snapshot.Messages))
	}
}

func TestConcurrentTurnsDoNotShareState(t *testing.T) {
	// Each Run owns its own state. Two turns driven at once must not interleave
	// sequences, snapshots or commits.
	const turns = 12
	var (
		wait    sync.WaitGroup
		results = make([]engine.Result, turns)
		errs    = make([]error, turns)
	)
	for index := 0; index < turns; index++ {
		wait.Add(1)
		go func(slot int) {
			defer wait.Done()
			request := baseRequest(engine.Ports{
				Model:      answeringModel(fmt.Sprintf("answer-%d", slot)),
				Durability: &engine.RecordingDurabilityPort{},
			})
			request.TurnId = fmt.Sprintf("turn-%d", slot)
			results[slot], errs[slot] = engine.Run(context.Background(), request)
		}(index)
	}
	wait.Wait()

	for index := range results {
		if errs[index] != nil {
			t.Fatalf("turn %d: %v", index, errs[index])
		}
		if results[index].Commit.TurnId != fmt.Sprintf("turn-%d", index) {
			t.Errorf("turn %d committed as %q", index, results[index].Commit.TurnId)
		}
		want := fmt.Sprintf("answer-%d", index)
		if got := fmt.Sprint(*results[index].Commit.Output); got != want {
			t.Errorf("turn %d output = %q, want %q", index, got, want)
		}
	}
}

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

type retryFunc func(ctx context.Context, request model.RetryPolicyRequest) error

func (f retryFunc) Backoff(ctx context.Context, request model.RetryPolicyRequest) error {
	return f(ctx, request)
}

type failingDurability struct{ err error }

func (d failingDurability) Append(context.Context, model.EngineEvent) error { return d.err }

func (d failingDurability) AppendWithCheckpoint(context.Context, []model.EngineEvent, model.EngineCheckpoint) error {
	return d.err
}

type cancelAwareDurability struct{}

func (cancelAwareDurability) Append(ctx context.Context, _ model.EngineEvent) error {
	if err := ctx.Err(); err != nil {
		return errors.New("journal unavailable after cancellation")
	}
	return nil
}

func (cancelAwareDurability) AppendWithCheckpoint(
	ctx context.Context,
	_ []model.EngineEvent,
	_ model.EngineCheckpoint,
) error {
	return ctx.Err()
}

type refusingCommitPolicy struct{ reason string }

func (p refusingCommitPolicy) BeforeModel(
	_ context.Context,
	request model.HostPolicyRequest,
) (model.HostPolicyResult, error) {
	return model.HostPolicyResult{
		Messages:             request.Messages,
		StablePrefixMessages: request.StablePrefixMessages,
	}, nil
}

func (p refusingCommitPolicy) BeforeCommit(
	context.Context,
	model.FinalOutputPolicyRequest,
) (model.FinalOutputPolicyResult, error) {
	return model.FinalOutputPolicyResult{}, errors.New(p.reason)
}

// shrinkingPolicy drops all but the last message while claiming the original
// stable prefix, which the engine must clamp.
type shrinkingPolicy struct{}

func (shrinkingPolicy) BeforeModel(
	_ context.Context,
	request model.HostPolicyRequest,
) (model.HostPolicyResult, error) {
	return model.HostPolicyResult{
		Messages:             request.Messages[len(request.Messages)-1:],
		StablePrefixMessages: request.StablePrefixMessages,
	}, nil
}

func (shrinkingPolicy) BeforeCommit(
	_ context.Context,
	request model.FinalOutputPolicyRequest,
) (model.FinalOutputPolicyResult, error) {
	return model.FinalOutputPolicyResult{Output: request.Output}, nil
}

func containsKind(kinds []string, want string) bool {
	return countKind(kinds, want) > 0
}

func countKind(kinds []string, want string) int {
	count := 0
	for _, kind := range kinds {
		if kind == want {
			count++
		}
	}
	return count
}

func failurePayload(t *testing.T, durability *engine.RecordingDurabilityPort) map[string]interface{} {
	t.Helper()
	for _, event := range durability.Events() {
		if event.Kind != model.EngineEventKindTurnFailed || event.Payload == nil {
			continue
		}
		if payload, ok := (*event.Payload).(map[string]interface{}); ok {
			return payload
		}
	}
	t.Fatal("no turn_failed payload was recorded")
	return nil
}

// echoConversation renders one tool message per result, which is the minimum a
// conversation port must do for the next model call to make sense.
func echoConversation() engine.ConversationPort {
	return engine.ConversationFunc(func(_ model.ModelInvocationResponse, results []model.ModelToolResult) ([]model.Message, error) {
		messages := make([]model.Message, 0, len(results))
		for _, result := range results {
			messages = append(messages, model.Message{
				Role:     model.RoleTool,
				Parts:    []interface{}{model.TextPart{Kind: "text", Value: result.Name}},
				Metadata: map[string]interface{}{"tool_call_id": result.RequestId},
			})
		}
		return messages, nil
	})
}

func TestToolRoundWithNoConversationPortFailsLoudly(t *testing.T) {
	// Silently omitting the tool results would leave the provider with tool
	// calls it made and no answers, which it either rejects or hallucinates
	// around. Failing is the only honest outcome.
	request := baseRequest(engine.Ports{
		Model: toolRequestingModel("echo"),
		Tool: engine.ToolFunc(func(_ context.Context, r model.ModelToolRequest) (model.ModelToolResult, error) {
			return model.ModelToolResult{RequestId: r.Id, Name: r.Name, Outcome: model.ModelToolOutcomeSuccess}, nil
		}),
	})

	_, err := engine.Run(context.Background(), request)
	if err == nil {
		t.Fatal("a tool round with no ConversationPort must fail")
	}
	if !strings.Contains(err.Error(), engine.ErrorKindConversation) {
		t.Errorf("err = %v, want it to name the conversation phase", err)
	}
}
