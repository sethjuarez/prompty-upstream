package engine_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	engine "prompty/engine"
	model "prompty/model"
)

// specRoot walks up from the test's working directory until it finds the
// repository's spec/ directory, so the test works from a vendored or relocated
// module rather than a hard-coded relative path.
func specRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	start := dir
	for {
		candidate := filepath.Join(dir, "spec")
		if info, err := os.Stat(filepath.Join(candidate, "vectors")); err == nil && info.IsDir() {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skipf("shared spec/ directory not found above %s", start)
			return ""
		}
		dir = parent
	}
}

func readVectorFile(t *testing.T, relative string, target interface{}) {
	t.Helper()
	path := filepath.Join(specRoot(t), "vectors", filepath.FromSlash(relative))
	raw, err := os.ReadFile(path) // #nosec G304 -- derived from the repo layout.
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(raw, target); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}

// turnVectorFile is the shared engine turn vector file.
type turnVectorFile struct {
	Version string       `json:"version"`
	Cases   []turnVector `json:"cases"`
}

type turnVector struct {
	Name            string `json:"name"`
	CancelBeforeRun bool   `json:"cancelBeforeRun"`
	Messages        []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	Model       []turnVectorStep  `json:"model"`
	ToolOutputs map[string]string `json:"toolOutputs"`
	DenyTools   []string          `json:"denyTools"`
	Expected    struct {
		Status                 string   `json:"status"`
		Output                 string   `json:"output"`
		Iterations             int32    `json:"iterations"`
		Snapshots              int      `json:"snapshots"`
		SnapshotStablePrefixes []int32  `json:"snapshotStablePrefixes"`
		SnapshotPortability    []string `json:"snapshotPortability"`
		ToolResults            int      `json:"toolResults"`
		ToolResultOrder        []string `json:"toolResultOrder"`
		CommitPortability      string   `json:"commitPortability"`
		DelegatedState         int      `json:"delegatedState"`
		EventKinds             []string `json:"eventKinds"`
	} `json:"expected"`
}

type turnVectorStep struct {
	Assistant string `json:"assistant"`
	Output    string `json:"output"`
	Tools     []struct {
		ID        string                 `json:"id"`
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	} `json:"tools"`
	NextPortability string                          `json:"nextPortability"`
	DelegatedState  []model.DelegatedStateReference `json:"delegatedState"`
}

// TestEngineTurnVectors drives every case in the shared engine turn vectors
// through the real engine.
//
// Unlike the practical turn loop's vector test, which asserts only the
// observable outcome, this asserts the durable contract the vectors actually
// pin: the ordered engine event kinds, the per-invocation context snapshots and
// their stable prefixes, the context portability carried into each snapshot and
// into the commit, and the delegated state the provider handed back.
func TestEngineTurnVectors(t *testing.T) {
	var file turnVectorFile
	readVectorFile(t, "engine/turn_vectors.json", &file)
	if len(file.Cases) == 0 {
		t.Fatal("no engine turn vectors found")
	}

	executed := 0
	for _, vector := range file.Cases {
		vector := vector
		executed++
		t.Run(vector.Name, func(t *testing.T) { runEngineTurnVector(t, vector) })
	}
	t.Logf("engine turn vectors: %d executed, %d total", executed, len(file.Cases))
}

func runEngineTurnVector(t *testing.T, vector turnVector) {
	t.Helper()

	messages := make([]model.Message, 0, len(vector.Messages))
	for _, entry := range vector.Messages {
		messages = append(messages, model.Message{
			Role:  model.Role(entry.Role),
			Parts: []interface{}{model.TextPart{Kind: "text", Value: entry.Content}},
		})
	}

	denied := map[string]bool{}
	for _, name := range vector.DenyTools {
		denied[name] = true
	}

	durability := &engine.RecordingDurabilityPort{}
	steps := vector.Model
	invocations := 0
	// The messages each model invocation actually received, so the test can
	// prove the engine threaded the tool exchange forward rather than
	// silently dropping it.
	var observedMessages [][]model.Message

	ports := engine.Ports{
		Model: engine.ModelFunc(func(_ context.Context, request model.ModelInvocationRequest) (model.ModelInvocationResponse, error) {
			observedMessages = append(observedMessages, request.Context.Messages)
			if invocations >= len(steps) {
				return model.ModelInvocationResponse{}, fmt.Errorf(
					"the vector scripted %d model responses but the engine asked for %d", len(steps), invocations+1)
			}
			step := steps[invocations]
			invocations++
			return responseForStep(step), nil
		}),
		Permission: engine.PermissionFunc(func(_ context.Context, request model.ModelToolRequest) (model.EnginePermissionDecision, error) {
			if denied[request.Name] {
				reason := "permission denied"
				return model.EnginePermissionDecision{Approved: false, Reason: &reason}, nil
			}
			return model.EnginePermissionDecision{Approved: true}, nil
		}),
		Tool: engine.ToolFunc(func(_ context.Context, request model.ModelToolRequest) (model.ModelToolResult, error) {
			if denied[request.Name] {
				t.Errorf("a denied tool must never execute: %s", request.Name)
			}
			output := interface{}(vector.ToolOutputs[request.Id])
			return model.ModelToolResult{
				RequestId: request.Id,
				Name:      request.Name,
				Outcome:   model.ModelToolOutcomeSuccess,
				Output:    &output,
			}, nil
		}),
		// A real conversation port renders the exchange the provider expects.
		// This one is provider-neutral but not a no-op: it emits one tool
		// message per result, so the test can assert the engine actually
		// carried the tool round into the next model call.
		Conversation: engine.ConversationFunc(func(_ model.ModelInvocationResponse, results []model.ModelToolResult) ([]model.Message, error) {
			messages := make([]model.Message, 0, len(results))
			for _, result := range results {
				message := model.Message{
					Role:  model.RoleTool,
					Parts: []interface{}{model.TextPart{Kind: "text", Value: toolResultText(result)}},
					Metadata: map[string]interface{}{
						"tool_call_id": result.RequestId,
					},
				}
				messages = append(messages, message)
			}
			return messages, nil
		}),
		Durability: durability,
		PostCommit: engine.PostCommitFunc(func(context.Context, string, model.TurnCommit) error { return nil }),
		Clock:      &engine.FixedClock{Timestamp: "2026-06-28T00:00:00Z"},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if vector.CancelBeforeRun {
		cancel()
	}

	result, err := engine.Run(ctx, engine.Request{
		SessionId: "session-1",
		TurnId:    "turn-1",
		Messages:  messages,
		Ports:     ports,
	})

	if vector.Expected.Status == "cancelled" {
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected cancellation, got err=%v", err)
		}
	} else if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assertEngineOutcome(t, vector, result, durability)
	assertToolExchangeReachedTheModel(t, vector, observedMessages)
}

// assertToolExchangeReachedTheModel proves the engine carried each tool round
// into the following model call.
//
// Without this, a vector run passes on event ordering alone: the scripted model
// indexes its responses by call count and ignores what it was sent, so an
// engine that dropped every tool result would still produce the right events.
func assertToolExchangeReachedTheModel(t *testing.T, vector turnVector, observed [][]model.Message) {
	t.Helper()

	if len(observed) < 2 {
		return // single-invocation turns have no exchange to carry forward.
	}
	for index := 1; index < len(observed); index++ {
		previousTools := len(vector.Model[index-1].Tools)
		if previousTools == 0 {
			continue
		}
		grew := len(observed[index]) - len(observed[index-1])
		if grew != previousTools {
			t.Errorf("invocation %d saw %d messages, invocation %d saw %d — want %d more for the tool round",
				index-1, len(observed[index-1]), index, len(observed[index]), previousTools)
		}
		found := 0
		for _, message := range observed[index] {
			if message.Role == model.RoleTool {
				found++
			}
		}
		if found != previousTools {
			t.Errorf("invocation %d received %d tool messages, want %d", index, found, previousTools)
		}
	}
}

func toolResultText(result model.ModelToolResult) string {
	if result.Output == nil {
		return ""
	}
	return fmt.Sprint(*result.Output)
}

func responseForStep(step turnVectorStep) model.ModelInvocationResponse {
	response := model.ModelInvocationResponse{}

	if len(step.Tools) > 0 {
		response.ToolRequests = make([]model.ModelToolRequest, 0, len(step.Tools))
		for _, tool := range step.Tools {
			arguments := interface{}(tool.Arguments)
			response.ToolRequests = append(response.ToolRequests, model.ModelToolRequest{
				Id:        tool.ID,
				Name:      tool.Name,
				Arguments: &arguments,
			})
		}
	} else {
		output := interface{}(step.Output)
		response.Output = &output
	}

	if step.NextPortability != "" {
		response.NextContextState = &model.InvocationContextState{
			Portability:    model.InvocationContextPortability(step.NextPortability),
			DelegatedState: step.DelegatedState,
		}
	}
	return response
}

func assertEngineOutcome(
	t *testing.T,
	vector turnVector,
	result engine.Result,
	durability *engine.RecordingDurabilityPort,
) {
	t.Helper()
	expected := vector.Expected

	if got := string(result.Commit.Status); got != expected.Status {
		t.Errorf("commit status = %q, want %q", got, expected.Status)
	}
	if result.Commit.Iterations != expected.Iterations {
		t.Errorf("iterations = %d, want %d", result.Commit.Iterations, expected.Iterations)
	}
	if expected.Output != "" {
		got := ""
		if result.Commit.Output != nil {
			got = fmt.Sprint(*result.Commit.Output)
		}
		if got != expected.Output {
			t.Errorf("output = %q, want %q", got, expected.Output)
		}
	}

	if len(result.Snapshots) != expected.Snapshots {
		t.Errorf("snapshots = %d, want %d", len(result.Snapshots), expected.Snapshots)
	}
	if expected.SnapshotStablePrefixes != nil {
		got := make([]int32, 0, len(result.Snapshots))
		for _, snapshot := range result.Snapshots {
			got = append(got, snapshot.StablePrefixMessages)
		}
		if !equalInt32s(got, expected.SnapshotStablePrefixes) {
			t.Errorf("snapshot stable prefixes = %v, want %v", got, expected.SnapshotStablePrefixes)
		}
	}
	if expected.SnapshotPortability != nil {
		got := make([]string, 0, len(result.Snapshots))
		for _, snapshot := range result.Snapshots {
			got = append(got, string(snapshot.ContextState.Portability))
		}
		if !equalStrings(got, expected.SnapshotPortability) {
			t.Errorf("snapshot portability = %v, want %v", got, expected.SnapshotPortability)
		}
	}

	if len(result.ToolResults) != expected.ToolResults {
		t.Errorf("tool results = %d, want %d", len(result.ToolResults), expected.ToolResults)
	}
	if expected.ToolResultOrder != nil {
		got := make([]string, 0, len(result.ToolResults))
		for _, toolResult := range result.ToolResults {
			got = append(got, toolResult.RequestId)
		}
		if !equalStrings(got, expected.ToolResultOrder) {
			t.Errorf("tool result order = %v, want %v", got, expected.ToolResultOrder)
		}
	}

	if expected.CommitPortability != "" {
		if got := string(result.Commit.ContextState.Portability); got != expected.CommitPortability {
			t.Errorf("commit portability = %q, want %q", got, expected.CommitPortability)
		}
	}
	if expected.DelegatedState > 0 {
		if got := len(result.Commit.ContextState.DelegatedState); got != expected.DelegatedState {
			t.Errorf("commit delegated state = %d, want %d", got, expected.DelegatedState)
		}
	}

	if expected.EventKinds != nil {
		if !equalStrings(durability.EventKinds(), expected.EventKinds) {
			t.Errorf("event kinds =\n  %v\nwant\n  %v", durability.EventKinds(), expected.EventKinds)
		}
	}

	// Every event the engine returned must also have reached the durability
	// port, or the durable record is not the record the caller was shown.
	if len(result.Events) != len(durability.Events()) {
		t.Errorf("engine returned %d events but persisted %d", len(result.Events), len(durability.Events()))
	}

	// A denied tool must still come back to the model with something, or the
	// provider rejects the next turn for an unanswered tool call.
	for _, toolResult := range result.ToolResults {
		if toolResult.Outcome == model.ModelToolOutcomeFailed && toolResult.Output == nil {
			t.Errorf("failed tool %q produced no model-visible output", toolResult.Name)
		}
	}
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for index := range got {
		if got[index] != want[index] {
			return false
		}
	}
	return true
}

func equalInt32s(got, want []int32) bool {
	if len(got) != len(want) {
		return false
	}
	for index := range got {
		if got[index] != want[index] {
			return false
		}
	}
	return true
}
