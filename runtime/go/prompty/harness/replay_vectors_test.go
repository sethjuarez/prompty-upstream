package harness_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	harness "prompty/harness"
	model "prompty/model"
)

// specRoot walks up from the test's working directory until it finds the
// repository's spec/ directory. Tests run from runtime/go/prompty/harness, but
// the module may be vendored or relocated, so the path is discovered rather
// than hard-coded.
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
			t.Skipf("shared spec/ directory not found above %s; skipping vector tests", start)
			return ""
		}
		dir = parent
	}
}

func readVectorFile(t *testing.T, relative string, target interface{}) {
	t.Helper()
	path := filepath.Join(specRoot(t), "vectors", filepath.FromSlash(relative))
	raw, err := os.ReadFile(path) // #nosec G304 -- path is derived from the repo layout, not user input.
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(raw, target); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}

// replayVectorFile is the shared harness replay vector file.
type replayVectorFile struct {
	Version     int              `json:"version"`
	Description string           `json:"description"`
	Clock       string           `json:"clock"`
	SessionId   string           `json:"sessionId"`
	TurnId      string           `json:"turnId"`
	Scenarios   []replayScenario `json:"scenarios"`
}

type replayScenario struct {
	Name          string                 `json:"name"`
	Inputs        map[string]interface{} `json:"inputs"`
	MaxIterations *int32                 `json:"maxIterations"`
	Expected      []string               `json:"expected"`
}

// TestHarnessReplayVectors drives every scenario in the shared harness replay
// vectors through the real TurnRunner, then reads the journal back off disk and
// compares its normalized projection to the recorded sequence.
//
// The journal is round-tripped through a file rather than asserted from an
// in-memory buffer on purpose: the vectors are a durability contract, and a
// runner whose events are correct but whose journal is not would otherwise pass.
func TestHarnessReplayVectors(t *testing.T) {
	var file replayVectorFile
	readVectorFile(t, "harness/replay_vectors.json", &file)

	if len(file.Scenarios) == 0 {
		t.Fatal("no harness replay scenarios found")
	}

	executed := 0
	for _, scenario := range file.Scenarios {
		scenario := scenario
		executed++
		t.Run(scenario.Name, func(t *testing.T) {
			runReplayScenario(t, file, scenario)
		})
	}
	t.Logf("harness replay vectors: %d executed, %d total", executed, len(file.Scenarios))
}

func runReplayScenario(t *testing.T, file replayVectorFile, scenario replayScenario) {
	t.Helper()

	journalPath := filepath.Join(t.TempDir(), "session.jsonl")
	journal, err := harness.NewJSONLJournalWriter(journalPath)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}

	sink := &harness.CollectingEventSink{}
	checkpoints := harness.NewInMemoryCheckpointStore()

	runner := &harness.TurnRunner{
		EventSink:       sink,
		Journal:         journal,
		CheckpointStore: checkpoints,
		PermissionResolver: harness.FuncPermissionResolver{
			Resolve: scenarioResolver(scenario.Name),
		},
		HostToolExecutor: harness.FunctionHostToolExecutor{
			Handlers: map[string]harness.HostToolHandler{
				"add": func(args map[string]interface{}, _ model.HostToolRequest) (interface{}, error) {
					return numberOf(args["a"]) + numberOf(args["b"]), nil
				},
				"fail": func(map[string]interface{}, model.HostToolRequest) (interface{}, error) {
					return nil, fmt.Errorf("boom")
				},
			},
		},
		InvokeModel: scenarioModel(scenario.Name),
		Now:         harness.FixedClock(file.Clock),
		NextId:      harness.SequentialIDs(),
	}

	request := model.RunTurnRequest{
		SessionId: file.SessionId,
		TurnId:    file.TurnId,
		Inputs:    scenario.Inputs,
	}
	if scenario.MaxIterations != nil {
		request.Options = &model.TurnOptions{MaxIterations: scenario.MaxIterations}
	}

	result, err := runner.Run(request)
	if err != nil {
		t.Fatalf("run turn: %v", err)
	}

	verification, defects, err := harness.VerifyJournalFile(journalPath, scenario.Expected)
	if err != nil {
		t.Fatalf("verify journal: %v", err)
	}
	if len(defects) != 0 {
		t.Fatalf("journal has defects: %v", defects)
	}
	if verification.Status != model.ReplayVerificationStatusPassed {
		for _, mismatch := range verification.Mismatches {
			t.Errorf("journal[%d]: %s", mismatch.Index, mismatch.Message)
		}
		t.Fatalf("journal replay failed: %d expected records, %d actual",
			verification.ExpectedCount, verification.ActualCount)
	}

	// The live sink must observe exactly the events the journal recorded: a
	// runner whose durable log and whose observers disagree is worse than one
	// that fails, because neither view can be trusted afterwards.
	if got, want := len(sink.TurnEvents())+len(sink.SessionEvents()), len(scenario.Expected)-1; got != want {
		t.Errorf("sink saw %d events, journal recorded %d (excluding the summary)", got, want)
	}

	assertResultMatchesJournal(t, scenario, result, checkpoints, file.SessionId)
}

// assertResultMatchesJournal checks the returned result against the same facts
// the journal asserts, so a runner cannot satisfy the vectors by writing a
// correct journal while returning something else to its caller.
func assertResultMatchesJournal(
	t *testing.T,
	scenario replayScenario,
	result model.RunTurnResult,
	checkpoints *harness.InMemoryCheckpointStore,
	sessionId string,
) {
	t.Helper()

	summaryKey := scenario.Expected[len(scenario.Expected)-1]
	parts := splitKey(summaryKey)
	if len(parts) != 5 || parts[0] != "summary" {
		t.Fatalf("summary key %q is not summary:<session>:<status>:turns=N:checkpoints=N", summaryKey)
	}
	wantSessionId, wantStatus := parts[1], parts[2]
	var wantTurns, wantCheckpoints int32
	if _, err := fmt.Sscanf(parts[3], "turns=%d", &wantTurns); err != nil {
		t.Fatalf("summary turns %q: %v", parts[3], err)
	}
	if _, err := fmt.Sscanf(parts[4], "checkpoints=%d", &wantCheckpoints); err != nil {
		t.Fatalf("summary checkpoints %q: %v", parts[4], err)
	}

	if result.SessionId != wantSessionId {
		t.Errorf("result sessionId = %q, want %q", result.SessionId, wantSessionId)
	}
	if string(result.Status) != wantStatus {
		t.Errorf("result status = %q, want %q", result.Status, wantStatus)
	}
	if wantTurns != 1 {
		t.Fatalf("harness vectors assume a single turn, got turns=%d", wantTurns)
	}
	if int32(len(result.Checkpoints)) != wantCheckpoints {
		t.Errorf("result checkpoints = %d, want %d", len(result.Checkpoints), wantCheckpoints)
	}

	stored, err := checkpoints.ListCheckpoints(sessionId)
	if err != nil {
		t.Fatalf("list checkpoints: %v", err)
	}
	if int32(len(stored)) != wantCheckpoints {
		t.Errorf("stored checkpoints = %d, want %d", len(stored), wantCheckpoints)
	}
	for index, checkpoint := range stored {
		if checkpoint.CheckpointNumber == nil || *checkpoint.CheckpointNumber != int32(index+1) {
			t.Errorf("checkpoint %d has number %v, want %d", index, checkpoint.CheckpointNumber, index+1)
		}
	}
}

func splitKey(key string) []string {
	var (
		parts   []string
		current []rune
	)
	for _, r := range key {
		if r == ':' {
			parts = append(parts, string(current))
			current = nil
			continue
		}
		current = append(current, r)
	}
	return append(parts, string(current))
}

// scenarioResolver mirrors the cross-runtime scenario setup: every scenario
// approves except permission_denied.
func scenarioResolver(name string) harness.PermissionFunc {
	approved := name != "permission_denied"
	return func(request model.PermissionRequest) (model.PermissionDecision, error) {
		reason := "allow_all"
		if !approved {
			reason = "deny_all"
		}
		return model.PermissionDecision{
			RequestId:  request.RequestId,
			ToolCallId: request.ToolCallId,
			Permission: request.Permission,
			Approved:   approved,
			Reason:     harness.String(reason),
		}, nil
	}
}

// scenarioModel mirrors the cross-runtime scripted model: no_tool answers
// immediately, every other scenario requests one tool on iteration 0 and
// answers with the tool's result afterwards.
func scenarioModel(name string) harness.TurnModelFunc {
	toolName := "add"
	if name == "tool_failure" {
		toolName = "fail"
	}

	return func(request model.TurnModelRequest) (model.TurnModelResponse, error) {
		if name == "no_tool" {
			text := fmt.Sprintf("hello %v", request.Inputs["name"])
			return model.TurnModelResponse{
				Output:          harness.Value(map[string]interface{}{"text": text}),
				CheckpointState: map[string]interface{}{"stable": true},
			}, nil
		}

		if request.Iteration == 0 {
			return model.TurnModelResponse{
				ToolRequests: []model.HostToolRequest{{
					RequestId:  harness.String("exec-1"),
					ToolCallId: harness.String("call-1"),
					ToolName:   toolName,
					Arguments:  map[string]interface{}{"a": 2, "b": 3},
				}},
			}, nil
		}

		output := map[string]interface{}{}
		if len(request.ToolResults) > 0 {
			first := request.ToolResults[0]
			if first.Result != nil {
				output["toolResult"] = *first.Result
			}
			if first.ErrorKind != nil {
				output["errorKind"] = *first.ErrorKind
			}
		}
		return model.TurnModelResponse{Output: harness.Value(output)}, nil
	}
}

func numberOf(value interface{}) float64 {
	switch typed := value.(type) {
	case int:
		return float64(typed)
	case int32:
		return float64(typed)
	case int64:
		return float64(typed)
	case float64:
		return typed
	default:
		return 0
	}
}
