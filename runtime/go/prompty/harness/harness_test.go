package harness_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	harness "prompty/harness"
	model "prompty/model"
)

// ---------------------------------------------------------------------------
// Journal durability
// ---------------------------------------------------------------------------

func TestJournalRoundTripsEveryRecordKind(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session.jsonl")
	writer, err := harness.NewJSONLJournalWriter(path)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}

	if _, err := writer.AppendSession(sessionEvent(model.SessionEventTypeSessionStart, nil)); err != nil {
		t.Fatalf("append session: %v", err)
	}
	if _, err := writer.AppendTurn(turnEvent(model.TurnEventTypeTurnStart, 0, nil)); err != nil {
		t.Fatalf("append turn: %v", err)
	}
	status := model.SessionSummaryStatusSuccess
	if _, err := writer.Close(&model.SessionSummary{
		SessionId:   "session-1",
		Status:      &status,
		Turns:       harness.Int32(1),
		Checkpoints: harness.Int32(0),
	}); err != nil {
		t.Fatalf("close journal: %v", err)
	}

	records, defects, err := harness.ReadJournal(path)
	if err != nil {
		t.Fatalf("read journal: %v", err)
	}
	if len(defects) != 0 {
		t.Fatalf("unexpected defects: %v", defects)
	}
	if len(records) != 3 {
		t.Fatalf("read %d records, want 3", len(records))
	}
	if records[0].Session == nil || records[1].Turn == nil || records[2].Summary == nil {
		t.Fatalf("records did not decode into their typed forms: %#v", records)
	}
}

func TestJournalCloseIsIdempotentAndRejectsLateAppends(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session.jsonl")
	writer, err := harness.NewJSONLJournalWriter(path)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	if _, err := writer.Close(nil); err != nil {
		t.Fatalf("first close: %v", err)
	}
	// A deferred Close after an explicit one is idiomatic Go and must not be
	// an error.
	if _, err := writer.Close(nil); err != nil {
		t.Fatalf("second close: %v", err)
	}

	ok, err := writer.AppendTurn(turnEvent(model.TurnEventTypeTurnEnd, 0, nil))
	if ok {
		t.Error("a closed journal must refuse further records")
	}
	if !errors.Is(err, harness.ErrJournalClosed) {
		t.Errorf("err = %v, want ErrJournalClosed", err)
	}
}

func TestJournalReportsTornTrailingLineWithoutLosingThePrefix(t *testing.T) {
	// This is exactly what a process killed mid-append leaves behind: complete
	// records, then a fragment with no terminating newline. Everything before
	// the fragment is durable and must still be readable.
	path := filepath.Join(t.TempDir(), "session.jsonl")
	content := strings.Join([]string{
		`{"kind":"session","event":{"id":"a","type":"session_start","timestamp":"t","sessionId":"s","turnId":"u","payload":{}}}`,
		`{"kind":"turn","event":{"id":"b","type":"turn_start","timestamp":"t","turnId":"u","iteration":0,"payload":{}}}`,
		`{"kind":"turn","event":{"id":"c","type":"llm_st`,
	}, "\n")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write journal: %v", err)
	}

	records, defects, err := harness.ReadJournal(path)
	if err != nil {
		t.Fatalf("read journal: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("read %d records, want the 2 durable ones", len(records))
	}
	if len(defects) != 1 {
		t.Fatalf("defects = %v, want exactly one partial record", defects)
	}
	if !defects[0].Partial {
		t.Errorf("defect = %v, want it flagged as a partial write", defects[0])
	}
	if defects[0].Line != 3 {
		t.Errorf("defect line = %d, want 3", defects[0].Line)
	}
	if !strings.Contains(defects[0].String(), "partial record") {
		t.Errorf("defect message = %q, want it to name the partial record", defects[0].String())
	}
}

func TestJournalReportsCorruptRecordAndKeepsReading(t *testing.T) {
	// A corrupt record in the middle must not cost the reader the records
	// after it: a journal is forensic evidence, and refusing the whole file on
	// one bad line destroys the thing it was meant to recover.
	path := filepath.Join(t.TempDir(), "session.jsonl")
	content := strings.Join([]string{
		`{"kind":"turn","event":{"id":"a","type":"turn_start","timestamp":"t","turnId":"u","iteration":0,"payload":{}}}`,
		`{"kind":"turn","event":` + "\x00" + `broken}`,
		`{"kind":"turn","event":{"id":"c","type":"turn_end","timestamp":"t","turnId":"u","iteration":1,"payload":{"status":"success"}}}`,
		``,
	}, "\n")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write journal: %v", err)
	}

	records, defects, err := harness.ReadJournal(path)
	if err != nil {
		t.Fatalf("read journal: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("read %d records, want the 2 readable ones", len(records))
	}
	if len(defects) != 1 || defects[0].Line != 2 || defects[0].Partial {
		t.Fatalf("defects = %v, want one non-partial defect on line 2", defects)
	}
}

func TestJournalReportsTypeMismatchAndKeepsReading(t *testing.T) {
	cases := []struct {
		name      string
		malformed string
		label     string
	}{
		{"turn", `{"kind":"turn","event":{"id":123,"type":"turn_start","timestamp":"t"}}`, "invalid turn event shape"},
		{"session", `{"kind":"session","event":{"id":123,"type":"session_start","timestamp":"t"}}`, "invalid session event shape"},
		{"summary", `{"kind":"summary","summary":{"sessionId":123}}`, "invalid session summary shape"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			content := strings.Join([]string{
				tc.malformed,
				`{"kind":"turn","event":{"id":"valid","type":"turn_end","timestamp":"t","turnId":"u","iteration":1,"payload":{"status":"success"}}}`,
				``,
			}, "\n")

			records, defects, err := harness.ReadJournalFrom(strings.NewReader(content))
			if err != nil {
				t.Fatalf("ReadJournalFrom: %v", err)
			}
			if len(records) != 1 || records[0].Turn == nil || records[0].Turn.Id != "valid" {
				t.Fatalf("records = %#v, want the valid record after the defect", records)
			}
			if len(defects) != 1 || defects[0].Line != 1 || !strings.Contains(defects[0].Err.Error(), tc.label) {
				t.Fatalf("defects = %#v, want %q on line 1", defects, tc.label)
			}
		})
	}
}

func TestJournalRejectsUnknownRecordKind(t *testing.T) {
	// An unknown kind is not silently dropped: it means the writer and the
	// reader disagree about the format, which a caller must be told about.
	path := filepath.Join(t.TempDir(), "session.jsonl")
	if err := os.WriteFile(path, []byte("{\"kind\":\"telemetry\",\"event\":{}}\n"), 0o600); err != nil {
		t.Fatalf("write journal: %v", err)
	}
	records, defects, err := harness.ReadJournal(path)
	if err != nil {
		t.Fatalf("read journal: %v", err)
	}
	if len(records) != 0 || len(defects) != 1 {
		t.Fatalf("records=%d defects=%v, want 0 records and 1 defect", len(records), defects)
	}
}

func TestJournalReadFailureIsReportedNotSpunOn(t *testing.T) {
	// A reader that keeps failing must produce an error, not an endless stream
	// of defects: skipping past an unconsumed read error loops forever.
	_, _, err := harness.ReadJournalFrom(failingReader{err: errors.New("disk fell over")})
	if err == nil {
		t.Fatal("a persistent read failure must be reported")
	}
	if !strings.Contains(err.Error(), "disk fell over") {
		t.Errorf("err = %v, want the underlying read failure", err)
	}
}

// failingReader always fails, and never advances, which is exactly the shape
// that would spin a reader that treats every error as a skippable defect.
type failingReader struct{ err error }

func (r failingReader) Read([]byte) (int, error) { return 0, r.err }

func TestJournalReadsAnEmptyFileCleanly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.jsonl")
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	records, defects, err := harness.ReadJournal(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(records) != 0 || len(defects) != 0 {
		t.Errorf("empty journal produced %d records and %d defects", len(records), len(defects))
	}
}

func TestJournalRequiresAPath(t *testing.T) {
	if _, err := harness.NewJSONLJournalWriter("   "); err == nil {
		t.Fatal("an empty journal path must be rejected rather than writing somewhere arbitrary")
	}
}

func TestJournalCreatesParentDirectories(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "deeper", "session.jsonl")
	writer, err := harness.NewJSONLJournalWriter(path)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	if _, err := writer.Close(nil); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("journal was not created at %s: %v", path, err)
	}
}

func TestJournalConcurrentAppendsStayWholeAndOrdered(t *testing.T) {
	// Records written from many goroutines must each land as one complete
	// line. A record split by another goroutine's write is unrecoverable.
	path := filepath.Join(t.TempDir(), "session.jsonl")
	writer, err := harness.NewJSONLJournalWriter(path)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	writer.Sync = false // the test asserts framing, not fsync durability

	const writers, perWriter = 8, 25
	var wait sync.WaitGroup
	for w := 0; w < writers; w++ {
		wait.Add(1)
		go func(id int) {
			defer wait.Done()
			for i := 0; i < perWriter; i++ {
				payload := map[string]interface{}{
					"writer": id,
					"index":  i,
					"filler": strings.Repeat("x", 400),
				}
				if _, err := writer.AppendTurn(turnEvent(model.TurnEventTypeStatus, int32(i), payload)); err != nil {
					t.Errorf("append: %v", err)
					return
				}
			}
		}(w)
	}
	wait.Wait()
	if _, err := writer.Close(nil); err != nil {
		t.Fatalf("close: %v", err)
	}

	records, defects, err := harness.ReadJournal(path)
	if err != nil {
		t.Fatalf("read journal: %v", err)
	}
	if len(defects) != 0 {
		t.Fatalf("concurrent appends produced defects: %v", defects)
	}
	if len(records) != writers*perWriter {
		t.Fatalf("read %d records, want %d", len(records), writers*perWriter)
	}
}

// ---------------------------------------------------------------------------
// Checkpoint store
// ---------------------------------------------------------------------------

func TestCheckpointStoreIsolatesSessions(t *testing.T) {
	// Hosts number checkpoints per turn, so identical ids across sessions are
	// the normal case, not an edge one.
	store := harness.NewInMemoryCheckpointStore()
	for _, session := range []string{"session-a", "session-b"} {
		if _, err := store.Save(model.Checkpoint{
			Id:               harness.String("turn-1-checkpoint-0"),
			SessionId:        harness.String(session),
			CheckpointNumber: harness.Int32(1),
			Title:            session,
		}); err != nil {
			t.Fatalf("save %s: %v", session, err)
		}
	}

	loaded, err := store.Load("session-a", "turn-1-checkpoint-0")
	if err != nil || loaded == nil {
		t.Fatalf("load session-a: %v %v", loaded, err)
	}
	if loaded.Title != "session-a" {
		t.Errorf("session-a loaded %q — the sessions collided", loaded.Title)
	}

	listed, err := store.ListCheckpoints("session-a")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(listed) != 1 {
		t.Errorf("session-a listed %d checkpoints, want 1", len(listed))
	}
}

func TestCheckpointStoreMissingCheckpointIsNotAnError(t *testing.T) {
	store := harness.NewInMemoryCheckpointStore()
	loaded, err := store.Load("nobody", "nothing")
	if err != nil {
		t.Fatalf("loading an absent checkpoint must not error: %v", err)
	}
	if loaded != nil {
		t.Errorf("loaded = %#v, want nil", loaded)
	}
}

func TestCheckpointStoreRequiresIdentifiers(t *testing.T) {
	store := harness.NewInMemoryCheckpointStore()
	if _, err := store.Save(model.Checkpoint{Id: harness.String("c1")}); err == nil {
		t.Error("a checkpoint with no session id must be rejected — it could never be loaded back")
	}
	if _, err := store.Save(model.Checkpoint{SessionId: harness.String("s1")}); err == nil {
		t.Error("a checkpoint with no id must be rejected")
	}
}

func TestCheckpointStoreOrdersByNumberNotIdentifier(t *testing.T) {
	// "turn-1-checkpoint-10" sorts before "...-2" lexicographically, which
	// would present a resumable history out of order.
	store := harness.NewInMemoryCheckpointStore()
	for _, number := range []int32{10, 2, 1} {
		if _, err := store.Save(model.Checkpoint{
			Id:               harness.String(fmt.Sprintf("turn-1-checkpoint-%d", number)),
			SessionId:        harness.String("session-1"),
			CheckpointNumber: harness.Int32(number),
		}); err != nil {
			t.Fatalf("save: %v", err)
		}
	}

	listed, err := store.ListCheckpoints("session-1")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	got := make([]int32, 0, len(listed))
	for _, checkpoint := range listed {
		got = append(got, *checkpoint.CheckpointNumber)
	}
	if fmt.Sprint(got) != "[1 2 10]" {
		t.Errorf("checkpoint order = %v, want [1 2 10]", got)
	}

	latest, err := store.Latest("session-1")
	if err != nil || latest == nil {
		t.Fatalf("latest: %v %v", latest, err)
	}
	if *latest.CheckpointNumber != 10 {
		t.Errorf("latest = %d, want 10", *latest.CheckpointNumber)
	}
}

func TestCheckpointStoreDoesNotShareMutableState(t *testing.T) {
	store := harness.NewInMemoryCheckpointStore()
	original := model.Checkpoint{
		Id:        harness.String("c1"),
		SessionId: harness.String("s1"),
		State:     map[string]interface{}{"iteration": 0},
	}
	if _, err := store.Save(original); err != nil {
		t.Fatalf("save: %v", err)
	}
	// Mutating the caller's map must not reach into the store.
	original.State["iteration"] = 99

	loaded, err := store.Load("s1", "c1")
	if err != nil || loaded == nil {
		t.Fatalf("load: %v %v", loaded, err)
	}
	if loaded.State["iteration"] != 0 {
		t.Errorf("stored state was mutated through the caller's map: %v", loaded.State)
	}

	loaded.State["iteration"] = 42
	reloaded, _ := store.Load("s1", "c1")
	if reloaded.State["iteration"] != 0 {
		t.Errorf("stored state was mutated through a loaded copy: %v", reloaded.State)
	}
}

func TestCheckpointStoreIsConcurrencySafe(t *testing.T) {
	store := harness.NewInMemoryCheckpointStore()
	const goroutines = 16
	var wait sync.WaitGroup
	for index := 0; index < goroutines; index++ {
		wait.Add(1)
		go func(number int32) {
			defer wait.Done()
			_, _ = store.Save(model.Checkpoint{
				Id:               harness.String(fmt.Sprintf("c%d", number)),
				SessionId:        harness.String("s1"),
				CheckpointNumber: harness.Int32(number),
			})
			_, _ = store.ListCheckpoints("s1")
			_, _ = store.Load("s1", "c0")
		}(int32(index))
	}
	wait.Wait()

	listed, err := store.ListCheckpoints("s1")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(listed) != goroutines {
		t.Errorf("stored %d checkpoints, want %d", len(listed), goroutines)
	}
}

// ---------------------------------------------------------------------------
// Host tool executor and permission resolvers
// ---------------------------------------------------------------------------

func TestFunctionHostToolExecutorReportsMissingTool(t *testing.T) {
	executor := harness.FunctionHostToolExecutor{}
	result, err := executor.Execute(model.HostToolRequest{ToolName: "nope"})
	if err != nil {
		t.Fatalf("a missing tool must be a failed result, not an error: %v", err)
	}
	if result.Success {
		t.Error("a missing tool must not report success")
	}
	if result.ErrorKind == nil || *result.ErrorKind != harness.ErrorKindNotFound {
		t.Errorf("errorKind = %v, want %q", result.ErrorKind, harness.ErrorKindNotFound)
	}
	if result.Result == nil {
		t.Error("a missing tool must still produce a model-visible message")
	}
}

func TestFunctionHostToolExecutorConvertsHandlerFailure(t *testing.T) {
	executor := harness.FunctionHostToolExecutor{
		Handlers: map[string]harness.HostToolHandler{
			"boom": func(map[string]interface{}, model.HostToolRequest) (interface{}, error) {
				return nil, errors.New("disk on fire")
			},
		},
	}
	result, err := executor.Execute(model.HostToolRequest{ToolName: "boom"})
	if err != nil {
		t.Fatalf("a failing tool must not abort the turn: %v", err)
	}
	if result.ErrorKind == nil || *result.ErrorKind != harness.ErrorKindException {
		t.Errorf("errorKind = %v, want %q", result.ErrorKind, harness.ErrorKindException)
	}
	payload, _ := (*result.Result).(map[string]interface{})
	if payload["message"] != "disk on fire" {
		t.Errorf("result = %v, want the handler's message", payload)
	}
}

func TestFunctionHostToolExecutorConvertsHandlerPanic(t *testing.T) {
	executor := harness.FunctionHostToolExecutor{
		Handlers: map[string]harness.HostToolHandler{
			"boom": func(map[string]interface{}, model.HostToolRequest) (interface{}, error) {
				panic("broken tool")
			},
		},
	}
	result, err := executor.Execute(model.HostToolRequest{ToolName: "boom"})
	if err != nil {
		t.Fatalf("a panicking tool must be a failed result, not an error: %v", err)
	}
	if result.Success {
		t.Fatal("a panicking tool reported success")
	}
	if result.ErrorKind == nil || *result.ErrorKind != harness.ErrorKindException {
		t.Fatalf("error kind = %v, want %q", result.ErrorKind, harness.ErrorKindException)
	}
	if result.Result == nil || !strings.Contains(fmt.Sprint(*result.Result), "tool panicked: broken tool") {
		t.Fatalf("result = %v, want panic details", result.Result)
	}
}

func TestFunctionHostToolExecutorPassesEmptyArguments(t *testing.T) {
	var seen map[string]interface{}
	executor := harness.FunctionHostToolExecutor{
		Handlers: map[string]harness.HostToolHandler{
			"noop": func(args map[string]interface{}, _ model.HostToolRequest) (interface{}, error) {
				seen = args
				return "ok", nil
			},
		},
	}
	if _, err := executor.Execute(model.HostToolRequest{ToolName: "noop"}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if seen == nil {
		t.Error("a handler must never receive a nil argument map")
	}
}

func TestPermissionResolversEchoCorrelationFields(t *testing.T) {
	request := model.PermissionRequest{
		RequestId:  harness.String("req-1"),
		ToolCallId: harness.String("call-1"),
		Permission: "tool.execute",
		Target:     harness.String("write_file"),
	}

	for name, resolver := range map[string]model.PermissionResolver{
		"allow": harness.AllowAllPermissionResolver{},
		"deny":  harness.DenyAllPermissionResolver{},
	} {
		decision, err := resolver.Request(request)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if decision.RequestId == nil || *decision.RequestId != "req-1" {
			t.Errorf("%s: decision lost the request id — it cannot be matched back to the call", name)
		}
		if decision.ToolCallId == nil || *decision.ToolCallId != "call-1" {
			t.Errorf("%s: decision lost the tool call id", name)
		}
		if decision.Permission != "tool.execute" {
			t.Errorf("%s: permission = %q, want tool.execute", name, decision.Permission)
		}
	}
}

func TestAllowListPermissionResolver(t *testing.T) {
	resolver := harness.AllowListPermissionResolver{Allowed: map[string]bool{"read_file": true}}

	allowed, _ := resolver.Request(model.PermissionRequest{Target: harness.String("read_file")})
	if !allowed.Approved {
		t.Error("a listed target must be approved")
	}
	refused, _ := resolver.Request(model.PermissionRequest{Target: harness.String("delete_everything")})
	if refused.Approved {
		t.Error("an unlisted target must be refused")
	}
	if refused.Reason == nil || !strings.Contains(*refused.Reason, "delete_everything") {
		t.Errorf("refusal reason = %v, want it to name the target", refused.Reason)
	}
	// An absent target is not on any list and must be refused, not defaulted in.
	if missing, _ := resolver.Request(model.PermissionRequest{}); missing.Approved {
		t.Error("a request with no target must be refused")
	}
}

// ---------------------------------------------------------------------------
// Runner behaviour
// ---------------------------------------------------------------------------

func TestRunnerCancellationFromACollaboratorIsNotMisreadAsMaxIterations(t *testing.T) {
	// A permission resolver or tool executor that honours the context returns
	// a wrapped cancellation. If the runner only trusted its own context
	// checks, the journal would blame the iteration bound for an abandoned
	// turn — which is exactly the wrong thing to tell an operator.
	journal := &harness.MemoryJournalWriter{}
	ctx, cancel := context.WithCancel(context.Background())

	runner := &harness.TurnRunner{
		EventSink:       &harness.CollectingEventSink{},
		Journal:         journal,
		CheckpointStore: harness.NewInMemoryCheckpointStore(),
		PermissionResolver: harness.FuncPermissionResolver{
			Resolve: func(model.PermissionRequest) (model.PermissionDecision, error) {
				cancel()
				return model.PermissionDecision{}, context.Canceled
			},
		},
		HostToolExecutor: harness.FunctionHostToolExecutor{},
		InvokeModel: func(model.TurnModelRequest) (model.TurnModelResponse, error) {
			return model.TurnModelResponse{
				ToolRequests: []model.HostToolRequest{{RequestId: harness.String("exec-1"), ToolName: "anything"}},
			}, nil
		},
		Now:    harness.FixedClock("2026-06-28T00:00:00Z"),
		NextId: harness.SequentialIDs(),
	}
	// MaxIterations of 1 makes the misclassification maximally tempting: the
	// loop would otherwise be at its bound.
	one := int32(1)
	result, err := runner.RunContext(ctx, model.RunTurnRequest{
		SessionId: "session-1",
		TurnId:    "turn-1",
		Options:   &model.TurnOptions{MaxIterations: &one},
	})

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if result.Status != model.RunTurnStatusCancelled {
		t.Errorf("status = %q, want cancelled", result.Status)
	}

	normalized, normErr := harness.NormalizeRecords(journal.Records())
	if normErr != nil {
		t.Fatalf("normalize: %v", normErr)
	}
	keys := harness.ReplayKeys(normalized)
	for _, key := range keys {
		if strings.Contains(key, "max_iterations") {
			t.Errorf("journal %v blames max_iterations for a cancelled turn", keys)
		}
	}
	if !containsKey(keys, "session:session_end:session-1:turn-1:cancelled") {
		t.Errorf("journal %v is missing a cancelled session_end", keys)
	}
}

func TestRunnerClosesTheJournalOnAnEarlyFailure(t *testing.T) {
	// The caller has no reference to the journal, so a runner that abandons a
	// turn before finalize would leak the file handle forever.
	journal := &closeTrackingJournal{}
	runner := &harness.TurnRunner{
		EventSink:       harness.FuncEventSink{OnSession: func(model.SessionEvent) error { return errors.New("sink offline") }},
		Journal:         journal,
		CheckpointStore: harness.NewInMemoryCheckpointStore(),
		InvokeModel: func(model.TurnModelRequest) (model.TurnModelResponse, error) {
			return model.TurnModelResponse{}, nil
		},
	}

	if _, err := runner.Run(model.RunTurnRequest{SessionId: "s", TurnId: "t"}); err == nil {
		t.Fatal("expected the sink failure to abort the turn")
	}
	if journal.closes == 0 {
		t.Error("the journal was never closed on the failure path")
	}
}

// closeTrackingJournal counts Close calls so a leak is observable.
type closeTrackingJournal struct {
	harness.MemoryJournalWriter
	closes int
}

func (j *closeTrackingJournal) Close(summary *model.SessionSummary) (bool, error) {
	j.closes++
	return j.MemoryJournalWriter.Close(summary)
}

func TestJournalBoundsAnEndlessRecordInsteadOfExhaustingMemory(t *testing.T) {
	// A stream that never produces a newline is the shape that would make a
	// read-the-whole-line-then-check implementation allocate without limit,
	// and a scan-until-terminator resynchronisation spin forever. It must
	// terminate with an error instead.
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _, err := harness.ReadJournalFrom(endlessReader{})
		if err == nil {
			t.Error("an endless record must fail the read rather than be tolerated")
		}
	}()

	select {
	case <-done:
	case <-time.After(60 * time.Second):
		t.Fatal("reading an endless record did not terminate")
	}
}

// endlessReader emits non-newline bytes forever.
type endlessReader struct{}

func (endlessReader) Read(p []byte) (int, error) {
	for index := range p {
		p[index] = 'x'
	}
	return len(p), nil
}

func TestJournalCapsAnOversizedRecordWithoutLosingTheRest(t *testing.T) {
	// The cap must apply while the record is being consumed, not after it has
	// already been allocated, and the reader must resynchronise on the next
	// line rather than giving up on the file.
	path := filepath.Join(t.TempDir(), "session.jsonl")
	oversized := `{"kind":"turn","event":{"id":"big","payload":"` + strings.Repeat("x", 17<<20) + `"}}`
	good := `{"kind":"turn","event":{"id":"c","type":"turn_end","timestamp":"t","turnId":"u","iteration":1,"payload":{"status":"success"}}}`
	if err := os.WriteFile(path, []byte(oversized+"\n"+good+"\n"), 0o600); err != nil {
		t.Fatalf("write journal: %v", err)
	}

	records, defects, err := harness.ReadJournal(path)
	if err != nil {
		t.Fatalf("read journal: %v", err)
	}
	if len(defects) != 1 || defects[0].Line != 1 {
		t.Fatalf("defects = %v, want one on line 1", defects)
	}
	if len(records) != 1 {
		t.Fatalf("read %d records, want the 1 well-formed record after the oversized one", len(records))
	}
	// The retained diagnostic must be bounded too, or the defect report
	// becomes the memory problem.
	if len(defects[0].Content) > 1024 {
		t.Errorf("defect content is %d bytes, want it truncated for logging", len(defects[0].Content))
	}
}

func TestRunnerRejectsMissingCollaborators(t *testing.T) {
	runner := &harness.TurnRunner{}
	_, err := runner.Run(model.RunTurnRequest{SessionId: "s", TurnId: "t"})
	if !errors.Is(err, harness.ErrRunnerMisconfigured) {
		t.Fatalf("err = %v, want ErrRunnerMisconfigured", err)
	}
	for _, field := range []string{"EventSink", "Journal", "CheckpointStore", "InvokeModel"} {
		if !strings.Contains(err.Error(), field) {
			t.Errorf("error %q does not name the missing %s", err, field)
		}
	}
}

func TestRunnerIsDeterministicUnderAFixedClockAndIDs(t *testing.T) {
	// Two runs with the same fixed clock and id generator must produce
	// byte-identical journals; otherwise journal comparison is a test of the
	// clock rather than of the turn.
	first := runDeterministicTurn(t)
	second := runDeterministicTurn(t)
	if first != second {
		t.Errorf("two deterministic runs diverged:\n%s\nvs\n%s", first, second)
	}
	if !strings.Contains(first, `"timestamp":"2026-06-28T00:00:00Z"`) {
		t.Error("the fixed clock was not applied to journal records")
	}
	if !strings.Contains(first, `"id":"session-event-1"`) {
		t.Error("the sequential id generator was not applied to journal records")
	}
}

func runDeterministicTurn(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "session.jsonl")
	journal, err := harness.NewJSONLJournalWriter(path)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	runner := &harness.TurnRunner{
		EventSink:       &harness.CollectingEventSink{},
		Journal:         journal,
		CheckpointStore: harness.NewInMemoryCheckpointStore(),
		InvokeModel: func(model.TurnModelRequest) (model.TurnModelResponse, error) {
			return model.TurnModelResponse{Output: harness.Value("done")}, nil
		},
		Now:    harness.FixedClock("2026-06-28T00:00:00Z"),
		NextId: harness.SequentialIDs(),
	}
	if _, err := runner.Run(model.RunTurnRequest{SessionId: "session-1", TurnId: "turn-1"}); err != nil {
		t.Fatalf("run: %v", err)
	}
	raw, err := os.ReadFile(path) // #nosec G304 -- test-owned temp path.
	if err != nil {
		t.Fatalf("read journal: %v", err)
	}
	return string(raw)
}

func TestRunnerCancellationStillProducesATerminalRecord(t *testing.T) {
	// An abandoned session that leaves no terminal record is indistinguishable
	// from one that crashed.
	journal := &harness.MemoryJournalWriter{}
	sink := &harness.CollectingEventSink{}

	ctx, cancel := context.WithCancel(context.Background())
	runner := &harness.TurnRunner{
		EventSink:       sink,
		Journal:         journal,
		CheckpointStore: harness.NewInMemoryCheckpointStore(),
		InvokeModel: func(request model.TurnModelRequest) (model.TurnModelResponse, error) {
			// Cancel after the first round so the turn stops between iterations.
			cancel()
			return model.TurnModelResponse{
				ToolRequests: []model.HostToolRequest{{
					RequestId: harness.String("exec-1"),
					ToolName:  "noop",
				}},
			}, nil
		},
		PermissionResolver: harness.AllowAllPermissionResolver{},
		HostToolExecutor: harness.FunctionHostToolExecutor{
			Handlers: map[string]harness.HostToolHandler{
				"noop": func(map[string]interface{}, model.HostToolRequest) (interface{}, error) { return "ok", nil },
			},
		},
		Now:    harness.FixedClock("2026-06-28T00:00:00Z"),
		NextId: harness.SequentialIDs(),
	}

	result, err := runner.RunContext(ctx, model.RunTurnRequest{SessionId: "session-1", TurnId: "turn-1"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if result.Status != model.RunTurnStatusCancelled {
		t.Errorf("status = %q, want cancelled", result.Status)
	}

	records := journal.Records()
	normalized, err := harness.NormalizeRecords(records)
	if err != nil {
		t.Fatalf("normalize: %v", err)
	}
	keys := harness.ReplayKeys(normalized)
	last := keys[len(keys)-1]
	if !strings.HasPrefix(last, "summary:") || !strings.Contains(last, "cancelled") {
		t.Errorf("journal ends with %q, want a cancelled summary", last)
	}
	if !containsKey(keys, "turn:turn_end:1:cancelled") {
		t.Errorf("journal %v is missing a cancelled turn_end", keys)
	}
	if !containsKey(keys, "session:session_end:session-1:turn-1:cancelled") {
		t.Errorf("journal %v is missing a cancelled session_end", keys)
	}
}

func TestRunnerCancellationBeforeTheFirstModelCallCostsNothing(t *testing.T) {
	journal := &harness.MemoryJournalWriter{}
	called := false

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	runner := &harness.TurnRunner{
		EventSink:       &harness.CollectingEventSink{},
		Journal:         journal,
		CheckpointStore: harness.NewInMemoryCheckpointStore(),
		InvokeModel: func(model.TurnModelRequest) (model.TurnModelResponse, error) {
			called = true
			return model.TurnModelResponse{}, nil
		},
		Now:    harness.FixedClock("2026-06-28T00:00:00Z"),
		NextId: harness.SequentialIDs(),
	}

	result, err := runner.RunContext(ctx, model.RunTurnRequest{SessionId: "session-1", TurnId: "turn-1"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if called {
		t.Error("a turn cancelled before it started must not pay for a provider call")
	}
	if result.Iterations != 0 {
		t.Errorf("iterations = %d, want 0", result.Iterations)
	}
}

func TestRunnerSurfacesSinkFailuresRatherThanContinuing(t *testing.T) {
	// A partially recorded turn is not replayable. Continuing past a sink
	// failure would produce a journal describing a run that did not happen.
	failure := errors.New("sink offline")
	runner := &harness.TurnRunner{
		EventSink:       harness.FuncEventSink{OnSession: func(model.SessionEvent) error { return failure }},
		Journal:         &harness.MemoryJournalWriter{},
		CheckpointStore: harness.NewInMemoryCheckpointStore(),
		InvokeModel: func(model.TurnModelRequest) (model.TurnModelResponse, error) {
			return model.TurnModelResponse{}, nil
		},
	}
	if _, err := runner.Run(model.RunTurnRequest{SessionId: "s", TurnId: "t"}); !errors.Is(err, failure) {
		t.Fatalf("err = %v, want the sink failure", err)
	}
}

func TestRunnerSurfacesModelFailures(t *testing.T) {
	failure := errors.New("provider unreachable")
	runner := &harness.TurnRunner{
		EventSink:       &harness.CollectingEventSink{},
		Journal:         &harness.MemoryJournalWriter{},
		CheckpointStore: harness.NewInMemoryCheckpointStore(),
		InvokeModel: func(model.TurnModelRequest) (model.TurnModelResponse, error) {
			return model.TurnModelResponse{}, failure
		},
	}
	if _, err := runner.Run(model.RunTurnRequest{SessionId: "s", TurnId: "t"}); !errors.Is(err, failure) {
		t.Fatalf("err = %v, want the model failure", err)
	}
}

func TestRunnerGeneratesAPermissionIDWhenTheToolRequestHasNone(t *testing.T) {
	journal := &harness.MemoryJournalWriter{}
	runner := &harness.TurnRunner{
		EventSink:          &harness.CollectingEventSink{},
		Journal:            journal,
		CheckpointStore:    harness.NewInMemoryCheckpointStore(),
		PermissionResolver: harness.AllowAllPermissionResolver{},
		HostToolExecutor: harness.FunctionHostToolExecutor{
			Handlers: map[string]harness.HostToolHandler{
				"noop": func(map[string]interface{}, model.HostToolRequest) (interface{}, error) { return "ok", nil },
			},
		},
		InvokeModel: func(request model.TurnModelRequest) (model.TurnModelResponse, error) {
			if request.Iteration == 0 {
				return model.TurnModelResponse{
					ToolRequests: []model.HostToolRequest{{ToolName: "noop"}},
				}, nil
			}
			return model.TurnModelResponse{Output: harness.Value("done")}, nil
		},
		Now:    harness.FixedClock("2026-06-28T00:00:00Z"),
		NextId: harness.SequentialIDs(),
	}

	if _, err := runner.Run(model.RunTurnRequest{SessionId: "s", TurnId: "t"}); err != nil {
		t.Fatalf("run: %v", err)
	}

	normalized, err := harness.NormalizeRecords(journal.Records())
	if err != nil {
		t.Fatalf("normalize: %v", err)
	}
	for _, record := range normalized {
		if record.Type != nil && *record.Type == "permission_requested" {
			if record.RequestId == nil || !strings.HasPrefix(*record.RequestId, "permission-") {
				t.Errorf("generated permission id = %v, want a permission-* identifier", record.RequestId)
			}
			return
		}
	}
	t.Error("no permission_requested record was journalled")
}

func TestRunnerRecordsDeniedToolResult(t *testing.T) {
	// A denial must remain model-visible and replay-visible so a forensic reader
	// can distinguish "denied" from "never asked".
	journal := &harness.MemoryJournalWriter{}
	runner := &harness.TurnRunner{
		EventSink:          &harness.CollectingEventSink{},
		Journal:            journal,
		CheckpointStore:    harness.NewInMemoryCheckpointStore(),
		PermissionResolver: harness.DenyAllPermissionResolver{},
		HostToolExecutor: harness.FunctionHostToolExecutor{
			Handlers: map[string]harness.HostToolHandler{
				"danger": func(map[string]interface{}, model.HostToolRequest) (interface{}, error) {
					t.Error("a denied tool must never execute")
					return nil, nil
				},
			},
		},
		InvokeModel: func(request model.TurnModelRequest) (model.TurnModelResponse, error) {
			if request.Iteration == 0 {
				return model.TurnModelResponse{
					ToolRequests: []model.HostToolRequest{{
						RequestId: harness.String("exec-1"),
						ToolName:  "danger",
					}},
				}, nil
			}
			return model.TurnModelResponse{Output: harness.Value("understood")}, nil
		},
		Now:    harness.FixedClock("2026-06-28T00:00:00Z"),
		NextId: harness.SequentialIDs(),
	}

	if _, err := runner.Run(model.RunTurnRequest{SessionId: "s", TurnId: "t"}); err != nil {
		t.Fatalf("run: %v", err)
	}

	normalized, err := harness.NormalizeRecords(journal.Records())
	if err != nil {
		t.Fatalf("normalize: %v", err)
	}
	keys := harness.ReplayKeys(normalized)

	if !containsKey(keys, "turn:tool_result:0:danger:false:permission_denied") {
		t.Errorf("journal %v does not record the denied tool result", keys)
	}
	for _, key := range keys {
		if strings.HasPrefix(key, "turn:tool_execution_start") {
			t.Errorf("journal records an execution that never happened: %s", key)
		}
	}
}

func TestReplayVerifierReportsExtraAndMissingRecords(t *testing.T) {
	expected := []model.ReplayJournalRecord{
		{Kind: model.ReplayRecordKindTurn, Type: harness.String("turn_start"), Iteration: harness.Int32(0)},
		{Kind: model.ReplayRecordKindTurn, Type: harness.String("llm_start"), Iteration: harness.Int32(0)},
	}
	actual := []model.ReplayJournalRecord{
		{Kind: model.ReplayRecordKindTurn, Type: harness.String("turn_start"), Iteration: harness.Int32(0)},
	}

	result := harness.ReplayVerifier{}.Verify(model.ReplayVerificationRequest{Expected: expected, Actual: actual})
	if result.Status != model.ReplayVerificationStatusFailed {
		t.Fatalf("status = %q, want failed", result.Status)
	}
	if len(result.Mismatches) != 1 || result.Mismatches[0].Message != "Missing replay record" {
		t.Fatalf("mismatches = %#v, want one missing record", result.Mismatches)
	}
	if result.ExpectedCount != 2 || result.ActualCount != 1 {
		t.Errorf("counts = %d/%d, want 2/1", result.ExpectedCount, result.ActualCount)
	}

	swapped := harness.ReplayVerifier{}.Verify(model.ReplayVerificationRequest{Expected: actual, Actual: expected})
	if swapped.Mismatches[0].Message != "Unexpected extra replay record" {
		t.Errorf("message = %q, want the extra-record message", swapped.Mismatches[0].Message)
	}
}

func TestReplayVerifierIsOrderSensitive(t *testing.T) {
	// A replay journal is an ordered decision log: the same events in a
	// different order describe a different turn.
	first := model.ReplayJournalRecord{Kind: model.ReplayRecordKindTurn, Type: harness.String("llm_start")}
	second := model.ReplayJournalRecord{Kind: model.ReplayRecordKindTurn, Type: harness.String("llm_complete")}

	result := harness.ReplayVerifier{}.Verify(model.ReplayVerificationRequest{
		Expected: []model.ReplayJournalRecord{first, second},
		Actual:   []model.ReplayJournalRecord{second, first},
	})
	if result.Status != model.ReplayVerificationStatusFailed {
		t.Error("a reordered journal must not verify")
	}
}

func TestNormalizeRejectsAnUnknownStatus(t *testing.T) {
	// A status outside the replay enum must not be silently dropped: a
	// comparison built on quietly discarded fields passes for the wrong reason.
	record := harness.JournalRecord{
		Kind: "turn",
		Turn: &model.TurnEvent{
			Type:      model.TurnEventTypeTurnEnd,
			Iteration: harness.Int32(1),
			Payload:   map[string]interface{}{"status": "interrupted"},
		},
	}
	if _, err := harness.NormalizeRecord(record); err == nil {
		t.Fatal("an unmappable status must be reported, not dropped")
	}
}

// ---------------------------------------------------------------------------
// Memory and resume
// ---------------------------------------------------------------------------

func TestMemoryStoreRoundTripsThroughTheEmittedSnapshot(t *testing.T) {
	store := harness.NewInMemoryMemoryStore()
	store.AddText(model.MemoryCategoryCore, "The user prefers metric units", "units")
	store.AddText(model.MemoryCategoryArchival, "Discussed the Paris trip", "travel")

	snapshot := store.Snapshot()
	if len(snapshot.Entries) != 2 {
		t.Fatalf("snapshot has %d entries, want 2", len(snapshot.Entries))
	}
	for _, entry := range snapshot.Entries {
		if entry.CreatedAt == nil {
			t.Error("every entry must be stamped so it is orderable")
		}
	}

	restored := harness.NewInMemoryMemoryStore()
	restored.Restore(snapshot)
	if restored.Len() != 2 {
		t.Errorf("restored %d entries, want 2", restored.Len())
	}
	if len(restored.ByCategory(model.MemoryCategoryCore)) != 1 {
		t.Error("category filtering did not survive the round trip")
	}
	if len(restored.Search("METRIC")) != 1 {
		t.Error("search must be case-insensitive")
	}
	if len(restored.Search("travel")) != 1 {
		t.Error("search must match tags as well as content")
	}
	if len(restored.Search("   ")) != 0 {
		t.Error("a blank query must match nothing rather than everything")
	}
}

func TestMemoryStoreDoesNotShareMutableState(t *testing.T) {
	store := harness.NewInMemoryMemoryStore()
	store.AddText(model.MemoryCategoryCore, "original", "tag")

	entries := store.Entries()
	entries[0].Content = "mutated"
	entries[0].Tags[0] = "mutated"

	if store.Entries()[0].Content != "original" || store.Entries()[0].Tags[0] != "tag" {
		t.Error("the store was mutated through a returned entry")
	}
}

func TestLatestResumePointFindsTheNewestCheckpoint(t *testing.T) {
	store := harness.NewInMemoryCheckpointStore()
	if _, _, err := mustNoResume(store); err != nil {
		t.Fatal(err)
	}

	for _, number := range []int32{1, 2, 3} {
		if _, err := store.Save(model.Checkpoint{
			Id:               harness.String(fmt.Sprintf("turn-1-checkpoint-%d", number-1)),
			SessionId:        harness.String("session-1"),
			CheckpointNumber: harness.Int32(number),
			State:            map[string]interface{}{"iteration": int(number - 1), "output": "answer"},
		}); err != nil {
			t.Fatalf("save: %v", err)
		}
	}

	point, ok, err := harness.LatestResumePoint(store, "session-1")
	if err != nil || !ok {
		t.Fatalf("resume point: %v %v", ok, err)
	}
	if point.Iteration != 2 {
		t.Errorf("iteration = %d, want 2", point.Iteration)
	}
	if point.CompletedIterations != 3 {
		t.Errorf("completed iterations = %d, want 3", point.CompletedIterations)
	}
	if output, ok := harness.CheckpointOutput(point.Checkpoint); !ok || output != "answer" {
		t.Errorf("checkpoint output = %v/%t, want \"answer\"/true", output, ok)
	}
}

func mustNoResume(store model.CheckpointStore) (harness.ResumePoint, bool, error) {
	point, ok, err := harness.LatestResumePoint(store, "session-1")
	if err != nil {
		return point, ok, err
	}
	if ok {
		return point, ok, errors.New("an empty store must report no resume point")
	}
	return point, ok, nil
}

func TestReplayJournalUpToTruncatesAtACheckpoint(t *testing.T) {
	journal := &harness.MemoryJournalWriter{}
	runner := &harness.TurnRunner{
		EventSink:          &harness.CollectingEventSink{},
		Journal:            journal,
		CheckpointStore:    harness.NewInMemoryCheckpointStore(),
		PermissionResolver: harness.AllowAllPermissionResolver{},
		HostToolExecutor: harness.FunctionHostToolExecutor{
			Handlers: map[string]harness.HostToolHandler{
				"noop": func(map[string]interface{}, model.HostToolRequest) (interface{}, error) { return "ok", nil },
			},
		},
		InvokeModel: func(request model.TurnModelRequest) (model.TurnModelResponse, error) {
			if request.Iteration == 0 {
				return model.TurnModelResponse{
					ToolRequests: []model.HostToolRequest{{RequestId: harness.String("exec-1"), ToolName: "noop"}},
				}, nil
			}
			return model.TurnModelResponse{Output: harness.Value("done")}, nil
		},
		Now:    harness.FixedClock("2026-06-28T00:00:00Z"),
		NextId: harness.SequentialIDs(),
	}
	if _, err := runner.Run(model.RunTurnRequest{SessionId: "s", TurnId: "t"}); err != nil {
		t.Fatalf("run: %v", err)
	}

	all := journal.Records()
	prefix := harness.ReplayJournalUpTo(all, 1)
	if len(prefix) >= len(all) {
		t.Fatalf("prefix has %d records, want fewer than the whole journal's %d", len(prefix), len(all))
	}
	last := prefix[len(prefix)-1]
	if last.Session == nil || last.Session.Type != model.SessionEventTypeCheckpointCreated {
		t.Errorf("prefix ends with %#v, want the checkpoint_created event", last)
	}

	// A checkpoint the journal never reached returns the whole journal: there
	// is nothing beyond its own end to replay.
	if len(harness.ReplayJournalUpTo(all, 99)) != len(all) {
		t.Error("an unreached checkpoint must return the whole journal")
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func containsKey(keys []string, want string) bool {
	for _, key := range keys {
		if key == want {
			return true
		}
	}
	return false
}

func turnEvent(eventType model.TurnEventType, iteration int32, payload map[string]interface{}) model.TurnEvent {
	if payload == nil {
		payload = map[string]interface{}{}
	}
	return model.TurnEvent{
		Id:        "event",
		Type:      eventType,
		Timestamp: "2026-06-28T00:00:00Z",
		TurnId:    harness.String("turn-1"),
		Iteration: harness.Int32(iteration),
		Payload:   payload,
	}
}

func sessionEvent(eventType model.SessionEventType, payload map[string]interface{}) model.SessionEvent {
	if payload == nil {
		payload = map[string]interface{}{}
	}
	return model.SessionEvent{
		Id:        "event",
		Type:      eventType,
		Timestamp: "2026-06-28T00:00:00Z",
		SessionId: harness.String("session-1"),
		TurnId:    harness.String("turn-1"),
		Payload:   payload,
	}
}
