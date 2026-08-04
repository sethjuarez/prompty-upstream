package harness_test

import (
	"context"
	"fmt"
	"path/filepath"

	harness "prompty/harness"
	model "prompty/model"
)

// Example_durableTurn runs one turn end to end through the harness and reads
// the journal back, which is the whole durability story in one place.
func Example_durableTurn() {
	directory, cleanup := exampleDirectory()
	defer cleanup()

	journal, err := harness.NewJSONLJournalWriter(filepath.Join(directory, "session.jsonl"))
	if err != nil {
		fmt.Println("open journal:", err)
		return
	}

	sink := &harness.CollectingEventSink{}
	checkpoints := harness.NewInMemoryCheckpointStore()

	runner := &harness.TurnRunner{
		EventSink:       sink,
		Journal:         journal,
		CheckpointStore: checkpoints,
		// Only the tools on the list may run; everything else is refused and
		// the refusal is shown to the model.
		PermissionResolver: harness.AllowListPermissionResolver{
			Allowed: map[string]bool{"add": true},
		},
		HostToolExecutor: harness.FunctionHostToolExecutor{
			Handlers: map[string]harness.HostToolHandler{
				"add": func(args map[string]interface{}, _ model.HostToolRequest) (interface{}, error) {
					return args["a"].(int) + args["b"].(int), nil
				},
			},
		},
		InvokeModel: func(request model.TurnModelRequest) (model.TurnModelResponse, error) {
			if request.Iteration == 0 {
				return model.TurnModelResponse{
					ToolRequests: []model.HostToolRequest{{
						RequestId: harness.String("exec-1"),
						ToolName:  "add",
						Arguments: map[string]interface{}{"a": 2, "b": 3},
					}},
				}, nil
			}
			return model.TurnModelResponse{Output: harness.Value("2 + 3 = 5")}, nil
		},
		// A fixed clock and id generator make the journal byte-identical
		// across runs, which is what turns replay into a regression test.
		Now:    harness.FixedClock("2026-06-28T00:00:00Z"),
		NextId: harness.SequentialIDs(),
	}

	result, err := runner.RunContext(context.Background(), model.RunTurnRequest{
		SessionId: "session-1",
		TurnId:    "turn-1",
	})
	if err != nil {
		fmt.Println("run turn:", err)
		return
	}

	fmt.Println("status:", result.Status)
	fmt.Println("iterations:", result.Iterations)
	fmt.Println("checkpoints:", len(result.Checkpoints))
	fmt.Println("observed events:", len(sink.TurnEvents())+len(sink.SessionEvents()))

	// Output:
	// status: success
	// iterations: 2
	// checkpoints: 2
	// observed events: 16
}

// Example_replayJournal reads a journal back and projects it onto the
// normalized, cross-runtime replay shape.
func Example_replayJournal() {
	directory, cleanup := exampleDirectory()
	defer cleanup()
	path := filepath.Join(directory, "session.jsonl")

	journal, err := harness.NewJSONLJournalWriter(path)
	if err != nil {
		fmt.Println("open journal:", err)
		return
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
		fmt.Println("run turn:", err)
		return
	}

	records, defects, err := harness.ReadJournal(path)
	if err != nil {
		fmt.Println("read journal:", err)
		return
	}
	// A torn trailing line is what a crashed process leaves behind; the
	// records before it are still durable and still readable.
	fmt.Println("defects:", len(defects))

	normalized, err := harness.NormalizeRecords(records)
	if err != nil {
		fmt.Println("normalize:", err)
		return
	}
	for _, key := range harness.ReplayKeys(normalized) {
		fmt.Println(key)
	}

	// Output:
	// defects: 0
	// session:session_start:session-1:turn-1
	// turn:turn_start:0
	// turn:llm_start:0
	// turn:llm_complete:0
	// session:checkpoint_created:session-1:turn-1
	// turn:turn_end:1:success
	// session:session_end:session-1:turn-1:success
	// summary:session-1:success:turns=1:checkpoints=1
}
