// Package harness implements the durable Prompty session harness in Go: the
// event sinks, replay journal, checkpoint store, permission resolvers and host
// tool executor a host needs to run an agent turn that can be observed,
// persisted, resumed and replayed.
//
// # Relationship to the emitted model package
//
// Every type crossing a boundary here is the emitted one from prompty/model —
// model.TurnEvent, model.SessionEvent, model.Checkpoint, model.HostToolRequest,
// model.HostToolResult, model.PermissionRequest, model.PermissionDecision,
// model.SessionSummary, model.ReplayJournalRecord and the emitted protocol
// interfaces. The generated contract is the cross-runtime one; this package
// supplies the Go implementations behind it and never redefines a wire shape.
//
// The model tree also carries minimal reference implementations. This package
// supplies host-facing variants with cancellation, validation, hardened error
// handling, guaranteed journal closure, and crash-tolerant replay while keeping
// the same emitted boundary types and event sequence.
//
// # A durable turn
//
//	sink := &harness.CollectingEventSink{}
//	journal, err := harness.NewJSONLJournalWriter(filepath.Join(dir, "session.jsonl"))
//	if err != nil {
//		return err
//	}
//	defer journal.Close(nil)
//
//	runner := &harness.TurnRunner{
//		EventSink:          sink,
//		Journal:            journal,
//		CheckpointStore:    harness.NewInMemoryCheckpointStore(),
//		PermissionResolver: harness.AllowAllPermissionResolver{},
//		HostToolExecutor: harness.FunctionHostToolExecutor{
//			Handlers: map[string]harness.HostToolHandler{
//				"add": func(args map[string]interface{}, _ model.HostToolRequest) (interface{}, error) {
//					return args["a"].(float64) + args["b"].(float64), nil
//				},
//			},
//		},
//		InvokeModel: func(request model.TurnModelRequest) (model.TurnModelResponse, error) {
//			// Call the provider, or replay a recorded response.
//			return model.TurnModelResponse{Output: harness.Value("done")}, nil
//		},
//	}
//
//	result, err := runner.RunContext(ctx, model.RunTurnRequest{
//		SessionId: "session-1",
//		TurnId:    "turn-1",
//		Inputs:    map[string]interface{}{"name": "Ada"},
//	})
//
// # Replaying a journal
//
// The journal is newline-delimited JSON, one record per line, so it can be
// tailed, truncated and diffed with ordinary tools. Reading it back yields the
// normalized model.ReplayJournalRecord projection that replay verification
// compares across runtimes:
//
//	records, defects, err := harness.ReadJournal(path)
//	if err != nil {
//		return err
//	}
//	if len(defects) > 0 {
//		// A torn final line from a crashed process, or a corrupted record.
//		log.Printf("journal has %d defective records", len(defects))
//	}
//	actual, err := harness.NormalizeRecords(records)
//	if err != nil {
//		return err
//	}
//	verification := harness.ReplayVerifier{}.Verify(model.ReplayVerificationRequest{
//		Expected: expected,
//		Actual:   actual,
//	})
//
// # Determinism
//
// TurnRunner takes its clock and its identifier generator as fields. Left nil
// they are wall-clock time and a monotonic counter; supplied, they make a turn
// byte-identical across runs, which is what makes journal comparison a usable
// regression test rather than a diff of timestamps.
package harness
