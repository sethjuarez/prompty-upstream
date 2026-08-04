// Package engine implements the provider-neutral Prompty turn engine: the
// deterministic state machine that turns a prepared conversation into a
// committed turn, over the emitted engine request, response and state types.
//
// # What the engine is for
//
// prompty.Run is the practical host surface — render, call the provider, run
// tools, answer. It is the right thing for an application. It is deliberately
// *not* durable: it has no sequence numbers, no checkpoints, no snapshots and
// no notion of a turn that survives the process that started it.
//
// The engine is the durable one. Every decision it takes is an ordered
// model.EngineEvent, every model round produces a resumable
// model.EngineCheckpoint and an auditable model.ModelInvocationContextSnapshot,
// and the turn ends in a model.TurnCommit that says exactly what happened.
// A host that needs to answer "what did the agent do, and can I resume it"
// wants this; a host that just needs an answer wants prompty.Run.
//
// # Ports
//
// The engine owns ordering, retry, cancellation and durability. Everything else
// is a port the host supplies: ModelPort, HostPolicyPort, PermissionPort,
// ToolPort, ConversationPort, DurabilityPort, PostCommitPort, RetryPolicyPort
// and Clock. Nothing in this package knows a provider's wire format, which is
// what makes the same engine drive OpenAI, Anthropic and a recorded fixture.
//
// Adapters in adapters.go bridge the ports to the existing runtime — a
// prompty.ToolRegistry becomes a ToolPort, a prompty.Executor and
// prompty.Processor pair becomes a ModelPort — so the engine reuses the
// provider and tool code rather than reimplementing it.
//
// # Context portability
//
// A turn's model.InvocationContextState records whether the conversation is
// portable (fully described by its messages), delegated (the provider holds
// state the engine references by id) or opaque. A provider that returns a
// continuation handle rather than a transcript sets it on
// ModelInvocationResponse.NextContextState, and the engine carries it into
// every later snapshot and into the commit — so a resume knows whether it may
// replay messages or must reattach to provider-side state.
//
// # Usage
//
//	result, err := engine.Run(ctx, engine.Request{
//		SessionId:     "session-1",
//		TurnId:        "turn-1",
//		Messages:      prepared,
//		MaxIterations: 10,
//		Ports: engine.Ports{
//			Model:        modelPort,
//			Tool:         engine.RegistryToolPort{Registry: tools},
//			Permission:   engine.AllowAllPermissionPort{},
//			Conversation: conversationPort,
//			Durability:   durability,
//		},
//	})
//	if err != nil {
//		return err
//	}
//	// result.Commit.Status, result.Snapshots, result.ToolResults
package engine
