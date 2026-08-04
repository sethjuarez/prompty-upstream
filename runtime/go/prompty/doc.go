// Package prompty implements the hand-written Prompty runtime for Go.
//
// The canonical data model lives in the generated package prompty/model (emitted
// by the Typra emitter from the TypeSpec schema under schema/). This package is
// deliberately additive: it never redefines an emitted type, it composes them.
//
// The pipeline follows spec/spec.md:
//
//	§4 LOAD    Load / LoadString  — frontmatter + body -> model.Prompty
//	§5 RENDER  Render             — template engine + rich-kind nonce substitution
//	§6 PARSE   Parse              — role markers -> []model.Message
//	§5+§6      Prepare            — render + parse + thread expansion
//	§7 EXECUTE Execute            — messages -> raw provider response
//	§8 PROCESS Process            — raw response -> clean typed result
//	§9 RUN     Run / RunMessages  — the turn loop, driving tool rounds
//	§8.2      StreamRun           — one model call as processed chunks
//
// Renderers, parsers, executors and processors are resolved through
// concurrency-safe registries (§11.3) keyed off the agent:
//
//	renderer  <- agent.Template.Format.Kind   ("jinja2", "nunjucks", "mustache")
//	parser    <- agent.Template.Parser.Kind   ("prompty")
//	executor  <- agent.Model.Provider
//	processor <- agent.Model.Provider
//
// This package ships built-in renderers and the built-in "prompty" role parser.
// Executors and processors are provider packages and are not registered here, so
// nothing in this package performs network I/O.
//
// # Provider registration is explicit
//
// Importing a provider package has no side effects: it registers nothing. A host
// opts in by calling that package's Register function during start-up, before
// the first Execute or Run.
//
//	import (
//	    "prompty"
//	    "prompty/anthropic"
//	    "prompty/openai"
//	)
//
//	func main() {
//	    openai.Register()          // provider: "openai"
//	    openai.RegisterFoundry()   // providers: "foundry", "azure"
//	    anthropic.Register()       // provider: "anthropic"
//	}
//
// This is a deliberate choice over an init-time side effect. A blank import
// would link an HTTP client and its transitive dependencies into a binary that
// may never make a provider call, and it would make the set of reachable network
// endpoints depend on the import graph rather than on a line of code someone
// wrote. RegisterAs takes a configured executor instead of the default, which is
// how a host supplies its own HTTP client, proxy, timeout or credential source.
//
// Registering twice under the same key replaces the earlier entry, and
// ClearCache resets every registry to its built-in defaults — host-registered
// providers must be registered again afterwards.
//
// # A complete turn
//
// Load an agent, prepare its conversation, then run the turn loop. Run is the
// host-facing surface: it drives model calls and tool rounds until the model
// answers, or until MaxIterations stops it.
//
//	openai.Register()
//
//	agent, err := prompty.Load("agents/weather.prompty")
//	if err != nil {
//		return err
//	}
//
//	tools := prompty.NewToolRegistry()
//	tools.RegisterText("get_weather", func(ctx context.Context, args map[string]interface{}) (string, error) {
//		return lookupWeather(ctx, args["city"].(string))
//	})
//
//	result, err := prompty.Run(ctx, agent, map[string]interface{}{"city": "Paris"}, prompty.RunOptions{
//		Tools: tools,
//		Permit: func(_ context.Context, call model.ToolCall, _ map[string]interface{}) prompty.PermissionDecision {
//			if call.Name == "delete_everything" {
//				return prompty.Deny("that tool is not available in this session")
//			}
//			return prompty.Allow()
//		},
//		OnEvent: func(event prompty.Event) { log.Printf("%s %v", event.Type, event.Data) },
//	})
//	if err != nil {
//		return err
//	}
//	fmt.Println(result.Text())
//
// A host that maintains its own history — a chat session, a resumed thread —
// calls PrepareWithContext once and RunMessages for each turn, so the prompt is
// not re-rendered on every exchange.
//
// StreamRun performs a single model call and yields processed chunks as they
// arrive, for a UI that renders tokens rather than waiting for the answer.
//
//	stream, err := prompty.StreamRun(ctx, agent, inputs)
//	if err != nil {
//		return err
//	}
//	for chunk := range stream.Chunks() {
//		render(chunk)
//	}
//	return stream.Err()
//
// # Optional turn policies
//
// Three host policies wrap the loop. All are opt-in, and leaving them unset
// changes nothing about how a turn runs.
//
//   - ContextBudget trims the conversation to a character budget before each
//     model call, dropping the oldest non-system messages and inserting a
//     summary in their place (§13.3). Compaction replaces that mechanical
//     summary with a model- or host-generated one.
//
//   - Guardrails check the conversation before each model call, the final
//     answer before it is returned, and each tool before it runs (§13.4). An
//     input or output denial fails the turn with a *GuardrailError; a tool
//     denial becomes a model-visible refusal, so the model can adapt.
//
//   - Steering is a queue a host fills from another goroutine while the turn is
//     running (§13.5). It is drained between iterations and injected as user
//     messages, which is how a user redirects an agent mid-turn without
//     cancelling it.
//
//     steering := prompty.NewSteering()
//     go func() { steering.Send("actually, in Celsius") }()
//
//     result, err := prompty.Run(ctx, agent, inputs, prompty.RunOptions{
//     Tools:         tools,
//     ContextBudget: 50_000,
//     Steering:      steering,
//     Guardrails: &prompty.Guardrails{
//     Input: func(_ context.Context, messages []model.Message, _ model.Prompty) prompty.GuardrailResult {
//     if containsSecrets(messages) {
//     return prompty.DenyGuardrail("the prompt contains credentials")
//     }
//     return prompty.AllowGuardrail()
//     },
//     },
//     })
//
// # Model discovery
//
// Discovery reports which models a provider connection exposes. Listing is
// behind a registered ModelLister, so a host that never discovers models never
// links a discovery client, and a test registers a fake instead of a network.
//
//	openai.RegisterModelLister(openai.ModelLister{Fetch: fetchOpenAIModels})
//
//	models, err := prompty.ListModels("openai", connection)
//
// Results are enriched from a shared, provider-keyed capability dataset vendored
// from spec/data/model_capabilities.json under one cross-runtime rule:
// provider-supplied fields always win, and dataset entries only fill fields the
// provider left empty, matched by longest id prefix at token boundaries. That is
// how an OpenAI listing — which returns nothing but ids — comes back with the
// same context windows and modalities as an Anthropic one.
//
// # Durability and the engine
//
// Run answers a question. Two sibling packages answer the harder ones.
//
//   - prompty/harness is the durable session harness: typed turn and session
//     events, a newline-delimited JSON replay journal, checkpoints, permission
//     resolvers and host tool execution. Use it when a session must be
//     observed, persisted, resumed or replayed.
//   - prompty/engine is the provider-neutral turn engine: an ordered
//     model.EngineEvent per decision, a model.EngineCheckpoint per round, a
//     model.ModelInvocationContextSnapshot per model call, and a
//     model.TurnCommit at the end. Use it when a turn must survive the process
//     that started it, or when the provider holds delegated state the host has
//     to reattach to.
//
// Both reuse this package rather than reimplementing it: the engine's
// ProviderModelPort drives the same Executor and Processor contracts, and its
// RegistryToolPort dispatches through the same ToolRegistry.
//
// # Package layout
//
// The dependency graph is acyclic and deliberately one-directional:
//
//	prompty/model  <- prompty/wire  <- prompty  <- prompty/openai, prompty/anthropic
//	                                          \-- prompty/harness, prompty/engine
//
// prompty/wire holds the provider-neutral primitives (JSON Schema projection,
// option dialects, SSE decoding, diagnostic redaction) that both this package
// and the providers need. This package never imports a provider, a harness or
// the engine, so all of them may freely import it.
//
// # Security posture
//
//   - ${file:...} frontmatter references are confined to the .prompty file's own
//     directory after canonicalization. Additional roots are opt-in through
//     LoadOptions.AllowedFileRoots and can only be supplied by the host
//     application — frontmatter can never widen its own sandbox (§2.11).
//   - Template engines run with in-memory loaders only, so {% include %},
//     {% extends %} and {{> partial}} cannot reach the filesystem (§5.5).
//   - Template output is never HTML-escaped; .prompty templates are not HTML.
//   - .env files are never auto-loaded; that is an application concern (§4.3).
//   - Provider credentials live only in a request's Header map. Every diagnostic
//     path — error strings, request rendering, logged headers — goes through
//     prompty/wire's redactors, and a bound tool parameter is stripped from the
//     schema the model sees so the host, not the model, controls its value.
package prompty
