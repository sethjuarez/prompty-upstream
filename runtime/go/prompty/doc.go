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
// # Package layout
//
// The dependency graph is acyclic and deliberately one-directional:
//
//	prompty/model  <- prompty/wire  <- prompty  <- prompty/openai, prompty/anthropic
//
// prompty/wire holds the provider-neutral primitives (JSON Schema projection,
// option dialects, SSE decoding, diagnostic redaction) that both this package
// and the providers need. This package never imports a provider, so a provider
// may freely import it.
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
