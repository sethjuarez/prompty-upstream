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
package prompty
