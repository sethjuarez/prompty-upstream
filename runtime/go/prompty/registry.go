package prompty

import (
	"context"
	"sort"
	"sync"

	model "prompty/model"
)

// The registries resolve pipeline components by key (spec §11.3):
//
//	renderer  <- agent.Template.Format.Kind
//	parser    <- agent.Template.Parser.Kind
//	executor  <- agent.Model.Provider
//	processor <- agent.Model.Provider
//
// The registered interfaces are the emitted ones from prompty/model, so a
// component written against the generated contract plugs in unchanged. A
// component may additionally implement the matching Context* interface below to
// receive cancellation; the pipeline prefers it when present.

// Renderer is the emitted renderer contract (model.Renderer).
type Renderer = model.Renderer

// Parser is the emitted parser contract (model.Parser).
type Parser = model.Parser

// Executor is the emitted executor contract (model.Executor).
type Executor = model.Executor

// Processor is the emitted processor contract (model.Processor).
type Processor = model.Processor

// ContextRenderer is the optional cancellation-aware extension of Renderer.
type ContextRenderer interface {
	RenderContext(ctx context.Context, agent model.Prompty, template string, inputs map[string]interface{}) (string, error)
}

// ContextParser is the optional cancellation-aware extension of Parser.
type ContextParser interface {
	ParseContext(ctx context.Context, agent model.Prompty, rendered string, parserContext *map[string]interface{}) ([]model.Message, error)
}

// registry is a small concurrency-safe key/value store shared by all four
// component kinds. Registration and lookup are pure in-memory operations and
// never perform I/O (spec §12.1).
type registry[T any] struct {
	mu        sync.RWMutex
	component string
	entries   map[string]T
	defaults  func() map[string]T
}

func newRegistry[T any](component string, defaults func() map[string]T) *registry[T] {
	r := &registry[T]{component: component, defaults: defaults}
	r.reset()
	return r
}

func (r *registry[T]) reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.entries = map[string]T{}
	if r.defaults != nil {
		for k, v := range r.defaults() {
			r.entries[k] = v
		}
	}
}

func (r *registry[T]) register(key string, value T) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.entries[key] = value
}

func (r *registry[T]) unregister(key string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.entries, key)
}

func (r *registry[T]) get(key string) (T, error) {
	r.mu.RLock()
	value, ok := r.entries[key]
	r.mu.RUnlock()
	if !ok {
		var zero T
		return zero, newInvokerError(r.component, key)
	}
	return value, nil
}

func (r *registry[T]) has(key string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.entries[key]
	return ok
}

func (r *registry[T]) keys() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.entries))
	for k := range r.entries {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

var (
	renderers = newRegistry("renderer", builtinRenderers)
	parsers   = newRegistry("parser", builtinParsers)
	// Executors and processors are provider packages (openai, anthropic,
	// foundry). Nothing is registered by default, so importing this package
	// never brings a network client along.
	executors  = newRegistry[Executor]("executor", nil)
	processors = newRegistry[Processor]("processor", nil)
)

func builtinRenderers() map[string]Renderer {
	jinja := NewJinja2Renderer()
	return map[string]Renderer{
		"jinja2":   jinja,
		"nunjucks": jinja,
		"mustache": NewMustacheRenderer(),
	}
}

func builtinParsers() map[string]Parser {
	return map[string]Parser{"prompty": NewPromptyParser()}
}

// RegisterRenderer registers (or replaces) a renderer under key.
func RegisterRenderer(key string, renderer Renderer) { renderers.register(key, renderer) }

// UnregisterRenderer removes the renderer registered under key.
func UnregisterRenderer(key string) { renderers.unregister(key) }

// GetRenderer returns the renderer registered under key, or an *InvokerError.
func GetRenderer(key string) (Renderer, error) { return renderers.get(key) }

// HasRenderer reports whether a renderer is registered under key.
func HasRenderer(key string) bool { return renderers.has(key) }

// RendererKeys lists the registered renderer keys in sorted order.
func RendererKeys() []string { return renderers.keys() }

// RegisterParser registers (or replaces) a parser under key.
func RegisterParser(key string, parser Parser) { parsers.register(key, parser) }

// UnregisterParser removes the parser registered under key.
func UnregisterParser(key string) { parsers.unregister(key) }

// GetParser returns the parser registered under key, or an *InvokerError.
func GetParser(key string) (Parser, error) { return parsers.get(key) }

// HasParser reports whether a parser is registered under key.
func HasParser(key string) bool { return parsers.has(key) }

// ParserKeys lists the registered parser keys in sorted order.
func ParserKeys() []string { return parsers.keys() }

// RegisterExecutor registers (or replaces) an executor under a provider key.
func RegisterExecutor(key string, executor Executor) { executors.register(key, executor) }

// UnregisterExecutor removes the executor registered under key.
func UnregisterExecutor(key string) { executors.unregister(key) }

// GetExecutor returns the executor registered under key, or an *InvokerError.
func GetExecutor(key string) (Executor, error) { return executors.get(key) }

// HasExecutor reports whether an executor is registered under key.
func HasExecutor(key string) bool { return executors.has(key) }

// ExecutorKeys lists the registered executor keys in sorted order.
func ExecutorKeys() []string { return executors.keys() }

// RegisterProcessor registers (or replaces) a processor under a provider key.
func RegisterProcessor(key string, processor Processor) { processors.register(key, processor) }

// UnregisterProcessor removes the processor registered under key.
func UnregisterProcessor(key string) { processors.unregister(key) }

// GetProcessor returns the processor registered under key, or an *InvokerError.
func GetProcessor(key string) (Processor, error) { return processors.get(key) }

// HasProcessor reports whether a processor is registered under key.
func HasProcessor(key string) bool { return processors.has(key) }

// ProcessorKeys lists the registered processor keys in sorted order.
func ProcessorKeys() []string { return processors.keys() }

// ClearCache resets every registry to its built-in defaults (spec §11.3). It
// exists so tests can undo registrations without leaking state between cases.
// Host-registered executors and processors must be registered again afterward.
func ClearCache() {
	renderers.reset()
	parsers.reset()
	executors.reset()
	processors.reset()
}

// rendererFor resolves the renderer for an agent from its template format kind.
func rendererFor(agent model.Prompty) (Renderer, error) {
	kind := DefaultFormatKind
	if agent.Template != nil && agent.Template.Format.Kind != "" {
		kind = agent.Template.Format.Kind
	}
	return GetRenderer(kind)
}

// parserFor resolves the parser for an agent from its template parser kind.
func parserFor(agent model.Prompty) (Parser, error) {
	kind := DefaultParserKind
	if agent.Template != nil && agent.Template.Parser.Kind != "" {
		kind = agent.Template.Parser.Kind
	}
	return GetParser(kind)
}

// ExecutorFor resolves the executor for an agent from its model provider.
func ExecutorFor(agent model.Prompty) (Executor, error) {
	return GetExecutor(providerKey(agent))
}

// ProcessorFor resolves the processor for an agent from its model provider.
func ProcessorFor(agent model.Prompty) (Processor, error) {
	return GetProcessor(providerKey(agent))
}

func providerKey(agent model.Prompty) string {
	if agent.Model.Provider != nil {
		return *agent.Model.Provider
	}
	return ""
}
