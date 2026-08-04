package prompty

import (
	"context"

	model "prompty/model"
)

// The composition APIs below are the public entry points a host uses once a
// provider is registered (spec §1.4, §11.3):
//
//	Execute       agent + messages -> raw provider response
//	Process       raw response     -> clean typed result
//	ExecuteStream agent + messages -> raw provider stream
//	ProcessStream raw stream       -> *Stream of processed chunks
//	Run           agent + inputs   -> final answer, driving tool rounds
//	RunStream     agent + inputs   -> streamed answer for the first model call
//
// Every one takes a context.Context. Components that implement the optional
// Context* extension receive it; the rest are guarded by a boundary check so a
// cancelled context still short-circuits before a provider call is made.

// ContextExecutor is the optional cancellation-aware extension of Executor.
type ContextExecutor interface {
	ExecuteContext(ctx context.Context, agent model.Prompty, messages []model.Message) (interface{}, error)
	ExecuteStreamContext(ctx context.Context, agent model.Prompty, messages []model.Message) (interface{}, error)
}

// ContextProcessor is the optional cancellation-aware extension of Processor.
type ContextProcessor interface {
	ProcessContext(ctx context.Context, agent model.Prompty, response interface{}) (interface{}, error)
	ProcessStreamContext(ctx context.Context, stream interface{}) (interface{}, error)
}

// Execute calls the agent's provider with the given messages and returns the
// raw provider response.
func Execute(ctx context.Context, agent model.Prompty, messages []model.Message) (interface{}, error) {
	executor, err := ExecutorFor(agent)
	if err != nil {
		return nil, err
	}
	return executeWith(ctx, executor, agent, messages)
}

// Process turns a raw provider response into a clean typed result: a string for
// text, []model.ToolCall for a tool round, or the decoded value when the agent
// declares outputs.
func Process(ctx context.Context, agent model.Prompty, response interface{}) (interface{}, error) {
	processor, err := ProcessorFor(agent)
	if err != nil {
		return nil, err
	}
	return processWith(ctx, processor, agent, response)
}

// ExecuteStream calls the agent's provider in streaming mode and returns the
// raw provider stream, which is provider-specific and normally handed straight
// to ProcessStream.
func ExecuteStream(ctx context.Context, agent model.Prompty, messages []model.Message) (interface{}, error) {
	executor, err := ExecutorFor(agent)
	if err != nil {
		return nil, err
	}
	return executeStreamWith(ctx, executor, agent, messages)
}

// ProcessStream converts a raw provider stream into processed chunks.
func ProcessStream(ctx context.Context, agent model.Prompty, stream interface{}) (*Stream, error) {
	processor, err := ProcessorFor(agent)
	if err != nil {
		return nil, err
	}
	return processStreamWith(ctx, processor, stream)
}

// Stream runs one model call end to end and returns processed chunks.
//
// It covers the single-shot streaming case. Tool rounds cannot be streamed
// meaningfully without buffering the whole tool-call block first, so a turn
// that needs tools should use Run.
func StreamRun(ctx context.Context, agent model.Prompty, inputs map[string]interface{}) (*Stream, error) {
	messages, err := PrepareWithContext(ctx, agent, inputs)
	if err != nil {
		return nil, err
	}
	return StreamMessages(ctx, agent, messages)
}

// StreamMessages streams one model call over an already-prepared conversation.
func StreamMessages(ctx context.Context, agent model.Prompty, messages []model.Message) (*Stream, error) {
	executor, err := ExecutorFor(agent)
	if err != nil {
		return nil, err
	}
	processor, err := ProcessorFor(agent)
	if err != nil {
		return nil, err
	}

	raw, err := executeStreamWith(ctx, executor, agent, messages)
	if err != nil {
		return nil, err
	}
	return processStreamWith(ctx, processor, raw)
}

// executeWith prefers the cancellation-aware interface when the executor offers
// one, and otherwise falls back to the emitted contract behind a boundary check.
func executeWith(ctx context.Context, executor Executor, agent model.Prompty, messages []model.Message) (interface{}, error) {
	if ce, ok := executor.(ContextExecutor); ok {
		return ce.ExecuteContext(ctx, agent, messages)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return executor.Execute(agent, messages)
}

func executeStreamWith(ctx context.Context, executor Executor, agent model.Prompty, messages []model.Message) (interface{}, error) {
	if ce, ok := executor.(ContextExecutor); ok {
		return ce.ExecuteStreamContext(ctx, agent, messages)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return executor.ExecuteStream(agent, messages)
}

func processWith(ctx context.Context, processor Processor, agent model.Prompty, response interface{}) (interface{}, error) {
	if cp, ok := processor.(ContextProcessor); ok {
		return cp.ProcessContext(ctx, agent, response)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return processor.Process(agent, response)
}

// processStreamWith normalises whatever the processor returns into a *Stream.
//
// A processor that already speaks the runtime's streaming abstraction returns
// one directly; one written only against the emitted contract may return a
// channel or a slice, and both are adapted rather than rejected.
func processStreamWith(ctx context.Context, processor Processor, raw interface{}) (*Stream, error) {
	var (
		processed interface{}
		err       error
	)
	if cp, ok := processor.(ContextProcessor); ok {
		processed, err = cp.ProcessStreamContext(ctx, raw)
	} else {
		if cerr := ctx.Err(); cerr != nil {
			return nil, cerr
		}
		processed, err = processor.ProcessStream(raw)
	}
	if err != nil {
		return nil, err
	}
	return AsStream(ctx, processed)
}

// AsStream adapts a processor's streaming result into a *Stream.
func AsStream(ctx context.Context, processed interface{}) (*Stream, error) {
	switch s := processed.(type) {
	case *Stream:
		if s == nil {
			return nil, newValueError("Processor returned a nil stream")
		}
		return s, nil
	case <-chan Chunk:
		return adaptChannel(ctx, s), nil
	case chan Chunk:
		return adaptChannel(ctx, s), nil
	case []Chunk:
		return StreamOf(ctx, s...), nil
	default:
		return nil, newValueError("Processor returned %T, which is not a stream", processed)
	}
}

func adaptChannel(ctx context.Context, source <-chan Chunk) *Stream {
	return NewStream(ctx, func(inner context.Context, emit func(Chunk) bool) error {
		for {
			select {
			case <-inner.Done():
				return inner.Err()
			case chunk, ok := <-source:
				if !ok {
					return nil
				}
				if !emit(chunk) {
					return nil
				}
			}
		}
	})
}
