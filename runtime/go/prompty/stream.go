package prompty

import (
	"context"
	"fmt"
	"sync"

	model "prompty/model"
)

// Chunk is one processed item of a model stream. Its dynamic type is always one
// of the emitted variants: model.TextChunk, model.ThinkingChunk,
// model.ToolChunk, model.UsageChunk or model.ErrorChunk (spec §8.2).
//
// It is an alias for interface{} rather than a sealed interface because the
// emitted variants are plain structs with no shared method set, and adding one
// would mean touching generated code.
type Chunk = interface{}

// Chunk kind discriminators, matching the emitted `kind` field.
const (
	ChunkKindText     = "text"
	ChunkKindThinking = "thinking"
	ChunkKindTool     = "tool"
	ChunkKindUsage    = "usage"
	ChunkKindError    = "error"
)

// NewTextChunk builds a text delta chunk.
func NewTextChunk(value string) model.TextChunk {
	return model.TextChunk{Kind: ChunkKindText, Value: value}
}

// NewThinkingChunk builds a reasoning delta chunk.
func NewThinkingChunk(value string) model.ThinkingChunk {
	return model.ThinkingChunk{Kind: ChunkKindThinking, Value: value}
}

// NewToolChunk builds a completed tool-call chunk. Providers accumulate partial
// argument deltas and emit one of these once a call is whole.
func NewToolChunk(call model.ToolCall) model.ToolChunk {
	return model.ToolChunk{Kind: ChunkKindTool, ToolCall: call}
}

// NewUsageChunk builds a token-usage chunk.
func NewUsageChunk(usage model.InvocationUsage) model.UsageChunk {
	return model.UsageChunk{Kind: ChunkKindUsage, Usage: usage}
}

// NewErrorChunk builds an in-band error chunk. A provider uses this for a
// failure the stream can report and continue past; a failure that ends the
// stream is reported through Stream.Err instead.
func NewErrorChunk(message string) model.ErrorChunk {
	return model.ErrorChunk{Kind: ChunkKindError, Message: message}
}

// ChunkKind returns the discriminator of a processed chunk, or "" when the
// value is not a recognised variant.
func ChunkKind(chunk Chunk) string {
	switch c := chunk.(type) {
	case model.TextChunk:
		return c.Kind
	case *model.TextChunk:
		return c.Kind
	case model.ThinkingChunk:
		return c.Kind
	case *model.ThinkingChunk:
		return c.Kind
	case model.ToolChunk:
		return c.Kind
	case *model.ToolChunk:
		return c.Kind
	case model.UsageChunk:
		return c.Kind
	case *model.UsageChunk:
		return c.Kind
	case model.ErrorChunk:
		return c.Kind
	case *model.ErrorChunk:
		return c.Kind
	default:
		return ""
	}
}

// ChunkText returns the text carried by a text chunk. The second result is
// false for every other chunk kind, including thinking chunks — reasoning text
// must never be silently concatenated into a user-visible answer.
func ChunkText(chunk Chunk) (string, bool) {
	switch c := chunk.(type) {
	case model.TextChunk:
		return c.Value, true
	case *model.TextChunk:
		return c.Value, true
	default:
		return "", false
	}
}

// StreamProducer fills a stream.
//
// It runs on its own goroutine and must return as soon as emit reports false,
// which happens when the consumer closed the stream or ctx was cancelled.
// Returning a non-nil error surfaces through Stream.Err.
type StreamProducer func(ctx context.Context, emit func(Chunk) bool) error

// Stream is a cancellable, single-consumer stream of processed chunks.
//
// The producer goroutine is owned by the Stream: it is started by NewStream and
// is guaranteed to have exited by the time Close returns. A consumer that stops
// reading early must call Close, otherwise the producer stays blocked on its
// pending send. Close is safe to call more than once and safe to call
// concurrently with a reader.
type Stream struct {
	chunks chan Chunk
	// parent is the caller's context, kept so Err can distinguish an upstream
	// cancellation from an ordinary consumer-initiated Close.
	parent context.Context
	cancel context.CancelFunc
	// stopped is the derived context's Done channel. emit selects on it — not
	// on done — because done is closed by the producer itself and so could
	// never unblock the producer's own pending send.
	stopped <-chan struct{}

	closeOnce sync.Once
	userClose bool

	mu  sync.Mutex
	err error
	// done closes when the producer goroutine has exited.
	done chan struct{}
}

// NewStream starts produce on a new goroutine and returns the stream it feeds.
// The returned stream must be drained to completion or closed.
func NewStream(ctx context.Context, produce StreamProducer) *Stream {
	if ctx == nil {
		ctx = context.Background()
	}
	child, cancel := context.WithCancel(ctx)

	s := &Stream{
		chunks:  make(chan Chunk),
		parent:  ctx,
		cancel:  cancel,
		stopped: child.Done(),
		done:    make(chan struct{}),
	}

	go func() {
		// s.setErr runs before any defer, so the error is always recorded
		// before the channel closes and a consumer that observes the close and
		// then reads Err never races with the assignment.
		defer close(s.chunks)
		defer close(s.done)
		defer cancel()
		defer func() {
			// A panicking producer must not deadlock the consumer on a channel
			// that never closes.
			if r := recover(); r != nil {
				s.setErr(fmt.Errorf("prompty: stream producer panicked: %v", r))
			}
		}()
		s.setErr(produce(child, s.emit))
	}()

	return s
}

// StreamOf returns a stream that yields the given chunks and ends. It exists
// for tests and for providers that already hold a fully materialised result.
func StreamOf(ctx context.Context, chunks ...Chunk) *Stream {
	return NewStream(ctx, func(_ context.Context, emit func(Chunk) bool) error {
		for _, chunk := range chunks {
			if !emit(chunk) {
				return nil
			}
		}
		return nil
	})
}

// emit hands one chunk to the consumer, reporting false when the stream is
// finished and the producer should stop.
func (s *Stream) emit(chunk Chunk) bool {
	select {
	case s.chunks <- chunk:
		return true
	case <-s.stopped:
		return false
	}
}

// Chunks exposes the underlying channel for `for chunk := range s.Chunks()`.
// The channel closes when the producer finishes; check Err afterwards.
func (s *Stream) Chunks() <-chan Chunk { return s.chunks }

// Next pulls the next chunk. The second result is false once the stream has
// ended, at which point Err reports why.
func (s *Stream) Next() (Chunk, bool) {
	chunk, ok := <-s.chunks
	return chunk, ok
}

// Err returns the reason the stream ended: the producer's error, or the
// caller's context error when the caller cancelled, or nil for a clean end or a
// consumer-initiated Close.
func (s *Stream) Err() error {
	s.mu.Lock()
	err, closedByConsumer := s.err, s.userClose
	s.mu.Unlock()

	if err != nil {
		return err
	}
	if closedByConsumer {
		return nil
	}
	return s.parent.Err()
}

func (s *Stream) setErr(err error) {
	if err == nil {
		return
	}
	s.mu.Lock()
	if s.err == nil {
		s.err = err
	}
	s.mu.Unlock()
}

// Close stops the stream and blocks until the producer goroutine has exited.
//
// It is the consumer's obligation whenever the stream is abandoned before it
// ends on its own. Draining is what unblocks a producer parked on a send, so
// Close cannot be reduced to a bare cancel.
func (s *Stream) Close() error {
	s.closeOnce.Do(func() {
		// userClose is read by Err, which a consumer may call concurrently with
		// Close, so it is guarded by the same mutex as err.
		s.mu.Lock()
		s.userClose = true
		s.mu.Unlock()

		s.cancel()
		// Unblock any pending send, then let the producer run to completion.
		for range s.chunks { //nolint:revive // draining is the point
		}
	})
	<-s.done
	return s.Err()
}

// StreamResult is the accumulated outcome of a processed stream.
type StreamResult struct {
	// Text is every text delta concatenated in arrival order.
	Text string
	// Thinking is every reasoning delta concatenated in arrival order.
	Thinking string
	// ToolCalls are the completed tool calls, in the order the provider closed
	// them. Order is part of the contract: tool results must be replayed to the
	// model in the same order the calls were requested.
	ToolCalls []model.ToolCall
	// Usage is the last usage chunk seen, zero when the provider sent none.
	Usage model.InvocationUsage
	// Errors are in-band error chunk messages. Their presence does not by itself
	// mean the stream failed; a fatal failure is reported by Collect's error.
	Errors []string
}

// Collect drains a stream into a StreamResult.
//
// The stream is always closed before Collect returns, including on the error
// path, so a caller cannot leak the producer by forgetting to.
func Collect(ctx context.Context, stream *Stream) (StreamResult, error) {
	var result StreamResult
	if stream == nil {
		return result, newValueError("Cannot collect a nil stream")
	}
	defer stream.Close()

	var (
		text     []byte
		thinking []byte
	)

	for {
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		case chunk, ok := <-stream.chunks:
			if !ok {
				result.Text = string(text)
				result.Thinking = string(thinking)
				return result, stream.Err()
			}
			switch c := chunk.(type) {
			case model.TextChunk:
				text = append(text, c.Value...)
			case model.ThinkingChunk:
				thinking = append(thinking, c.Value...)
			case model.ToolChunk:
				result.ToolCalls = append(result.ToolCalls, c.ToolCall)
			case model.UsageChunk:
				result.Usage = c.Usage
			case model.ErrorChunk:
				result.Errors = append(result.Errors, c.Message)
			}
		}
	}
}
