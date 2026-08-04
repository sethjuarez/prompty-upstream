package prompty_test

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"testing"
	"time"

	prompty "prompty"
	model "prompty/model"
)

// ---------------------------------------------------------------------------
// Stream lifecycle
// ---------------------------------------------------------------------------

func TestStreamDeliversChunksInOrder(t *testing.T) {
	stream := prompty.StreamOf(context.Background(),
		prompty.NewTextChunk("Hel"),
		prompty.NewThinkingChunk("hmm"),
		prompty.NewTextChunk("lo"),
		prompty.NewToolChunk(model.ToolCall{Id: "call_1", Name: "t", Arguments: "{}"}),
		prompty.NewUsageChunk(model.InvocationUsage{InputTokens: 3, OutputTokens: 2, TotalTokens: 5}),
		prompty.NewErrorChunk("degraded"),
	)

	result, err := prompty.Collect(context.Background(), stream)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if result.Text != "Hello" {
		t.Errorf("text = %q", result.Text)
	}
	if result.Thinking != "hmm" {
		t.Errorf("thinking = %q", result.Thinking)
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].Id != "call_1" {
		t.Errorf("tool calls = %+v", result.ToolCalls)
	}
	if result.Usage.TotalTokens != 5 {
		t.Errorf("usage = %+v", result.Usage)
	}
	if len(result.Errors) != 1 || result.Errors[0] != "degraded" {
		t.Errorf("errors = %v", result.Errors)
	}
}

func TestChunkKindAndText(t *testing.T) {
	cases := []struct {
		chunk prompty.Chunk
		kind  string
	}{
		{prompty.NewTextChunk("a"), prompty.ChunkKindText},
		{prompty.NewThinkingChunk("a"), prompty.ChunkKindThinking},
		{prompty.NewToolChunk(model.ToolCall{}), prompty.ChunkKindTool},
		{prompty.NewUsageChunk(model.InvocationUsage{}), prompty.ChunkKindUsage},
		{prompty.NewErrorChunk("a"), prompty.ChunkKindError},
	}
	for _, testCase := range cases {
		if got := prompty.ChunkKind(testCase.chunk); got != testCase.kind {
			t.Errorf("ChunkKind(%T) = %q, want %q", testCase.chunk, got, testCase.kind)
		}
	}
	if got := prompty.ChunkKind("not a chunk"); got != "" {
		t.Errorf("ChunkKind of a foreign value = %q, want empty", got)
	}

	// Reasoning text must never be mistaken for the answer.
	if _, ok := prompty.ChunkText(prompty.NewThinkingChunk("secret reasoning")); ok {
		t.Error("ChunkText accepted a thinking chunk")
	}
	if text, ok := prompty.ChunkText(prompty.NewTextChunk("answer")); !ok || text != "answer" {
		t.Errorf("ChunkText = %q, %v", text, ok)
	}
}

// TestStreamCloseStopsAnInfiniteProducer is the leak assertion: a producer that
// would emit forever must be stopped by Close, and Close must not return until
// the goroutine has exited.
func TestStreamCloseStopsAnInfiniteProducer(t *testing.T) {
	exited := make(chan struct{})

	stream := prompty.NewStream(context.Background(), func(ctx context.Context, emit func(prompty.Chunk) bool) error {
		defer close(exited)
		for i := 0; ; i++ {
			if !emit(prompty.NewTextChunk(fmt.Sprint(i))) {
				return nil
			}
		}
	})

	for i := 0; i < 3; i++ {
		if _, ok := stream.Next(); !ok {
			t.Fatalf("stream ended early at %d", i)
		}
	}

	done := make(chan error, 1)
	go func() { done <- stream.Close() }()

	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Close: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not return; the producer goroutine leaked")
	}

	select {
	case <-exited:
	case <-time.After(5 * time.Second):
		t.Fatal("Close returned before the producer goroutine exited")
	}
}

// TestStreamContextCancellationStopsProducer proves an upstream cancellation
// unblocks a producer parked on a send.
func TestStreamContextCancellationStopsProducer(t *testing.T) {
	exited := make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())

	stream := prompty.NewStream(ctx, func(inner context.Context, emit func(prompty.Chunk) bool) error {
		defer close(exited)
		for {
			if !emit(prompty.NewTextChunk("x")) {
				return inner.Err()
			}
		}
	})

	if _, ok := stream.Next(); !ok {
		t.Fatal("expected at least one chunk")
	}
	cancel()

	select {
	case <-exited:
	case <-time.After(5 * time.Second):
		t.Fatal("cancelling the context did not stop the producer")
	}

	// Draining lets the channel close; Err then reports the cancellation.
	for range stream.Chunks() { //nolint:revive // draining is the point
	}
	if !errors.Is(stream.Err(), context.Canceled) {
		t.Errorf("Err() = %v, want context.Canceled", stream.Err())
	}
}

// TestStreamCloseIsIdempotent: a defer plus an explicit Close is normal.
func TestStreamCloseIsIdempotent(t *testing.T) {
	stream := prompty.StreamOf(context.Background(), prompty.NewTextChunk("a"))
	for i := 0; i < 3; i++ {
		done := make(chan struct{})
		go func() { defer close(done); _ = stream.Close() }()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatalf("Close call %d blocked", i)
		}
	}
}

func TestStreamProducerErrorSurfaces(t *testing.T) {
	sentinel := errors.New("provider exploded")

	stream := prompty.NewStream(context.Background(), func(context.Context, func(prompty.Chunk) bool) error {
		return sentinel
	})

	if _, err := prompty.Collect(context.Background(), stream); !errors.Is(err, sentinel) {
		t.Errorf("Collect error = %v, want %v", err, sentinel)
	}
}

// TestStreamProducerPanicDoesNotDeadlock: a panicking producer must become an
// error, not a consumer blocked forever on a channel that never closes.
func TestStreamProducerPanicDoesNotDeadlock(t *testing.T) {
	stream := prompty.NewStream(context.Background(), func(context.Context, func(prompty.Chunk) bool) error {
		panic("boom")
	})

	done := make(chan error, 1)
	go func() {
		_, err := prompty.Collect(context.Background(), stream)
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil || !strings.Contains(err.Error(), "panicked") {
			t.Errorf("error = %v, want a panic report", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("a panicking producer deadlocked the consumer")
	}
}

// TestCollectStopsOnContextCancellation and closes the stream on the way out.
func TestCollectStopsOnContextCancellation(t *testing.T) {
	exited := make(chan struct{})

	stream := prompty.NewStream(context.Background(), func(_ context.Context, emit func(prompty.Chunk) bool) error {
		defer close(exited)
		for {
			if !emit(prompty.NewTextChunk("x")) {
				return nil
			}
		}
	})

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	if _, err := prompty.Collect(ctx, stream); !errors.Is(err, context.Canceled) {
		t.Errorf("Collect error = %v, want context.Canceled", err)
	}

	// Collect closes the stream on every path, so the producer is already gone.
	select {
	case <-exited:
	case <-time.After(5 * time.Second):
		t.Fatal("Collect returned without tearing down the producer")
	}
}

// TestNoGoroutineLeakAcrossManyStreams is the aggregate leak check: after
// creating and tearing down many streams the goroutine count must return to
// roughly where it started.
func TestNoGoroutineLeakAcrossManyStreams(t *testing.T) {
	settle := func() int {
		for i := 0; i < 50; i++ {
			runtime.GC()
			time.Sleep(10 * time.Millisecond)
		}
		return runtime.NumGoroutine()
	}

	before := settle()

	for i := 0; i < 200; i++ {
		stream := prompty.NewStream(context.Background(), func(_ context.Context, emit func(prompty.Chunk) bool) error {
			for {
				if !emit(prompty.NewTextChunk("x")) {
					return nil
				}
			}
		})
		if _, ok := stream.Next(); !ok {
			t.Fatalf("stream %d produced nothing", i)
		}
		if err := stream.Close(); err != nil {
			t.Fatalf("stream %d close: %v", i, err)
		}
	}

	after := settle()
	// A small delta is normal test-runtime noise; a leak of one goroutine per
	// stream would show up as roughly 200.
	if after-before > 20 {
		t.Errorf("goroutines grew from %d to %d across 200 streams", before, after)
	}
}

// TestAsStreamAdaptsChannelsAndSlices proves a processor written only against
// the emitted contract still composes.
func TestAsStreamAdaptsChannelsAndSlices(t *testing.T) {
	source := make(chan prompty.Chunk, 2)
	source <- prompty.NewTextChunk("a")
	source <- prompty.NewTextChunk("b")
	close(source)

	fromChannel, err := prompty.AsStream(context.Background(), source)
	if err != nil {
		t.Fatalf("AsStream(chan): %v", err)
	}
	result, err := prompty.Collect(context.Background(), fromChannel)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if result.Text != "ab" {
		t.Errorf("text = %q, want \"ab\"", result.Text)
	}

	fromSlice, err := prompty.AsStream(context.Background(),
		[]prompty.Chunk{prompty.NewTextChunk("c")})
	if err != nil {
		t.Fatalf("AsStream(slice): %v", err)
	}
	result, err = prompty.Collect(context.Background(), fromSlice)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if result.Text != "c" {
		t.Errorf("text = %q, want \"c\"", result.Text)
	}

	if _, err := prompty.AsStream(context.Background(), 42); err == nil {
		t.Error("AsStream accepted a value that is not a stream")
	}
}
