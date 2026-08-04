package openai

import (
	"context"
	"io"
	"sort"
	"sync"

	prompty "prompty"
	model "prompty/model"
	wire "prompty/wire"
)

// RawStream is the executor's streaming handoff: an undecoded SSE body plus
// enough context for the processor to know which dialect to decode.
//
// Body is owned by whoever consumes the stream and is closed exactly once, by
// DecodeStream, including on every error path.
type RawStream struct {
	APIType string
	Body    io.ReadCloser
}

// DecodeStream turns a raw SSE body into processed chunks.
//
// The returned stream owns Body: it is closed when the producer finishes, when
// the consumer calls Close, and when ctx is cancelled. That is what keeps a
// cancelled turn from leaking both a goroutine and a socket.
func DecodeStream(ctx context.Context, raw RawStream) *prompty.Stream {
	// The body is closed by two different goroutines — the producer on its way
	// out, and the cancellation watchdog to interrupt a read parked on the
	// network — and io.ReadCloser gives no guarantee about a concurrent Close.
	// Funnelling both through a sync.Once makes exactly one of them win.
	var closeOnce sync.Once
	closeBody := func() { closeOnce.Do(func() { _ = raw.Body.Close() }) }

	return prompty.NewStream(ctx, func(inner context.Context, emit func(prompty.Chunk) bool) error {
		defer closeBody()

		// A cancelled context has to interrupt a read that is parked on the
		// network, which a decoder loop alone cannot do. Closing the body from
		// a watchdog unblocks it; the watchdog exits with the producer.
		finished := make(chan struct{})
		defer close(finished)
		go func() {
			select {
			case <-inner.Done():
				closeBody()
			case <-finished:
			}
		}()

		decoder := wire.NewSSEDecoder(raw.Body)
		accumulator := newToolAccumulator()

		for {
			if err := inner.Err(); err != nil {
				return err
			}
			event, ok := decoder.Next()
			if !ok {
				break
			}
			payload, ok := event.JSON()
			if !ok {
				continue
			}
			if !emitEvents(raw.APIType, payload, accumulator, emit) {
				return nil
			}
		}

		// Tool calls are only complete once the stream ends, because arguments
		// arrive as fragments across many events.
		for _, call := range accumulator.calls() {
			if !emit(prompty.NewToolChunk(call)) {
				return nil
			}
		}

		if err := decoder.Err(); err != nil {
			// A closed body is how cancellation is delivered, so report the
			// context error rather than the read failure it caused.
			if ctxErr := inner.Err(); ctxErr != nil {
				return ctxErr
			}
			return wire.NewProviderError("openai", "stream", 0, "", "", err)
		}
		return nil
	})
}

// emitEvents translates one decoded SSE payload into chunks. It returns false
// when the consumer has gone away and the producer should stop.
func emitEvents(apiType string, payload map[string]interface{}, acc *toolAccumulator, emit func(prompty.Chunk) bool) bool {
	// An in-band error object ends the useful part of the stream but is
	// reported as a chunk so a consumer sees why rather than just a short read.
	if errObj, ok := payload["error"].(map[string]interface{}); ok {
		message := stringField(errObj, "message")
		if message == "" {
			message = "provider reported an error"
		}
		return emit(prompty.NewErrorChunk(wire.RedactSecrets(message)))
	}

	if apiType == APITypeResponses {
		return emitResponsesEvent(payload, acc, emit)
	}
	return emitChatEvent(payload, acc, emit)
}

func emitChatEvent(payload map[string]interface{}, acc *toolAccumulator, emit func(prompty.Chunk) bool) bool {
	if usage, ok := payload["usage"].(map[string]interface{}); ok {
		if !emit(prompty.NewUsageChunk(usageFromMap(usage))) {
			return false
		}
	}

	choices, _ := payload["choices"].([]interface{})
	for _, item := range choices {
		choice, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		delta, ok := choice["delta"].(map[string]interface{})
		if !ok {
			continue
		}

		if reasoning := stringField(delta, "reasoning_content", "reasoning"); reasoning != "" {
			if !emit(prompty.NewThinkingChunk(reasoning)) {
				return false
			}
		}
		if content, ok := delta["content"].(string); ok && content != "" {
			if !emit(prompty.NewTextChunk(content)) {
				return false
			}
		}

		rawCalls, _ := delta["tool_calls"].([]interface{})
		for _, rawCall := range rawCalls {
			call, ok := rawCall.(map[string]interface{})
			if !ok {
				continue
			}
			acc.accumulate(call)
		}
	}
	return true
}

func emitResponsesEvent(payload map[string]interface{}, acc *toolAccumulator, emit func(prompty.Chunk) bool) bool {
	switch payload["type"] {
	case "response.output_text.delta":
		if delta, ok := payload["delta"].(string); ok && delta != "" {
			return emit(prompty.NewTextChunk(delta))
		}

	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		if delta, ok := payload["delta"].(string); ok && delta != "" {
			return emit(prompty.NewThinkingChunk(delta))
		}

	case "response.output_item.added":
		if item, ok := payload["item"].(map[string]interface{}); ok && item["type"] == "function_call" {
			acc.start(int(toFloat(payload["output_index"])),
				stringField(item, "call_id", "id"), stringField(item, "name"))
		}

	case "response.function_call_arguments.delta":
		if delta, ok := payload["delta"].(string); ok {
			acc.appendArgs(int(toFloat(payload["output_index"])), delta)
		}

	case "response.completed":
		if response, ok := payload["response"].(map[string]interface{}); ok {
			if usage, ok := response["usage"].(map[string]interface{}); ok {
				return emit(prompty.NewUsageChunk(usageFromMap(usage)))
			}
		}
	}
	return true
}

// toolAccumulator reassembles streamed tool calls.
//
// Both dialects deliver a call's arguments as a run of fragments keyed by an
// index, so a call cannot be emitted until the stream ends. Ordering is by that
// index, not by arrival, because tool results must be replayed to the model in
// the order the calls were requested.
type toolAccumulator struct {
	order   []int
	partial map[int]*model.ToolCall
}

func newToolAccumulator() *toolAccumulator {
	return &toolAccumulator{partial: map[int]*model.ToolCall{}}
}

func (a *toolAccumulator) at(index int) *model.ToolCall {
	call, ok := a.partial[index]
	if !ok {
		call = &model.ToolCall{}
		a.partial[index] = call
		a.order = append(a.order, index)
	}
	return call
}

// accumulate folds one chat-dialect tool_calls delta into the pending call.
func (a *toolAccumulator) accumulate(delta map[string]interface{}) {
	index := 0
	if raw, ok := delta["index"]; ok {
		index = int(toFloat(raw))
	}
	call := a.at(index)

	if id := stringField(delta, "id"); id != "" {
		call.Id = id
	}
	function, _ := delta["function"].(map[string]interface{})
	if function == nil {
		return
	}
	if name := stringField(function, "name"); name != "" {
		call.Name = name
	}
	if args, ok := function["arguments"].(string); ok {
		call.Arguments += args
	}
}

func (a *toolAccumulator) start(index int, id, name string) {
	call := a.at(index)
	if id != "" {
		call.Id = id
	}
	if name != "" {
		call.Name = name
	}
}

func (a *toolAccumulator) appendArgs(index int, delta string) {
	a.at(index).Arguments += delta
}

// calls returns the completed calls in index order. Calls that never received a
// name are dropped: an unnamed call cannot be dispatched, and forwarding one
// would fail later with a less obvious error.
func (a *toolAccumulator) calls() []model.ToolCall {
	indexes := append([]int(nil), a.order...)
	sort.Ints(indexes)

	out := make([]model.ToolCall, 0, len(indexes))
	for _, index := range indexes {
		call := a.partial[index]
		if call == nil || call.Name == "" {
			continue
		}
		if call.Arguments == "" {
			call.Arguments = "{}"
		}
		out = append(out, *call)
	}
	return out
}
