package anthropic

import (
	"context"
	"io"
	"sort"
	"sync"

	prompty "prompty"
	model "prompty/model"
	wire "prompty/wire"
)

// RawStream is the executor's streaming handoff. Body is owned by whoever
// consumes the stream and is closed exactly once, by DecodeStream.
type RawStream struct {
	Body io.ReadCloser
}

// DecodeStream turns a raw Anthropic SSE body into processed chunks.
//
// The returned stream owns Body: it is closed when the producer finishes, when
// the consumer calls Close, and when ctx is cancelled.
func DecodeStream(ctx context.Context, raw RawStream) *prompty.Stream {
	// The body is closed by two different goroutines — the producer on its way
	// out, and the cancellation watchdog to interrupt a read parked on the
	// network — and io.ReadCloser gives no guarantee about a concurrent Close.
	// Funnelling both through a sync.Once makes exactly one of them win.
	var closeOnce sync.Once
	closeBody := func() { closeOnce.Do(func() { _ = raw.Body.Close() }) }

	return prompty.NewStream(ctx, func(inner context.Context, emit func(prompty.Chunk) bool) error {
		defer closeBody()

		// Cancellation has to interrupt a read parked on the network, which the
		// decode loop alone cannot do; closing the body from a watchdog
		// unblocks it. The watchdog exits with the producer.
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
		blocks := newBlockAccumulator()

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
			if !handleEvent(payload, blocks, emit) {
				return nil
			}
		}

		for _, call := range blocks.calls() {
			if !emit(prompty.NewToolChunk(call)) {
				return nil
			}
		}

		if err := decoder.Err(); err != nil {
			if ctxErr := inner.Err(); ctxErr != nil {
				return ctxErr
			}
			return wire.NewProviderError("anthropic", "stream", 0, "", "", err)
		}
		return nil
	})
}

// handleEvent translates one decoded Anthropic stream event into chunks.
// It returns false when the consumer has gone away.
func handleEvent(payload map[string]interface{}, blocks *blockAccumulator, emit func(prompty.Chunk) bool) bool {
	// `type` is carried in the JSON payload as well as the SSE event field, so
	// the payload alone is enough to dispatch on.
	switch payload["type"] {
	case "error":
		message := "provider reported an error"
		if errObj, ok := payload["error"].(map[string]interface{}); ok {
			if text := stringField(errObj, "message"); text != "" {
				message = text
			}
		}
		return emit(prompty.NewErrorChunk(wire.RedactSecrets(message)))

	case "message_start":
		if message, ok := payload["message"].(map[string]interface{}); ok {
			if usage, ok := message["usage"].(map[string]interface{}); ok {
				blocks.usage = usageFromMap(usage)
				blocks.sawUsage = true
			}
		}

	case "content_block_start":
		block, ok := payload["content_block"].(map[string]interface{})
		if !ok {
			return true
		}
		if block["type"] == "tool_use" {
			blocks.start(indexOf(payload), stringField(block, "id"), stringField(block, "name"))
		}

	case "content_block_delta":
		delta, ok := payload["delta"].(map[string]interface{})
		if !ok {
			return true
		}
		switch delta["type"] {
		case "text_delta":
			if text := stringField(delta, "text"); text != "" {
				return emit(prompty.NewTextChunk(text))
			}
		case "thinking_delta":
			if thinking := stringField(delta, "thinking"); thinking != "" {
				return emit(prompty.NewThinkingChunk(thinking))
			}
		case "input_json_delta":
			if partial, ok := delta["partial_json"].(string); ok {
				blocks.appendArgs(indexOf(payload), partial)
			}
		}

	case "message_delta":
		// The terminal delta carries the output token count; the input count
		// only ever appears on message_start, so the two are merged rather than
		// the later one replacing the earlier.
		if usage, ok := payload["usage"].(map[string]interface{}); ok {
			merged := usageFromMap(usage)
			if merged.InputTokens == 0 {
				merged.InputTokens = blocks.usage.InputTokens
			}
			merged.TotalTokens = merged.InputTokens + merged.OutputTokens
			blocks.usage = merged
			blocks.sawUsage = true
		}

	case "message_stop":
		if blocks.sawUsage {
			blocks.sawUsage = false
			return emit(prompty.NewUsageChunk(blocks.usage))
		}
	}
	return true
}

func indexOf(payload map[string]interface{}) int {
	return int(toFloat(payload["index"]))
}

// blockAccumulator reassembles streamed tool_use blocks, whose arguments arrive
// as partial_json fragments and are only complete at the end of the stream.
type blockAccumulator struct {
	order    []int
	partial  map[int]*model.ToolCall
	usage    model.InvocationUsage
	sawUsage bool
}

func newBlockAccumulator() *blockAccumulator {
	return &blockAccumulator{partial: map[int]*model.ToolCall{}}
}

func (a *blockAccumulator) at(index int) *model.ToolCall {
	call, ok := a.partial[index]
	if !ok {
		call = &model.ToolCall{}
		a.partial[index] = call
		a.order = append(a.order, index)
	}
	return call
}

func (a *blockAccumulator) start(index int, id, name string) {
	call := a.at(index)
	if id != "" {
		call.Id = id
	}
	if name != "" {
		call.Name = name
	}
}

func (a *blockAccumulator) appendArgs(index int, delta string) {
	a.at(index).Arguments += delta
}

// calls returns completed calls in content-block index order, which is the
// order the model requested them and therefore the order results must be
// replayed in. Unnamed calls are dropped because they cannot be dispatched.
func (a *blockAccumulator) calls() []model.ToolCall {
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

// ProcessStream implements model.Processor. Prefer ProcessStreamContext.
func (p *Processor) ProcessStream(stream interface{}) (interface{}, error) {
	return p.ProcessStreamContext(context.Background(), stream)
}

// ProcessStreamContext converts the executor's RawStream into processed chunks.
func (p *Processor) ProcessStreamContext(ctx context.Context, stream interface{}) (interface{}, error) {
	switch s := stream.(type) {
	case RawStream:
		return DecodeStream(ctx, s), nil
	case *RawStream:
		if s == nil {
			return nil, wire.NewProviderError("anthropic", "process stream", 0, "", "received a nil stream", nil)
		}
		return DecodeStream(ctx, *s), nil
	case *prompty.Stream:
		return s, nil
	default:
		return nil, wire.NewProviderError("anthropic", "process stream", 0, "",
			"expected an anthropic.RawStream from the executor", nil)
	}
}
