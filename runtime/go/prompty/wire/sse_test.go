package wire_test

import (
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	wire "prompty/wire"
)

// collectEvents drains a decoder into a slice.
func collectEvents(decoder *wire.SSEDecoder) []wire.SSEEvent {
	var events []wire.SSEEvent
	for {
		event, ok := decoder.Next()
		if !ok {
			return events
		}
		events = append(events, event)
	}
}

func TestSSEBasicDecoding(t *testing.T) {
	body := "data: {\"a\":1}\n\ndata: {\"a\":2}\n\ndata: [DONE]\n\n"

	events := collectEvents(wire.NewSSEDecoder(strings.NewReader(body)))
	if len(events) != 2 {
		t.Fatalf("decoded %d events, want 2: %+v", len(events), events)
	}
	for i, want := range []string{`{"a":1}`, `{"a":2}`} {
		if events[i].Data != want {
			t.Errorf("event[%d].Data = %q, want %q", i, events[i].Data, want)
		}
	}
}

// TestSSEStopsAtDoneSentinel: anything a producer sends after [DONE] is not
// part of the response and must not reach the consumer.
func TestSSEStopsAtDoneSentinel(t *testing.T) {
	body := "data: {\"a\":1}\n\ndata: [DONE]\n\ndata: {\"a\":2}\n\n"

	events := collectEvents(wire.NewSSEDecoder(strings.NewReader(body)))
	if len(events) != 1 {
		t.Fatalf("decoded %d events, want 1: %+v", len(events), events)
	}
}

func TestSSENamedEvents(t *testing.T) {
	body := "event: message_start\ndata: {\"type\":\"message_start\"}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

	events := collectEvents(wire.NewSSEDecoder(strings.NewReader(body)))
	if len(events) != 2 {
		t.Fatalf("decoded %d events, want 2", len(events))
	}
	if events[0].Event != "message_start" || events[1].Event != "message_stop" {
		t.Errorf("event names = %q, %q", events[0].Event, events[1].Event)
	}
}

// TestSSEMultiLineData: the grammar joins repeated data fields with a newline.
func TestSSEMultiLineData(t *testing.T) {
	body := "data: {\"a\":\ndata: 1}\n\n"

	events := collectEvents(wire.NewSSEDecoder(strings.NewReader(body)))
	if len(events) != 1 {
		t.Fatalf("decoded %d events, want 1", len(events))
	}
	if events[0].Data != "{\"a\":\n1}" {
		t.Errorf("Data = %q", events[0].Data)
	}
	decoded, ok := events[0].JSON()
	if !ok || decoded["a"] != float64(1) {
		t.Errorf("JSON() = %v, %v", decoded, ok)
	}
}

// TestSSEIgnoresCommentsAndUnknownFields: a heartbeat comment or an id field
// must not end or corrupt the stream.
func TestSSEIgnoresCommentsAndUnknownFields(t *testing.T) {
	body := ": heartbeat\n\nid: 7\nretry: 100\ndata: {\"a\":1}\n\n"

	events := collectEvents(wire.NewSSEDecoder(strings.NewReader(body)))
	if len(events) != 1 {
		t.Fatalf("decoded %d events, want 1: %+v", len(events), events)
	}
	if events[0].Data != `{"a":1}` {
		t.Errorf("Data = %q", events[0].Data)
	}
}

// TestSSEHandlesCRLF: some proxies rewrite line endings.
func TestSSEHandlesCRLF(t *testing.T) {
	body := "data: {\"a\":1}\r\n\r\ndata: {\"a\":2}\r\n\r\n"

	events := collectEvents(wire.NewSSEDecoder(strings.NewReader(body)))
	if len(events) != 2 {
		t.Fatalf("decoded %d events, want 2: %+v", len(events), events)
	}
}

// TestSSENoSpaceAfterColon: the space after the colon is optional.
func TestSSENoSpaceAfterColon(t *testing.T) {
	events := collectEvents(wire.NewSSEDecoder(strings.NewReader("data:{\"a\":1}\n\n")))
	if len(events) != 1 || events[0].Data != `{"a":1}` {
		t.Fatalf("events = %+v", events)
	}
}

// TestSSEFinalEventWithoutTrailingBlankLine: a producer that closes the
// connection mid-frame still owes us the last event.
func TestSSEFinalEventWithoutTrailingBlankLine(t *testing.T) {
	events := collectEvents(wire.NewSSEDecoder(strings.NewReader("data: {\"a\":1}\n")))
	if len(events) != 1 || events[0].Data != `{"a":1}` {
		t.Fatalf("events = %+v", events)
	}
}

func TestSSEMalformedJSONIsReportedByJSON(t *testing.T) {
	events := collectEvents(wire.NewSSEDecoder(strings.NewReader("data: not json\n\n")))
	if len(events) != 1 {
		t.Fatalf("decoded %d events, want 1", len(events))
	}
	if _, ok := events[0].JSON(); ok {
		t.Error("JSON() accepted a malformed payload")
	}
}

// failingReader yields one frame and then fails, standing in for a connection
// dropped mid-stream.
type failingReader struct {
	prefix string
	sent   bool
}

func (r *failingReader) Read(p []byte) (int, error) {
	if !r.sent {
		r.sent = true
		n := copy(p, r.prefix)
		return n, nil
	}
	return 0, errors.New("connection reset")
}

func TestSSEReportsReaderFailure(t *testing.T) {
	decoder := wire.NewSSEDecoder(&failingReader{prefix: "data: {\"a\":1}\n\n"})
	events := collectEvents(decoder)

	if len(events) != 1 {
		t.Fatalf("decoded %d events, want the one complete frame", len(events))
	}
	if decoder.Err() == nil {
		t.Error("Err() is nil after a reader failure")
	}
}

// TestSSEOversizedLineIsReportedNotPanicked bounds a hostile or broken stream:
// the decoder must stop with an error rather than grow without limit.
func TestSSEOversizedLineIsReportedNotPanicked(t *testing.T) {
	huge := "data: " + strings.Repeat("x", 4<<20) + "\n\n"

	done := make(chan struct{})
	var decoder *wire.SSEDecoder
	go func() {
		defer close(done)
		decoder = wire.NewSSEDecoder(strings.NewReader(huge))
		collectEvents(decoder)
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("decoding an oversized line did not terminate")
	}
	if decoder.Err() == nil {
		t.Error("an oversized line should be reported through Err()")
	}
}

func TestSSEEmptyStream(t *testing.T) {
	decoder := wire.NewSSEDecoder(strings.NewReader(""))
	if events := collectEvents(decoder); len(events) != 0 {
		t.Errorf("events = %+v, want none", events)
	}
	if decoder.Err() != nil {
		t.Errorf("Err() = %v, want nil for a clean empty stream", decoder.Err())
	}
}

// TestSSENextIsIdempotentAfterEnd: a consumer that keeps polling past the end
// must not block or panic.
func TestSSENextIsIdempotentAfterEnd(t *testing.T) {
	decoder := wire.NewSSEDecoder(strings.NewReader("data: [DONE]\n\n"))
	for i := 0; i < 4; i++ {
		if _, ok := decoder.Next(); ok {
			t.Fatalf("Next() returned an event on call %d after the stream ended", i)
		}
	}
}

var _ io.Reader = (*failingReader)(nil)
