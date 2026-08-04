package wire

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
)

// SSEEvent is one decoded Server-Sent Event.
type SSEEvent struct {
	// Event is the `event:` field, empty when the producer omitted it. OpenAI's
	// chat stream omits it; the Responses and Anthropic streams rely on it.
	Event string
	// Data is the raw `data:` payload with the terminal sentinel removed.
	Data string
}

// JSON decodes the event payload into a generic map. It returns false when the
// payload is not a JSON object, which is how the terminal sentinel and any
// keep-alive comment are filtered out by callers.
func (e SSEEvent) JSON() (map[string]interface{}, bool) {
	if e.Data == "" {
		return nil, false
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal([]byte(e.Data), &decoded); err != nil {
		return nil, false
	}
	return decoded, true
}

// doneSentinel terminates an OpenAI-style stream. Anthropic terminates with a
// message_stop event instead and never sends this.
const doneSentinel = "[DONE]"

// SSEDecoder incrementally decodes a Server-Sent Event byte stream.
//
// It is deliberately a pull decoder over an io.Reader rather than a channel:
// the caller owns the goroutine and therefore owns cancellation, which is what
// keeps provider streams from leaking a goroutine when a consumer walks away.
type SSEDecoder struct {
	scanner *bufio.Scanner
	// pending accumulates the fields of the event currently being read.
	pendingEvent string
	pendingData  []string
	done         bool
	err          error
}

// maxSSELineBytes bounds a single SSE line. Provider deltas are small, but a
// malformed or hostile stream must not be able to grow the buffer without
// limit; 1 MiB is far above any real chunk and far below a memory problem.
const maxSSELineBytes = 1 << 20

// NewSSEDecoder returns a decoder reading events from r.
func NewSSEDecoder(r io.Reader) *SSEDecoder {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 8192), maxSSELineBytes)
	return &SSEDecoder{scanner: scanner}
}

// Next returns the next event. It reports false when the stream ends, either
// because the producer sent the terminal sentinel or because the reader was
// exhausted or failed; check Err to tell a clean end from a failure.
func (d *SSEDecoder) Next() (SSEEvent, bool) {
	if d.done {
		return SSEEvent{}, false
	}

	for d.scanner.Scan() {
		line := strings.TrimSuffix(d.scanner.Text(), "\r")

		// A blank line dispatches the accumulated event.
		if line == "" {
			event, ok := d.flush()
			if d.done {
				return SSEEvent{}, false
			}
			if ok {
				return event, true
			}
			continue
		}

		// Comments (heartbeats) carry no fields.
		if strings.HasPrefix(line, ":") {
			continue
		}

		field, value := splitSSEField(line)
		switch field {
		case "event":
			d.pendingEvent = value
		case "data":
			d.pendingData = append(d.pendingData, value)
		default:
			// id and retry are irrelevant to provider streams; ignore silently
			// so an unknown field never aborts a stream.
		}
	}

	if err := d.scanner.Err(); err != nil {
		d.err = err
	}

	// A producer that closed without a trailing blank line still owes us the
	// last event. An event carrying only a name and no data is still an event —
	// Anthropic's message_stop is exactly that — so it is dispatched on the
	// same terms as one inside the loop.
	event, ok := d.flush()
	d.done = true
	if ok {
		return event, true
	}
	return SSEEvent{}, false
}

// Err reports the reader or scanner failure that ended the stream, if any.
func (d *SSEDecoder) Err() error { return d.err }

func (d *SSEDecoder) flush() (SSEEvent, bool) {
	if len(d.pendingData) == 0 && d.pendingEvent == "" {
		return SSEEvent{}, false
	}
	event := SSEEvent{Event: d.pendingEvent, Data: strings.Join(d.pendingData, "\n")}
	d.pendingEvent = ""
	d.pendingData = nil

	if strings.TrimSpace(event.Data) == doneSentinel {
		d.done = true
		return SSEEvent{}, false
	}
	return event, true
}

// splitSSEField splits "field: value" per the SSE grammar, where a single
// leading space after the colon is part of the delimiter, not the value.
func splitSSEField(line string) (string, string) {
	colon := strings.IndexByte(line, ':')
	if colon < 0 {
		return line, ""
	}
	field := line[:colon]
	value := line[colon+1:]
	return field, strings.TrimPrefix(value, " ")
}
