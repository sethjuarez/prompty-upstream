package wire

import (
	"fmt"
	"io"
)

// DefaultMaxResponseBytes bounds buffered provider responses to 64 MiB.
const DefaultMaxResponseBytes int64 = 64 << 20

// ReadResponseBody buffers at most max bytes and reports oversized responses
// instead of silently truncating them. A non-positive max uses the default.
func ReadResponseBody(reader io.Reader, max int64) ([]byte, error) {
	if max <= 0 {
		max = DefaultMaxResponseBytes
	}
	body, err := io.ReadAll(io.LimitReader(reader, max+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > max {
		return nil, fmt.Errorf("provider response exceeds %d bytes", max)
	}
	return body, nil
}
