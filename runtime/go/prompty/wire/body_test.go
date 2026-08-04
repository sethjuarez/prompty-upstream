package wire

import (
	"strings"
	"testing"
)

func TestReadResponseBodyRejectsOversizedBody(t *testing.T) {
	_, err := ReadResponseBody(strings.NewReader("12345"), 4)
	if err == nil {
		t.Fatal("ReadResponseBody accepted an oversized response")
	}
}

func TestReadResponseBodyAcceptsExactLimit(t *testing.T) {
	body, err := ReadResponseBody(strings.NewReader("1234"), 4)
	if err != nil {
		t.Fatalf("ReadResponseBody returned an error: %v", err)
	}
	if string(body) != "1234" {
		t.Fatalf("ReadResponseBody returned %q, want %q", body, "1234")
	}
}
