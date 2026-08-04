package wire_test

import (
	"strings"
	"testing"

	wire "prompty/wire"
)

// These are regressions for defects found in adversarial review of the initial
// implementation. Each one reproduces the original failure.

// TestRedactSecretsSurvivesNonASCIIText reproduces a panic and a corruption
// bug: the redactors folded the whole payload with strings.ToLower and applied
// the resulting byte offsets back to the original string. Lowercasing changes
// the byte length of some runes — U+212A KELVIN SIGN is three bytes and folds
// to a one-byte 'k', U+0130 is two bytes and folds to three — so the offsets
// diverged and the slice went out of range or landed mid-rune. Because the
// input is a provider response body, that is a remote-triggered panic.
func TestRedactSecretsSurvivesNonASCIIText(t *testing.T) {
	cases := []string{
		// U+212A KELVIN SIGN shrinks by two bytes when folded.
		"\u212A\u212A\u212A\u212A" + `{"api_key":"SECRET"}`,
		// U+0130 grows by one byte when folded.
		"\u0130\u0130\u0130" + `{"api_key":"SECRET"}`,
		// Mixed, with the credential ahead of the troublesome runes.
		`{"api_key":"SECRET","note":"` + strings.Repeat("\u212A", 32) + `"}`,
		`{"note":"` + strings.Repeat("\u0130", 32) + `","h":"Bearer SECRET"}`,
		// Emoji and other multi-byte runes that fold to themselves.
		`{"api_key":"SECRET","emoji":"🔑🔒🗝"}`,
	}

	for _, input := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("RedactSecrets panicked on %q: %v", input, r)
				}
			}()
			got := wire.RedactSecrets(input)
			if strings.Contains(got, "SECRET") {
				t.Errorf("RedactSecrets(%q) leaked the credential: %s", input, got)
			}
			if !utf8Valid(got) {
				t.Errorf("RedactSecrets(%q) produced invalid UTF-8: %q", input, got)
			}
		}()
	}
}

// TestRedactURLSurvivesNonASCIIText covers the same path through RedactURL.
func TestRedactURLSurvivesNonASCIIText(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("RedactURL panicked: %v", r)
		}
	}()

	got := wire.RedactURL("\u212A\u212A Bearer SECRET \u0130\u0130")
	if strings.Contains(got, "SECRET") {
		t.Errorf("RedactURL leaked the credential: %s", got)
	}
}

// TestRedactSecretsIsCaseInsensitiveForASCII confirms the ASCII-only folding
// still matches mixed-case field names.
func TestRedactSecretsIsCaseInsensitiveForASCII(t *testing.T) {
	for _, input := range []string{
		`{"API_KEY":"SECRET"}`,
		`{"Api-Key":"SECRET"}`,
		`{"AUTHORIZATION":"SECRET"}`,
		`{"h":"BEARER SECRET"}`,
		`{"h":"bearer SECRET"}`,
	} {
		if got := wire.RedactSecrets(input); strings.Contains(got, "SECRET") {
			t.Errorf("RedactSecrets(%s) leaked the credential: %s", input, got)
		}
	}
}

// TestSSEFinalNamedEventWithoutDataIsDelivered reproduces a dropped event: the
// end-of-stream path required a non-empty data field, so a producer that closed
// straight after a name-only frame — Anthropic's message_stop is exactly that —
// lost it, even though the identical frame inside the loop was delivered.
func TestSSEFinalNamedEventWithoutDataIsDelivered(t *testing.T) {
	decoder := wire.NewSSEDecoder(strings.NewReader("event: message_stop\n"))

	event, ok := decoder.Next()
	if !ok {
		t.Fatal("the final name-only event was dropped at end of stream")
	}
	if event.Event != "message_stop" {
		t.Errorf("Event = %q, want message_stop", event.Event)
	}
	if _, ok := decoder.Next(); ok {
		t.Error("the decoder produced an event past the end of the stream")
	}
}

// TestSSEFinalNamedEventMatchesInLoopBehaviour pins the two paths together.
func TestSSEFinalNamedEventMatchesInLoopBehaviour(t *testing.T) {
	withBlankLine := collectEvents(wire.NewSSEDecoder(strings.NewReader("event: message_stop\n\n")))
	withoutBlankLine := collectEvents(wire.NewSSEDecoder(strings.NewReader("event: message_stop\n")))

	if len(withBlankLine) != len(withoutBlankLine) {
		t.Errorf("a trailing blank line changed the event count: %d vs %d",
			len(withBlankLine), len(withoutBlankLine))
	}
}

func utf8Valid(s string) bool {
	for _, r := range s {
		if r == '\uFFFD' && !strings.Contains(s, "\uFFFD") {
			return false
		}
	}
	return true
}
