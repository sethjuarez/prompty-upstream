package wire

import (
	"errors"
	"fmt"
	"net/url"
	"strings"
)

// Redacted is the placeholder substituted for any credential that would
// otherwise reach a log line, an error message or a test failure.
const Redacted = "[REDACTED]"

// ErrProvider is the sentinel behind every provider transport or protocol
// failure raised by a provider package.
var ErrProvider = errors.New("prompty: provider request failed")

// ProviderError carries a provider failure with the diagnostic context a host
// needs, and nothing it must not have.
//
// Construct it through NewProviderError so the credential-bearing fields are
// scrubbed exactly once, at the boundary. Status is the HTTP status when the
// provider answered, and zero when the failure happened before a response.
type ProviderError struct {
	Provider string
	Op       string
	Status   int
	// Endpoint is the request URL with its query string stripped of secrets.
	Endpoint string
	// Body is the provider's response body, truncated and scrubbed.
	Body string
	Err  error
}

func (e *ProviderError) Error() string {
	var b strings.Builder
	b.WriteString("prompty/")
	b.WriteString(e.Provider)
	if e.Op != "" {
		b.WriteString(": ")
		b.WriteString(e.Op)
	}
	if e.Status != 0 {
		fmt.Fprintf(&b, " failed with HTTP %d", e.Status)
	} else {
		b.WriteString(" failed")
	}
	if e.Endpoint != "" {
		fmt.Fprintf(&b, " (%s)", e.Endpoint)
	}
	if e.Body != "" {
		fmt.Fprintf(&b, ": %s", e.Body)
	}
	if e.Err != nil {
		fmt.Fprintf(&b, ": %v", e.Err)
	}
	return b.String()
}

func (e *ProviderError) Unwrap() []error {
	if e.Err == nil {
		return []error{ErrProvider}
	}
	return []error{ErrProvider, e.Err}
}

// maxDiagnosticBody bounds how much of a provider response body is retained in
// an error. Provider errors are small; a huge body in an error string is a log
// hazard, not a diagnostic.
const maxDiagnosticBody = 2048

// NewProviderError builds a scrubbed provider error. endpoint and body are
// passed through the redactors, so callers may hand over the raw request URL
// and the raw response body without auditing them first.
func NewProviderError(provider, op string, status int, endpoint, body string, err error) *ProviderError {
	return &ProviderError{
		Provider: provider,
		Op:       op,
		Status:   status,
		Endpoint: RedactURL(endpoint),
		Body:     RedactSecrets(truncate(body, maxDiagnosticBody)),
		Err:      err,
	}
}

func truncate(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	return s[:limit] + "… (truncated)"
}

// secretQueryKeys are query parameters that carry credentials. Azure and
// several proxies accept a key in the query string, so a bare URL is not
// automatically safe to log.
var secretQueryKeys = map[string]bool{
	"api-key":       true,
	"api_key":       true,
	"apikey":        true,
	"key":           true,
	"access_token":  true,
	"code":          true,
	"sig":           true,
	"signature":     true,
	"subscription":  true,
	"client_secret": true,
}

// RedactURL returns a URL safe to put in a log line or an error: userinfo is
// dropped and credential-bearing query parameters are masked. A string that
// does not parse as a URL is returned scrubbed by RedactSecrets instead of
// being echoed verbatim.
func RedactURL(raw string) string {
	if raw == "" {
		return ""
	}
	parsed, err := url.Parse(raw)
	// url.Parse accepts almost any string as a relative reference, so a value
	// that is not really a URL parses "successfully" and would be echoed
	// verbatim. Anything without a scheme and a host is treated as opaque text
	// and scrubbed instead.
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return RedactSecrets(raw)
	}

	if parsed.User != nil {
		parsed.User = url.User(Redacted)
	}

	if query := parsed.Query(); len(query) > 0 {
		changed := false
		for key := range query {
			if secretQueryKeys[strings.ToLower(key)] {
				query.Set(key, Redacted)
				changed = true
			}
		}
		if changed {
			parsed.RawQuery = query.Encode()
		}
	}
	// Belt and braces: a credential smuggled somewhere this function does not
	// model still gets scrubbed. Neither redaction pattern occurs in a
	// well-formed URL, so this cannot over-mask a real endpoint.
	return RedactSecrets(parsed.String())
}

// secretJSONKeys are response/body field names whose values must never survive
// into a diagnostic.
var secretJSONKeys = []string{
	"api_key", "apiKey", "api-key", "authorization", "access_token",
	"accessToken", "client_secret", "clientSecret", "refresh_token",
	"refreshToken", "x-api-key",
}

// RedactSecrets masks credential-shaped values in free-form text.
//
// It covers the two shapes that actually leak: a JSON field whose name is
// credential-like, and a bearer token in an Authorization header echoed back by
// a proxy. It is a defence in depth measure — the providers here never place a
// key in a message to begin with — so it errs toward masking.
func RedactSecrets(text string) string {
	if text == "" {
		return text
	}
	for _, key := range secretJSONKeys {
		text = redactJSONField(text, key)
	}
	return redactBearer(text)
}

// asciiLower lowercases only ASCII letters.
//
// strings.ToLower must not be used for the offset-based scanning below:
// lowercasing can change a rune's byte length (U+212A KELVIN SIGN is 3 bytes and
// lowercases to a 1-byte 'k'; U+0130 is 2 bytes and lowercases to 3). An index
// found in a strings.ToLower copy would then be applied to a string of a
// different length, slicing mid-rune or out of range — a panic in a host process
// triggered by nothing more than a provider response body containing one of
// those characters. Folding ASCII only keeps every byte offset identical.
func asciiLower(s string) string {
	var folded []byte
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < 'A' || c > 'Z' {
			continue
		}
		if folded == nil {
			folded = []byte(s)
		}
		folded[i] = c + ('a' - 'A')
	}
	if folded == nil {
		return s
	}
	return string(folded)
}

// redactJSONField rewrites `"key": "value"` to `"key": "[REDACTED]"`, matching
// the key case-insensitively and tolerating arbitrary whitespace.
func redactJSONField(text, key string) string {
	lower := asciiLower(text)
	needle := `"` + asciiLower(key) + `"`

	var out strings.Builder
	cursor := 0
	for {
		idx := strings.Index(lower[cursor:], needle)
		if idx < 0 {
			break
		}
		start := cursor + idx
		valueStart, valueEnd, ok := jsonStringValueSpan(text, start+len(needle))
		if !ok {
			out.WriteString(text[cursor : start+len(needle)])
			cursor = start + len(needle)
			continue
		}
		out.WriteString(text[cursor:valueStart])
		out.WriteString(Redacted)
		cursor = valueEnd
	}
	out.WriteString(text[cursor:])
	return out.String()
}

// jsonStringValueSpan locates the body of the JSON string value that follows a
// key at index i, returning the offsets between the quotes.
func jsonStringValueSpan(text string, i int) (int, int, bool) {
	for i < len(text) && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r') {
		i++
	}
	if i >= len(text) || text[i] != ':' {
		return 0, 0, false
	}
	i++
	for i < len(text) && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r') {
		i++
	}
	if i >= len(text) || text[i] != '"' {
		return 0, 0, false
	}
	start := i + 1
	for j := start; j < len(text); j++ {
		if text[j] == '\\' {
			j++
			continue
		}
		if text[j] == '"' {
			return start, j, true
		}
	}
	return 0, 0, false
}

// redactBearer masks the token in any "Bearer <token>" run.
func redactBearer(text string) string {
	const prefix = "Bearer "
	lower := asciiLower(text)
	lowerPrefix := asciiLower(prefix)

	var out strings.Builder
	cursor := 0
	for {
		idx := strings.Index(lower[cursor:], lowerPrefix)
		if idx < 0 {
			break
		}
		start := cursor + idx + len(prefix)
		end := start
		for end < len(text) && !isTokenTerminator(text[end]) {
			end++
		}
		if end == start {
			out.WriteString(text[cursor:start])
			cursor = start
			continue
		}
		out.WriteString(text[cursor:start])
		out.WriteString(Redacted)
		cursor = end
	}
	out.WriteString(text[cursor:])
	return out.String()
}

func isTokenTerminator(c byte) bool {
	switch c {
	case ' ', '\t', '\n', '\r', '"', ',', '}', ']':
		return true
	default:
		return false
	}
}
