package wire_test

import (
	"strings"
	"testing"

	wire "prompty/wire"
)

// TestRedactURLMasksQueryCredentials covers the shapes that actually leak: an
// Azure key in the query string and userinfo in the authority.
func TestRedactURLMasksQueryCredentials(t *testing.T) {
	cases := []struct {
		name   string
		input  string
		secret string
	}{
		{
			name:   "azure api-key query parameter",
			input:  "https://r.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-01-01&api-key=SECRET",
			secret: "SECRET",
		},
		{"lowercase key parameter", "https://proxy.example.com/v1/chat?key=SECRET", "SECRET"},
		{"access token parameter", "https://proxy.example.com/v1/chat?access_token=SECRET", "SECRET"},
		{"client secret parameter", "https://proxy.example.com/token?client_secret=SECRET", "SECRET"},
		{"userinfo in the authority", "https://user:SECRET@proxy.example.com/v1/chat", "SECRET"},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			got := wire.RedactURL(testCase.input)
			if strings.Contains(got, testCase.secret) {
				t.Errorf("RedactURL leaked the credential: %s", got)
			}
			if !strings.Contains(got, "REDACTED") {
				t.Errorf("RedactURL did not mark the redaction: %s", got)
			}
		})
	}
}

// TestRedactURLPreservesNonSecrets: over-redaction destroys the diagnostic.
func TestRedactURLPreservesNonSecrets(t *testing.T) {
	input := "https://r.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-01-01"
	got := wire.RedactURL(input)

	for _, want := range []string{"r.openai.azure.com", "deployments/gpt-4o", "api-version=2024-01-01"} {
		if !strings.Contains(got, want) {
			t.Errorf("RedactURL dropped %q from %s", want, got)
		}
	}
}

func TestRedactURLHandlesUnparseableInput(t *testing.T) {
	if got := wire.RedactURL(""); got != "" {
		t.Errorf("RedactURL(\"\") = %q, want empty", got)
	}
	// A value that is not a URL must still be scrubbed rather than echoed.
	got := wire.RedactURL("Bearer sk-not-a-url")
	if strings.Contains(got, "sk-not-a-url") {
		t.Errorf("RedactURL echoed a credential-shaped non-URL: %s", got)
	}
}

func TestRedactSecretsMasksJSONFields(t *testing.T) {
	cases := []string{
		`{"api_key":"SECRET"}`,
		`{"apiKey":"SECRET"}`,
		`{"api-key": "SECRET"}`,
		`{"Authorization":"SECRET"}`,
		`{"access_token"  :  "SECRET"}`,
		`{"x-api-key":"SECRET","other":"kept"}`,
		`{"refreshToken":"SECRET"}`,
	}

	for _, input := range cases {
		got := wire.RedactSecrets(input)
		if strings.Contains(got, "SECRET") {
			t.Errorf("RedactSecrets(%s) leaked the credential: %s", input, got)
		}
	}

	if got := wire.RedactSecrets(`{"x-api-key":"SECRET","other":"kept"}`); !strings.Contains(got, "kept") {
		t.Errorf("RedactSecrets removed a non-secret field: %s", got)
	}
}

func TestRedactSecretsMasksBearerTokens(t *testing.T) {
	cases := []string{
		"Authorization: Bearer sk-abc123",
		`{"header":"Bearer sk-abc123"}`,
		"failed with Bearer sk-abc123, retrying",
	}

	for _, input := range cases {
		got := wire.RedactSecrets(input)
		if strings.Contains(got, "sk-abc123") {
			t.Errorf("RedactSecrets(%q) leaked the token: %s", input, got)
		}
		if !strings.Contains(got, wire.Redacted) {
			t.Errorf("RedactSecrets(%q) did not mark the redaction: %s", input, got)
		}
	}
}

// TestRedactSecretsMasksEveryOccurrence guards the scanning loop: a body that
// repeats a credential must not have only the first one masked.
func TestRedactSecretsMasksEveryOccurrence(t *testing.T) {
	input := `{"api_key":"SECRET1","nested":{"api_key":"SECRET2"},"h":"Bearer SECRET3"}`
	got := wire.RedactSecrets(input)

	for _, secret := range []string{"SECRET1", "SECRET2", "SECRET3"} {
		if strings.Contains(got, secret) {
			t.Errorf("RedactSecrets left %s in %s", secret, got)
		}
	}
}

func TestRedactSecretsLeavesCleanTextAlone(t *testing.T) {
	input := `{"error":{"message":"model not found","type":"invalid_request_error"}}`
	if got := wire.RedactSecrets(input); got != input {
		t.Errorf("RedactSecrets altered clean text\ngot  %s\nwant %s", got, input)
	}
}

// TestRedactSecretsHandlesEscapedQuotes: a naive scanner would stop at the
// escaped quote and mask the wrong span.
func TestRedactSecretsHandlesEscapedQuotes(t *testing.T) {
	input := `{"api_key":"SEC\"RET","message":"kept"}`
	got := wire.RedactSecrets(input)

	if strings.Contains(got, "SEC") {
		t.Errorf("RedactSecrets leaked part of the credential: %s", got)
	}
	if !strings.Contains(got, "kept") {
		t.Errorf("RedactSecrets consumed the following field: %s", got)
	}
}

func TestProviderErrorRendersWithoutSecrets(t *testing.T) {
	err := wire.NewProviderError("openai", "execute", 401,
		"https://r.openai.azure.com/openai/deployments/d/chat/completions?api-key=SECRET",
		`{"error":{"message":"unauthorized"},"api_key":"SECRET"}`, nil)

	rendered := err.Error()
	if strings.Contains(rendered, "SECRET") {
		t.Errorf("ProviderError leaked a credential: %s", rendered)
	}
	for _, want := range []string{"openai", "execute", "401", "unauthorized"} {
		if !strings.Contains(rendered, want) {
			t.Errorf("ProviderError dropped the diagnostic %q: %s", want, rendered)
		}
	}
}

// TestProviderErrorBodyIsBounded keeps a huge provider response out of a log.
func TestProviderErrorBodyIsBounded(t *testing.T) {
	err := wire.NewProviderError("openai", "execute", 500, "https://api.openai.com/v1/chat/completions",
		strings.Repeat("x", 1<<20), nil)

	if len(err.Error()) > 4096 {
		t.Errorf("ProviderError rendered %d bytes; diagnostic bodies must be bounded", len(err.Error()))
	}
	if !strings.Contains(err.Error(), "truncated") {
		t.Error("a truncated body should say so")
	}
}
