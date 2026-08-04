package openai_test

import (
	"strings"
	"testing"

	openai "prompty/openai"
)

// These are regressions for defects found in adversarial review of the initial
// implementation.

// TestConnectionDialectWinsOverRegistrationFallback reproduces a routing bug:
// the executor's dialect was treated as an override, so an agent registered
// under one provider key but carrying a connection of the other kind was shaped
// for the wrong endpoint — an agent with a Foundry connection registered under
// "azure" got the Azure deployment path and an api-key header, which the
// Foundry endpoint answers with a 404 or a 401.
func TestConnectionDialectWinsOverRegistrationFallback(t *testing.T) {
	cases := []struct {
		name       string
		fallback   openai.Dialect
		connection map[string]interface{}
		env        map[string]string
		wantURL    string
		wantHeader string
	}{
		{
			name:     "foundry connection under the azure fallback",
			fallback: openai.DialectAzure,
			connection: map[string]interface{}{
				"kind":     "foundry",
				"endpoint": "https://res.services.ai.azure.com/api/projects/p",
			},
			env:        map[string]string{openai.EnvInferenceCredential: "token"},
			wantURL:    "https://res.openai.azure.com/openai/v1/chat/completions",
			wantHeader: "Authorization",
		},
		{
			// The emitted Connection discriminator has no "azure" value; an
			// Azure OpenAI endpoint is declared as a key connection and is
			// recognised by its hostname.
			name:     "azure connection under the foundry fallback",
			fallback: openai.DialectFoundry,
			connection: map[string]interface{}{
				"kind": "key", "endpoint": "https://res.openai.azure.com", "apiKey": "azure-key",
			},
			wantURL:    "https://res.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=" + openai.DefaultAzureAPIVersion,
			wantHeader: "api-key",
		},
		{
			name:       "the fallback still applies when the connection says nothing",
			fallback:   openai.DialectAzure,
			connection: map[string]interface{}{"kind": "key", "endpoint": "https://gateway.internal", "apiKey": "k"},
			wantURL:    "https://gateway.internal/openai/deployments/gpt-4o-mini/chat/completions?api-version=" + openai.DefaultAzureAPIVersion,
			wantHeader: "api-key",
		},
		{
			name:       "no connection and no fallback is plain OpenAI",
			connection: nil,
			env:        map[string]string{openai.EnvOpenAIAPIKey: "sk"},
			wantURL:    "https://api.openai.com/v1/chat/completions",
			wantHeader: "Authorization",
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			agent := testAgent(t, testCase.connection, nil)
			request, err := openai.BuildProviderRequest(
				agent, userMessage("hi"), testCase.fallback, staticEnv(testCase.env), false)
			if err != nil {
				t.Fatalf("BuildProviderRequest: %v", err)
			}
			if request.URL != testCase.wantURL {
				t.Errorf("URL = %q, want %q", request.URL, testCase.wantURL)
			}
			if request.AuthHeader != testCase.wantHeader {
				t.Errorf("AuthHeader = %q, want %q", request.AuthHeader, testCase.wantHeader)
			}
		})
	}
}

// TestFoundryDoesNotSendAnAzureKeyAsABearerToken reproduces an auth bug: the
// Foundry branch fell back to AZURE_OPENAI_API_KEY and sent it as
// "Authorization: Bearer <key>". An Azure OpenAI key is not a bearer token, so
// that turned a plain misconfiguration into an opaque 401 from the provider
// instead of a clear local error naming the variable to set.
func TestFoundryDoesNotSendAnAzureKeyAsABearerToken(t *testing.T) {
	agent := testAgent(t, map[string]interface{}{
		"kind": "foundry", "endpoint": "https://res.services.ai.azure.com/api/projects/p",
	}, nil)

	_, err := openai.BuildProviderRequest(agent, userMessage("hi"), "",
		staticEnv(map[string]string{openai.EnvAzureAPIKey: "an-azure-api-key"}), false)
	if err == nil {
		t.Fatal("expected an error: an Azure API key is not a Foundry bearer token")
	}
	if !strings.Contains(err.Error(), openai.EnvInferenceCredential) {
		t.Errorf("error should name the credential to set, got: %v", err)
	}
	if strings.Contains(err.Error(), "an-azure-api-key") {
		t.Errorf("the error leaked the credential it rejected: %v", err)
	}

	// The supported Foundry credentials still work.
	request, err := openai.BuildProviderRequest(agent, userMessage("hi"), "",
		staticEnv(map[string]string{openai.EnvInferenceCredential: "foundry-token"}), false)
	if err != nil {
		t.Fatalf("BuildProviderRequest with a Foundry credential: %v", err)
	}
	if request.Header["Authorization"] != "Bearer foundry-token" {
		t.Errorf("Authorization = %q", request.Header["Authorization"])
	}
}

// TestResolveDialectIsCaseInsensitiveOnConnectionKind guards the ASCII folding
// used for the connection kind.
func TestResolveDialectIsCaseInsensitiveOnConnectionKind(t *testing.T) {
	for _, kind := range []string{"Foundry", "FOUNDRY", "foundry"} {
		got := openai.ResolveDialect("", map[string]interface{}{"kind": kind})
		if got != openai.DialectFoundry {
			t.Errorf("ResolveDialect(kind=%q) = %q, want foundry", kind, got)
		}
	}
	for _, kind := range []string{"Azure", "AZURE_OPENAI", "AzureOpenAI"} {
		got := openai.ResolveDialect("", map[string]interface{}{"kind": kind})
		if got != openai.DialectAzure {
			t.Errorf("ResolveDialect(kind=%q) = %q, want azure", kind, got)
		}
	}
}
