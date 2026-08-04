package openai_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	openai "prompty/openai"
)

type tokenProviderFunc func(context.Context) (string, error)

func (f tokenProviderFunc) Token(ctx context.Context) (string, error) {
	return f(ctx)
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func TestExecutorAcquiresFoundryTokenWhenCredentialIsAbsent(t *testing.T) {
	agent := testAgent(t, map[string]interface{}{
		"kind": "foundry", "endpoint": "https://res.services.ai.azure.com/api/projects/p",
	}, nil)
	acquired := 0
	executor := &openai.Executor{
		Dialect: openai.DialectFoundry,
		Env:     staticEnv(nil),
		TokenProvider: tokenProviderFunc(func(context.Context) (string, error) {
			acquired++
			return "entra-token", nil
		}),
		Client: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			if got := request.Header.Get("Authorization"); got != "Bearer entra-token" {
				t.Errorf("Authorization = %q, want bearer token", got)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(`{"choices":[]}`)),
				Header:     make(http.Header),
			}, nil
		})},
	}

	if _, err := executor.ExecuteContext(context.Background(), agent, userMessage("hi")); err != nil {
		t.Fatalf("ExecuteContext: %v", err)
	}
	if acquired != 1 {
		t.Fatalf("token acquisitions = %d, want 1", acquired)
	}
}

func TestExecutorPrefersExplicitFoundryToken(t *testing.T) {
	agent := testAgent(t, map[string]interface{}{
		"kind": "foundry", "endpoint": "https://res.services.ai.azure.com/api/projects/p", "apiKey": "explicit-token",
	}, nil)
	agent.Model.Connection = map[string]interface{}{
		"kind": "foundry", "endpoint": "https://res.services.ai.azure.com/api/projects/p", "apiKey": "explicit-token",
	}
	executor := &openai.Executor{
		Dialect: openai.DialectFoundry,
		Env:     staticEnv(nil),
		TokenProvider: tokenProviderFunc(func(context.Context) (string, error) {
			t.Fatal("TokenProvider must not be called when the connection has a token")
			return "", nil
		}),
		Client: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			if got := request.Header.Get("Authorization"); got != "Bearer explicit-token" {
				t.Errorf("Authorization = %q, want explicit token", got)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(`{"choices":[]}`)),
				Header:     make(http.Header),
			}, nil
		})},
	}

	if _, err := executor.ExecuteContext(context.Background(), agent, userMessage("hi")); err != nil {
		t.Fatalf("ExecuteContext: %v", err)
	}
}

func TestExecutorPrefersEnvironmentFoundryToken(t *testing.T) {
	agent := testAgent(t, map[string]interface{}{
		"kind": "foundry", "endpoint": "https://res.services.ai.azure.com/api/projects/p",
	}, nil)
	executor := &openai.Executor{
		Dialect: openai.DialectFoundry,
		Env:     staticEnv(map[string]string{openai.EnvInferenceCredential: "environment-token"}),
		TokenProvider: tokenProviderFunc(func(context.Context) (string, error) {
			t.Fatal("TokenProvider must not be called when the environment has a token")
			return "", nil
		}),
		Client: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			if got := request.Header.Get("Authorization"); got != "Bearer environment-token" {
				t.Errorf("Authorization = %q, want environment token", got)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(`{"choices":[]}`)),
				Header:     make(http.Header),
			}, nil
		})},
	}

	if _, err := executor.ExecuteContext(context.Background(), agent, userMessage("hi")); err != nil {
		t.Fatalf("ExecuteContext: %v", err)
	}
}

func TestExecutorSurfacesFoundryTokenFailureWithoutLeakingSecrets(t *testing.T) {
	agent := testAgent(t, map[string]interface{}{
		"kind": "foundry", "endpoint": "https://res.services.ai.azure.com/api/projects/p",
	}, nil)
	executor := &openai.Executor{
		Dialect: openai.DialectFoundry,
		Env:     staticEnv(nil),
		TokenProvider: tokenProviderFunc(func(context.Context) (string, error) {
			return "", errors.New(`credential failed with {"access_token":"secret-value"}`)
		}),
	}

	_, err := executor.ExecuteContext(context.Background(), agent, userMessage("hi"))
	if err == nil {
		t.Fatal("expected token acquisition error")
	}
	if !strings.Contains(err.Error(), "acquire Entra ID token") {
		t.Errorf("error = %v, want acquisition operation", err)
	}
	if strings.Contains(err.Error(), "secret-value") {
		t.Errorf("error leaked provider secret: %v", err)
	}
}

func TestExecutorReturnsCancellationFromFoundryTokenProvider(t *testing.T) {
	agent := testAgent(t, map[string]interface{}{
		"kind": "foundry", "endpoint": "https://res.services.ai.azure.com/api/projects/p",
	}, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	executor := &openai.Executor{
		Dialect: openai.DialectFoundry,
		Env:     staticEnv(nil),
		TokenProvider: tokenProviderFunc(func(ctx context.Context) (string, error) {
			return "", ctx.Err()
		}),
	}

	_, err := executor.ExecuteContext(ctx, agent, userMessage("hi"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
}
