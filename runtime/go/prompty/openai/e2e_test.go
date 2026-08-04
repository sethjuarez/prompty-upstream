//go:build integration

package openai_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	prompty "prompty"
	model "prompty/model"
	openai "prompty/openai"
)

func TestOpenAIE2EChatStreamStructuredAndToolTurn(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	modelID := os.Getenv("OPENAI_MODEL")
	if apiKey == "" || modelID == "" {
		t.Skip("OPENAI_API_KEY and OPENAI_MODEL are required")
	}

	provider := openai.ProviderOpenAI
	agent := model.Prompty{
		Name:  "openai-e2e",
		Model: model.Model{Id: modelID, Provider: &provider},
	}
	messages := []model.Message{model.NewUserMessage("Reply with exactly: pong")}
	executor := openai.NewExecutor(openai.DialectOpenAI)
	processor := openai.NewProcessor()

	t.Run("chat", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		raw, err := executor.ExecuteContext(ctx, agent, messages)
		if err != nil {
			t.Fatalf("ExecuteContext: %v", err)
		}
		result, err := processor.ProcessContext(ctx, agent, raw)
		if err != nil {
			t.Fatalf("ProcessContext: %v", err)
		}
		text, ok := result.(string)
		if !ok {
			t.Fatalf("chat result type = %T, want string", result)
		}
		if !strings.Contains(strings.ToLower(text), "pong") {
			t.Fatalf("chat result = %#v, want pong", result)
		}
	})

	t.Run("stream", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		raw, err := executor.ExecuteStreamContext(ctx, agent, messages)
		if err != nil {
			t.Fatalf("ExecuteStreamContext: %v", err)
		}
		processed, err := processor.ProcessStreamContext(ctx, raw)
		if err != nil {
			t.Fatalf("ProcessStreamContext: %v", err)
		}
		stream, ok := processed.(*prompty.Stream)
		if !ok {
			t.Fatalf("processed stream type = %T, want *prompty.Stream", processed)
		}
		var text strings.Builder
		for chunk := range stream.Chunks() {
			if value, ok := prompty.ChunkText(chunk); ok {
				text.WriteString(value)
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if !strings.Contains(strings.ToLower(text.String()), "pong") {
			t.Fatalf("stream result = %q, want pong", text.String())
		}
	})

	t.Run("structured", func(t *testing.T) {
		required := true
		structuredAgent := agent
		structuredAgent.Outputs = []interface{}{
			model.Property{Name: "answer", Kind: "string", Required: &required},
		}
		structuredMessages := []model.Message{
			model.NewUserMessage(`Return an object whose "answer" field is exactly "pong".`),
		}
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		raw, err := executor.ExecuteContext(ctx, structuredAgent, structuredMessages)
		if err != nil {
			t.Fatalf("ExecuteContext: %v", err)
		}
		result, err := processor.ProcessContext(ctx, structuredAgent, raw)
		if err != nil {
			t.Fatalf("ProcessContext: %v", err)
		}
		object, ok := result.(map[string]interface{})
		if !ok || object["answer"] != "pong" {
			t.Fatalf("structured result = %#v, want answer=pong", result)
		}
	})

	t.Run("tool_turn", func(t *testing.T) {
		description := "Return the fixed magic number."
		toolAgent := agent
		toolAgent.Tools = []interface{}{
			model.FunctionTool{
				Name:        "get_magic_number",
				Kind:        "function",
				Description: &description,
				Parameters:  []interface{}{},
			},
		}
		tools := prompty.NewToolRegistry()
		tools.RegisterText("get_magic_number", func(context.Context, map[string]interface{}) (string, error) {
			return "42", nil
		})
		toolMessages := []model.Message{
			model.NewUserMessage("Call get_magic_number, then reply with the number it returns."),
		}
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		result, err := prompty.RunMessages(ctx, toolAgent, toolMessages, prompty.RunOptions{
			Tools:     tools,
			Executor:  executor,
			Processor: processor,
		})
		if err != nil {
			t.Fatalf("RunMessages: %v", err)
		}
		if len(result.Dispatches) != 1 || result.Dispatches[0].Call.Name != "get_magic_number" {
			t.Fatalf("dispatches = %#v, want one get_magic_number call", result.Dispatches)
		}
		if !strings.Contains(result.Text(), "42") {
			t.Fatalf("tool result = %q, want 42", result.Text())
		}
	})
}

func TestFoundryE2EChatStreamStructuredAndToolTurn(t *testing.T) {
	endpoint := os.Getenv("FOUNDRY_PROJECT_ENDPOINT")
	if endpoint == "" {
		t.Skip("FOUNDRY_PROJECT_ENDPOINT is required")
	}
	tokenProvider, err := openai.NewDefaultAzureCredentialTokenProvider()
	if err != nil {
		t.Fatalf("NewDefaultAzureCredentialTokenProvider: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	token, err := tokenProvider.Token(ctx)
	if err != nil {
		t.Fatalf("acquire Foundry token: %v", err)
	}
	modelID := firstFoundryDeployment(t, ctx, endpoint, token)
	t.Setenv(openai.EnvInferenceCredential, "")

	provider := openai.ProviderFoundry
	connection := map[string]interface{}{"kind": "foundry", "endpoint": endpoint}
	agent := model.Prompty{
		Name:  "foundry-e2e",
		Model: model.Model{Id: modelID, Provider: &provider, Connection: connection},
	}
	executor := &openai.Executor{
		Dialect:       openai.DialectFoundry,
		TokenProvider: tokenProvider,
	}
	processor := openai.NewProcessor()
	messages := []model.Message{model.NewUserMessage("Reply with exactly: pong")}

	t.Run("chat", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		raw, err := executor.ExecuteContext(ctx, agent, messages)
		if err != nil {
			t.Fatalf("ExecuteContext: %v", err)
		}
		result, err := processor.ProcessContext(ctx, agent, raw)
		if err != nil {
			t.Fatalf("ProcessContext: %v", err)
		}
		text, ok := result.(string)
		if !ok || !strings.Contains(strings.ToLower(text), "pong") {
			t.Fatalf("chat result = %#v, want pong", result)
		}
	})

	t.Run("stream", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		raw, err := executor.ExecuteStreamContext(ctx, agent, messages)
		if err != nil {
			t.Fatalf("ExecuteStreamContext: %v", err)
		}
		processed, err := processor.ProcessStreamContext(ctx, raw)
		if err != nil {
			t.Fatalf("ProcessStreamContext: %v", err)
		}
		stream, ok := processed.(*prompty.Stream)
		if !ok {
			t.Fatalf("processed stream type = %T, want *prompty.Stream", processed)
		}
		var text strings.Builder
		for chunk := range stream.Chunks() {
			if value, ok := prompty.ChunkText(chunk); ok {
				text.WriteString(value)
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if !strings.Contains(strings.ToLower(text.String()), "pong") {
			t.Fatalf("stream result = %q, want pong", text.String())
		}
	})

	t.Run("structured", func(t *testing.T) {
		required := true
		structuredAgent := agent
		structuredAgent.Outputs = []interface{}{
			model.Property{Name: "answer", Kind: "string", Required: &required},
		}
		structuredMessages := []model.Message{
			model.NewUserMessage(`Return an object whose "answer" field is exactly "pong".`),
		}
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		raw, err := executor.ExecuteContext(ctx, structuredAgent, structuredMessages)
		if err != nil {
			t.Fatalf("ExecuteContext: %v", err)
		}
		result, err := processor.ProcessContext(ctx, structuredAgent, raw)
		if err != nil {
			t.Fatalf("ProcessContext: %v", err)
		}
		object, ok := result.(map[string]interface{})
		if !ok || object["answer"] != "pong" {
			t.Fatalf("structured result = %#v, want answer=pong", result)
		}
	})

	t.Run("tool_turn", func(t *testing.T) {
		description := "Return the fixed magic number."
		toolAgent := agent
		toolAgent.Tools = []interface{}{
			model.FunctionTool{
				Name:        "get_magic_number",
				Kind:        "function",
				Description: &description,
				Parameters:  []interface{}{},
			},
		}
		tools := prompty.NewToolRegistry()
		tools.RegisterText("get_magic_number", func(context.Context, map[string]interface{}) (string, error) {
			return "42", nil
		})
		toolMessages := []model.Message{
			model.NewUserMessage("Call get_magic_number, then reply with the number it returns."),
		}
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		result, err := prompty.RunMessages(ctx, toolAgent, toolMessages, prompty.RunOptions{
			Tools:     tools,
			Executor:  executor,
			Processor: processor,
		})
		if err != nil {
			t.Fatalf("RunMessages: %v", err)
		}
		if len(result.Dispatches) != 1 || result.Dispatches[0].Call.Name != "get_magic_number" {
			t.Fatalf("dispatches = %#v, want one get_magic_number call", result.Dispatches)
		}
		if !strings.Contains(result.Text(), "42") {
			t.Fatalf("tool result = %q, want 42", result.Text())
		}
	})
}

func firstFoundryDeployment(t *testing.T, ctx context.Context, endpoint, token string) string {
	t.Helper()
	url := strings.TrimSuffix(endpoint, "/") + "/deployments?api-version=v1"
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		t.Fatalf("build Foundry deployment request: %v", err)
	}
	request.Header.Set("Authorization", "Bearer "+token)
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("list Foundry deployments: %v", err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		t.Fatalf("read Foundry deployments: %v", err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		t.Fatalf("list Foundry deployments failed with HTTP %d: %s", response.StatusCode, string(body))
	}
	var payload struct {
		Value []map[string]interface{} `json:"value"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode Foundry deployments: %v", err)
	}
	for _, deployment := range payload.Value {
		for _, field := range []string{"name", "id", "modelName"} {
			if value, ok := deployment[field].(string); ok && value != "" {
				t.Logf("using Foundry deployment %s", value)
				return value
			}
		}
	}
	t.Fatal("Foundry project returned no invokable deployments")
	return ""
}
