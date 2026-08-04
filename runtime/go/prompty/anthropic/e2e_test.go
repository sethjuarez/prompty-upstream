//go:build integration

package anthropic_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	prompty "prompty"
	anthropic "prompty/anthropic"
	model "prompty/model"
)

func TestAnthropicE2EChatAndStream(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	modelID := os.Getenv("ANTHROPIC_MODEL")
	if apiKey == "" || modelID == "" {
		t.Skip("ANTHROPIC_API_KEY and ANTHROPIC_MODEL are required")
	}

	provider := anthropic.Provider
	agent := model.Prompty{
		Name:  "anthropic-e2e",
		Model: model.Model{Id: modelID, Provider: &provider},
	}
	messages := []model.Message{model.NewUserMessage("Reply with exactly: pong")}
	executor := anthropic.NewExecutor()
	processor := anthropic.NewProcessor()

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
}
