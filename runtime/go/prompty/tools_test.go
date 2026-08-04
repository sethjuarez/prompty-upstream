package prompty_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	prompty "prompty"
	model "prompty/model"
	openai "prompty/openai"
)

func TestToolRegistryLifecycle(t *testing.T) {
	registry := prompty.NewToolRegistry()

	if registry.Has("a") {
		t.Error("a fresh registry should be empty")
	}
	registry.RegisterText("b", func(context.Context, map[string]interface{}) (string, error) { return "", nil })
	registry.RegisterText("a", func(context.Context, map[string]interface{}) (string, error) { return "", nil })

	if names := registry.Names(); len(names) != 2 || names[0] != "a" || names[1] != "b" {
		t.Errorf("Names() = %v, want sorted [a b]", names)
	}

	registry.Unregister("a")
	if registry.Has("a") {
		t.Error("Unregister did not remove the tool")
	}
}

func TestDispatchRunsTheTool(t *testing.T) {
	registry := prompty.NewToolRegistry()
	registry.RegisterText("echo", func(_ context.Context, args map[string]interface{}) (string, error) {
		return "got " + args["value"].(string), nil
	})

	dispatch, err := registry.Dispatch(context.Background(),
		model.ToolCall{Id: "1", Name: "echo", Arguments: `{"value":"hi"}`}, nil, nil)
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if dispatch.Text() != "got hi" {
		t.Errorf("result = %q", dispatch.Text())
	}
	if dispatch.Denied {
		t.Error("dispatch was marked denied")
	}
}

// TestDispatchUnknownToolIsFatal: the model invented a capability, so feeding
// it a synthetic result would teach it the call was legitimate.
func TestDispatchUnknownToolIsFatal(t *testing.T) {
	registry := prompty.NewToolRegistry()

	_, err := registry.Dispatch(context.Background(),
		model.ToolCall{Id: "1", Name: "missing", Arguments: "{}"}, nil, nil)
	if err == nil {
		t.Fatal("expected an error for an unregistered tool")
	}
	if err.Error() != "Tool not registered: missing" {
		t.Errorf("error = %q", err.Error())
	}
	if !errors.Is(err, prompty.ErrToolNotRegistered) || !errors.Is(err, prompty.ErrValue) {
		t.Errorf("error does not wrap the expected sentinels: %v", err)
	}
}

// TestDispatchToolErrorIsModelVisible: an execution failure is information the
// model can act on, so it becomes a result rather than ending the turn.
func TestDispatchToolErrorIsModelVisible(t *testing.T) {
	registry := prompty.NewToolRegistry()
	registry.RegisterText("boom", func(context.Context, map[string]interface{}) (string, error) {
		return "", errors.New("service unavailable")
	})

	dispatch, err := registry.Dispatch(context.Background(),
		model.ToolCall{Id: "1", Name: "boom", Arguments: "{}"}, nil, nil)
	if err != nil {
		t.Fatalf("an execution failure must not end the turn: %v", err)
	}
	if !strings.Contains(dispatch.Text(), "service unavailable") {
		t.Errorf("result = %q, want the failure text", dispatch.Text())
	}
	if dispatch.Result.Status == nil || *dispatch.Result.Status != model.ToolResultStatusError {
		t.Errorf("status = %v, want error", dispatch.Result.Status)
	}
}

// TestDispatchToolPanicIsContained: a host tool is arbitrary code.
func TestDispatchToolPanicIsContained(t *testing.T) {
	registry := prompty.NewToolRegistry()
	registry.RegisterText("panicky", func(context.Context, map[string]interface{}) (string, error) {
		panic("kaboom")
	})

	dispatch, err := registry.Dispatch(context.Background(),
		model.ToolCall{Id: "1", Name: "panicky", Arguments: "{}"}, nil, nil)
	if err != nil {
		t.Fatalf("a panicking tool must not end the turn: %v", err)
	}
	if !strings.Contains(dispatch.Text(), "panicked") {
		t.Errorf("result = %q, want a panic report", dispatch.Text())
	}
}

// TestDispatchInvalidArgumentsAreModelVisible: a malformed argument string is a
// mistake the model can correct on the next turn.
func TestDispatchInvalidArgumentsAreModelVisible(t *testing.T) {
	registry := prompty.NewToolRegistry()
	called := false
	registry.RegisterText("echo", func(context.Context, map[string]interface{}) (string, error) {
		called = true
		return "", nil
	})

	dispatch, err := registry.Dispatch(context.Background(),
		model.ToolCall{Id: "1", Name: "echo", Arguments: `{"value":`}, nil, nil)
	if err != nil {
		t.Fatalf("a malformed argument string must not end the turn: %v", err)
	}
	if called {
		t.Error("the tool ran despite undecodable arguments")
	}
	if !strings.Contains(dispatch.Text(), "invalid arguments") {
		t.Errorf("result = %q", dispatch.Text())
	}
}

func TestDispatchDenialIsModelVisible(t *testing.T) {
	registry := prompty.NewToolRegistry()
	called := false
	registry.RegisterText("dangerous", func(context.Context, map[string]interface{}) (string, error) {
		called = true
		return "secrets", nil
	})

	dispatch, err := registry.Dispatch(context.Background(),
		model.ToolCall{Id: "1", Name: "dangerous", Arguments: "{}"}, nil,
		func(context.Context, model.ToolCall, map[string]interface{}) prompty.PermissionDecision {
			return prompty.Deny("not authorized")
		})
	if err != nil {
		t.Fatalf("a denial must not end the turn: %v", err)
	}
	if called {
		t.Error("a denied tool executed")
	}
	if !dispatch.Denied {
		t.Error("dispatch was not marked denied")
	}
	if !strings.Contains(dispatch.Text(), "not authorized") {
		t.Errorf("denial result = %q, want the reason", dispatch.Text())
	}
	if strings.Contains(dispatch.Text(), "secrets") {
		t.Error("a denied tool's output leaked into the result")
	}
}

// TestDispatchBindingsOverrideModelArguments is the security property: a bound
// parameter is the host's to set, so a model-supplied value must not win.
func TestDispatchBindingsOverrideModelArguments(t *testing.T) {
	registry := prompty.NewToolRegistry()
	var seen map[string]interface{}
	registry.RegisterText("get_weather", func(_ context.Context, args map[string]interface{}) (string, error) {
		seen = args
		return "ok", nil
	})

	_, err := registry.Dispatch(context.Background(),
		model.ToolCall{Id: "1", Name: "get_weather", Arguments: `{"city":"Paris","unit":"fahrenheit"}`},
		map[string]interface{}{"unit": "celsius"}, nil)
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if seen["unit"] != "celsius" {
		t.Errorf("unit = %v, want the bound value celsius", seen["unit"])
	}
	if seen["city"] != "Paris" {
		t.Errorf("city = %v, want the model's value", seen["city"])
	}
}

func TestDispatchRespectsCancellation(t *testing.T) {
	registry := prompty.NewToolRegistry()
	called := false
	registry.RegisterText("echo", func(context.Context, map[string]interface{}) (string, error) {
		called = true
		return "", nil
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := registry.Dispatch(ctx, model.ToolCall{Id: "1", Name: "echo", Arguments: "{}"}, nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want context.Canceled", err)
	}
	if called {
		t.Error("the tool ran despite a cancelled context")
	}
}

func TestDecodeToolArguments(t *testing.T) {
	cases := []struct {
		raw     string
		wantErr bool
		check   func(map[string]interface{}) error
	}{
		{raw: "", check: func(m map[string]interface{}) error { return expectEmpty(m) }},
		{raw: "   ", check: func(m map[string]interface{}) error { return expectEmpty(m) }},
		{raw: "{}", check: func(m map[string]interface{}) error { return expectEmpty(m) }},
		{raw: "null", check: func(m map[string]interface{}) error { return expectEmpty(m) }},
		{raw: `{"n":7}`, check: func(m map[string]interface{}) error {
			// Integers must stay integers: a tool that echoes an argument back
			// into a prompt renders 7 and 7.0 differently.
			if n, ok := m["n"].(int64); !ok || n != 7 {
				return fmt.Errorf("expected int64(7), got %T(%v)", m["n"], m["n"])
			}
			return nil
		}},
		{raw: `{"broken":`, wantErr: true},
		{raw: `[1,2]`, wantErr: true},
	}

	for _, testCase := range cases {
		args, err := prompty.DecodeToolArguments(testCase.raw)
		if testCase.wantErr {
			if err == nil {
				t.Errorf("DecodeToolArguments(%q) returned no error", testCase.raw)
			}
			continue
		}
		if err != nil {
			t.Errorf("DecodeToolArguments(%q): %v", testCase.raw, err)
			continue
		}
		if testCase.check != nil {
			if err := testCase.check(args); err != nil {
				t.Errorf("DecodeToolArguments(%q): %v", testCase.raw, err)
			}
		}
	}
}

func expectEmpty(m map[string]interface{}) error {
	if len(m) != 0 {
		return errors.New("expected empty arguments")
	}
	return nil
}

func TestEncodeToolArguments(t *testing.T) {
	if got, err := prompty.EncodeToolArguments(nil); err != nil || got != "{}" {
		t.Errorf("EncodeToolArguments(nil) = %q, %v; want \"{}\", nil", got, err)
	}
	got, err := prompty.EncodeToolArguments(map[string]interface{}{"city": "Paris"})
	if err != nil || got != `{"city":"Paris"}` {
		t.Errorf("EncodeToolArguments = %q, %v", got, err)
	}
}

// TestResolveBindingsSkipsAbsentInputs: injecting nil for an unset optional
// input would overwrite the model's own value with null.
func TestResolveBindingsSkipsAbsentInputs(t *testing.T) {
	agent, err := prompty.LoadFrontmatter(map[string]interface{}{
		"name": "b", "instructions": "",
		"model": map[string]interface{}{"id": "m", "provider": "openai"},
		"tools": []interface{}{
			map[string]interface{}{
				"name": "t", "kind": "function",
				"parameters": []interface{}{
					map[string]interface{}{"name": "unit", "kind": "string"},
					map[string]interface{}{"name": "region", "kind": "string"},
				},
				"bindings": map[string]interface{}{
					"unit":   map[string]interface{}{"input": "preferred_unit"},
					"region": map[string]interface{}{"input": "missing_input"},
				},
			},
		},
	}, t.TempDir(), prompty.LoadOptions{})
	if err != nil {
		t.Fatalf("load agent: %v", err)
	}

	bindings := prompty.AgentBindings(agent, map[string]interface{}{"preferred_unit": "celsius"})
	resolved := bindings["t"]
	if resolved["unit"] != "celsius" {
		t.Errorf("unit = %v, want celsius", resolved["unit"])
	}
	if _, present := resolved["region"]; present {
		t.Error("a binding whose input is absent should not be injected")
	}
}

// TestNilRegistryIsSafe: a turn configured without tools must fail with the
// same clear error as an empty registry, not a nil dereference.
func TestNilRegistryIsSafe(t *testing.T) {
	agent, err := prompty.LoadFrontmatter(map[string]interface{}{
		"name": "n", "instructions": "",
		"model": map[string]interface{}{"id": "m", "provider": "openai", "apiType": "chat"},
	}, t.TempDir(), prompty.LoadOptions{})
	if err != nil {
		t.Fatalf("load agent: %v", err)
	}

	executor := &scriptedExecutor{responses: []interface{}{
		map[string]interface{}{"choices": []interface{}{map[string]interface{}{
			"message": map[string]interface{}{"role": "assistant", "content": nil,
				"tool_calls": []interface{}{map[string]interface{}{
					"id": "1", "type": "function",
					"function": map[string]interface{}{"name": "anything", "arguments": "{}"},
				}}},
		}}},
	}}

	_, err = prompty.RunMessages(context.Background(), agent,
		[]model.Message{{Role: model.RoleUser, Parts: []interface{}{model.TextPart{Kind: "text", Value: "go"}}}},
		prompty.RunOptions{Executor: executor, Processor: openai.NewProcessor()})
	if err == nil || !errors.Is(err, prompty.ErrToolNotRegistered) {
		t.Errorf("error = %v, want ErrToolNotRegistered", err)
	}
}
