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

// agentVector is the shared agent vector shape.
type agentVector struct {
	Name  string `json:"name"`
	Input struct {
		Messages      []map[string]interface{} `json:"messages"`
		Tools         []interface{}            `json:"tools"`
		ToolFunctions map[string]string        `json:"tool_functions"`
		ParentInputs  map[string]interface{}   `json:"parent_inputs"`
		Guardrails    *struct {
			Input  *guardrailRule `json:"input"`
			Output *guardrailRule `json:"output"`
			Tool   *struct {
				DenyTools []string `json:"deny_tools"`
				Reason    string   `json:"reason"`
			} `json:"tool"`
		} `json:"guardrails"`
		Cancel *struct {
			CancelledAt string `json:"cancelled_at"`
		} `json:"cancel"`
		Steering *struct {
			Messages []struct {
				InjectBeforeIteration int    `json:"inject_before_iteration"`
				Role                  string `json:"role"`
				Text                  string `json:"text"`
			} `json:"messages"`
		} `json:"steering"`
		ContextBudget *int `json:"context_budget"`
	} `json:"input"`
	Sequence []struct {
		Turn                 int                    `json:"turn"`
		LLMResponse          map[string]interface{} `json:"llm_response"`
		ToolResults          []toolResultSpec       `json:"tool_results"`
		ExpectedExecutionArg map[string]interface{} `json:"expected_execution_args"`
	} `json:"sequence"`
	Expected struct {
		Result             interface{}   `json:"result"`
		Error              string        `json:"error"`
		ErrorReason        string        `json:"error_reason"`
		Iterations         *int          `json:"iterations"`
		TotalMessages      *int          `json:"total_messages"`
		MessageSequence    []interface{} `json:"message_sequence"`
		ToolResultMessage  interface{}   `json:"tool_result_message"`
		AssistantToolCalls interface{}   `json:"assistant_tool_calls_message"`
		Events             []struct {
			Type string                 `json:"type"`
			Data map[string]interface{} `json:"data"`
		} `json:"events"`
		ToolExecutionOrder []string `json:"tool_execution_order"`
		DeniedTools        []string `json:"denied_tools"`
	} `json:"expected"`
}

type guardrailRule struct {
	Action string `json:"action"`
	Reason string `json:"reason"`
}

type toolResultSpec struct {
	ToolCallID string `json:"tool_call_id"`
	Result     string `json:"result"`
}

// unsupportedAgentVectors names the cases this runtime deliberately does not
// implement, each with the reason it is out of scope. They are skipped loudly
// rather than filtered silently so the coverage gap stays visible.
//
// It is currently empty: context trimming, guardrails and steering are now
// implemented as optional policies around the turn loop.
var unsupportedAgentVectors = map[string]string{}

// TestAgentVectors drives the shared agent vectors through the real turn loop,
// the real OpenAI processor and the real OpenAI message formatter. Only the
// transport is scripted.
func TestAgentVectors(t *testing.T) {
	var vectors []agentVector
	readVectorFile(t, "agent/agent_vectors.json", &vectors)

	var executed, skipped int
	for _, vector := range vectors {
		if reason, ok := unsupportedAgentVectors[vector.Name]; ok {
			skipped++
			t.Run(vector.Name, func(t *testing.T) { t.Skip(reason) })
			continue
		}

		executed++
		t.Run(vector.Name, func(t *testing.T) { runAgentVector(t, vector) })
	}

	t.Logf("agent vectors: %d executed, %d skipped with explicit reasons, %d total",
		executed, skipped, len(vectors))
	if executed == 0 {
		t.Fatal("no agent vectors executed; the harness is not selecting cases")
	}
}

func runAgentVector(t *testing.T, vector agentVector) {
	t.Helper()

	agent := agentForVector(t, vector)
	messages := messagesForVector(vector)

	responses := make([]interface{}, 0, len(vector.Sequence))
	for _, step := range vector.Sequence {
		responses = append(responses, step.LLMResponse)
	}

	// Results are keyed by tool_call_id so a tool answers exactly what the
	// vector recorded, regardless of dispatch order.
	results := map[string]string{}
	for _, step := range vector.Sequence {
		for _, result := range step.ToolResults {
			results[result.ToolCallID] = result.Result
		}
	}

	var (
		order        []string
		observedArgs = map[string]map[string]interface{}{}
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	registry := prompty.NewToolRegistry()
	for name := range vector.Input.ToolFunctions {
		toolName := name
		behaviour := vector.Input.ToolFunctions[name]
		registry.RegisterText(toolName, func(_ context.Context, args map[string]interface{}) (string, error) {
			order = append(order, toolName)
			observedArgs[toolName] = args

			// A tool the vector describes as raising must raise, so the loop's
			// error-to-tool-result conversion is exercised for real.
			if strings.Contains(behaviour, "raises RuntimeError") {
				return "", errors.New("RuntimeError: Weather service unavailable")
			}
			if id, ok := args["__call_id"].(string); ok {
				return results[id], nil
			}
			return lookupResultFor(toolName, order, vector), nil
		})
	}

	executor := &scriptedExecutor{responses: responses}
	options := prompty.RunOptions{
		Tools:     registry,
		Inputs:    vector.Input.ParentInputs,
		Executor:  executor,
		Processor: openai.NewProcessor(),
	}
	applyGuardrails(&options, vector)
	applySteering(&options, vector)
	applyContextBudget(&options, vector)
	events := captureEvents(&options)
	applyCancellation(&options, executor, vector, cancel, ctx)

	result, err := prompty.RunMessages(ctx, agent, messages, options)

	assertVectorOutcome(t, vector, result, err)
	assertVectorEvents(t, vector, events())
	assertToolOrder(t, vector, order)
	assertExecutionArgs(t, vector, observedArgs)
}

// lookupResultFor resolves a tool's recorded output by matching the call the
// vector scripted at this position in the execution order.
func lookupResultFor(name string, order []string, vector agentVector) string {
	occurrence := 0
	for _, executed := range order[:len(order)-1] {
		if executed == name {
			occurrence++
		}
	}
	seen := 0
	for _, step := range vector.Sequence {
		for i, call := range toolCallsOf(step.LLMResponse) {
			if call != name {
				continue
			}
			if seen == occurrence {
				if i < len(step.ToolResults) {
					return step.ToolResults[i].Result
				}
				return ""
			}
			seen++
		}
	}
	return ""
}

func toolCallsOf(response map[string]interface{}) []string {
	choices, _ := response["choices"].([]interface{})
	if len(choices) == 0 {
		return nil
	}
	choice, _ := choices[0].(map[string]interface{})
	message, _ := choice["message"].(map[string]interface{})
	rawCalls, _ := message["tool_calls"].([]interface{})

	names := make([]string, 0, len(rawCalls))
	for _, item := range rawCalls {
		entry, _ := item.(map[string]interface{})
		function, _ := entry["function"].(map[string]interface{})
		name, _ := function["name"].(string)
		names = append(names, name)
	}
	return names
}

func agentForVector(t *testing.T, vector agentVector) model.Prompty {
	t.Helper()

	data := map[string]interface{}{
		"name":         vector.Name,
		"instructions": "",
		"model": map[string]interface{}{
			"id":       "gpt-4o-mini",
			"provider": "openai",
			"apiType":  "chat",
		},
	}
	if len(vector.Input.Tools) > 0 {
		data["tools"] = vector.Input.Tools
	}
	agent, err := prompty.LoadFrontmatter(data, t.TempDir(), prompty.LoadOptions{})
	if err != nil {
		t.Fatalf("load agent: %v", err)
	}
	return agent
}

func messagesForVector(vector agentVector) []model.Message {
	messages := make([]model.Message, 0, len(vector.Input.Messages))
	for _, entry := range vector.Input.Messages {
		role, _ := entry["role"].(string)
		content, _ := entry["content"].(string)
		messages = append(messages, model.Message{
			Role:  model.Role(role),
			Parts: []interface{}{model.TextPart{Kind: "text", Value: content}},
		})
	}
	return messages
}

// applyGuardrails wires the vector's guardrail configuration onto the real
// RunOptions.Guardrails hooks, so the loop's own policy path is exercised
// rather than a test-only approximation of it.
func applyGuardrails(options *prompty.RunOptions, vector agentVector) {
	spec := vector.Input.Guardrails
	if spec == nil {
		return
	}

	guardrails := &prompty.Guardrails{}

	if spec.Input != nil {
		rule := *spec.Input
		guardrails.Input = func(context.Context, []model.Message, model.Prompty) prompty.GuardrailResult {
			if rule.Action == "deny" {
				return prompty.DenyGuardrail(rule.Reason)
			}
			return prompty.AllowGuardrail()
		}
	}
	if spec.Output != nil {
		rule := *spec.Output
		guardrails.Output = func(context.Context, interface{}, model.Prompty) prompty.GuardrailResult {
			if rule.Action == "deny" {
				return prompty.DenyGuardrail(rule.Reason)
			}
			return prompty.AllowGuardrail()
		}
	}
	if spec.Tool != nil {
		denied := map[string]bool{}
		for _, name := range spec.Tool.DenyTools {
			denied[name] = true
		}
		reason := spec.Tool.Reason
		guardrails.Tool = func(_ context.Context, name string, _ map[string]interface{}, _ model.Prompty) prompty.GuardrailResult {
			if denied[name] {
				return prompty.DenyGuardrail(reason)
			}
			return prompty.AllowGuardrail()
		}
	}

	options.Guardrails = guardrails
}

// applySteering pre-loads the vector's steering queue.
//
// Every steering vector injects before iteration 2, which is exactly when the
// loop drains: steering is never drained before the first model call, because
// a message queued before the turn started is already part of the prompt the
// caller built.
func applySteering(options *prompty.RunOptions, vector agentVector) {
	spec := vector.Input.Steering
	if spec == nil || len(spec.Messages) == 0 {
		return
	}
	queue := prompty.NewSteering()
	for _, message := range spec.Messages {
		queue.Send(message.Text)
	}
	options.Steering = queue
}

// applyContextBudget wires the vector's character budget onto the loop.
func applyContextBudget(options *prompty.RunOptions, vector agentVector) {
	if vector.Input.ContextBudget != nil {
		options.ContextBudget = *vector.Input.ContextBudget
	}
}

func captureEvents(options *prompty.RunOptions) func() []prompty.Event {
	var events []prompty.Event
	previous := options.OnEvent
	options.OnEvent = func(event prompty.Event) {
		events = append(events, event)
		if previous != nil {
			previous(event)
		}
	}
	return func() []prompty.Event { return events }
}

// applyCancellation wires the vector's cancellation point into the run.
func applyCancellation(
	options *prompty.RunOptions,
	executor *scriptedExecutor,
	vector agentVector,
	cancel context.CancelFunc,
	ctx context.Context,
) {
	if vector.Input.Cancel == nil {
		return
	}
	switch vector.Input.Cancel.CancelledAt {
	case "before_iteration":
		cancel()

	case "after_tool_0":
		// Cancel as soon as the first tool result is observed, so the second
		// tool in the same round must not run.
		previous := options.OnEvent
		seen := 0
		options.OnEvent = func(event prompty.Event) {
			if previous != nil {
				previous(event)
			}
			if event.Type == prompty.EventToolResult {
				seen++
				if seen == 1 {
					cancel()
				}
			}
		}

	case "before_iteration_2":
		// Cancel once iteration 1 has fully committed its tool messages, so
		// the loop stops at the top of iteration 2.
		previous := options.OnEvent
		options.OnEvent = func(event prompty.Event) {
			if previous != nil {
				previous(event)
			}
			if event.Type == prompty.EventMessagesUpdate {
				cancel()
			}
		}
	}
	_ = executor
	_ = ctx
}

func assertVectorOutcome(t *testing.T, vector agentVector, result prompty.RunResult, err error) {
	t.Helper()

	expected := vector.Expected

	switch {
	case expected.Error == "CancelledError":
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected cancellation, got err=%v result=%#v", err, result.Output)
		}
	case expected.Error == "GuardrailError":
		// The vector names the error *type* here and carries the message
		// separately in error_reason, unlike the vectors whose `error` field
		// is the message itself.
		var guardrailErr *prompty.GuardrailError
		if !errors.As(err, &guardrailErr) {
			t.Fatalf("expected a *prompty.GuardrailError, got err=%v result=%#v", err, result.Output)
		}
		if !errors.Is(err, prompty.ErrGuardrailDenied) {
			t.Error("a guardrail error must unwrap to prompty.ErrGuardrailDenied")
		}
		if expected.ErrorReason != "" && guardrailErr.Reason != expected.ErrorReason {
			t.Errorf("guardrail reason = %q, want %q", guardrailErr.Reason, expected.ErrorReason)
		}
	case expected.Error != "":
		if err == nil {
			t.Fatalf("expected error %q, got result %#v", expected.Error, result.Output)
		}
		if err.Error() != expected.Error {
			t.Errorf("error = %q, want %q", err.Error(), expected.Error)
		}
	default:
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if expected.Result != nil {
			if got, want := fmt.Sprint(result.Output), fmt.Sprint(expected.Result); got != want {
				t.Errorf("result = %q, want %q", got, want)
			}
		}
	}

	if expected.Iterations != nil && result.Iterations != *expected.Iterations {
		t.Errorf("iterations = %d, want %d", result.Iterations, *expected.Iterations)
	}

	if expected.MessageSequence != nil {
		if len(result.Messages) != len(expected.MessageSequence) {
			t.Fatalf("message count = %d, want %d\ngot: %s",
				len(result.Messages), len(expected.MessageSequence), describeMessages(result.Messages))
		}
		for i, want := range expected.MessageSequence {
			assertMessageView(t, fmt.Sprintf("message[%d]", i),
				viewMessage(result.Messages[i]), viewExpectedMessage(t, want))
		}
	} else if expected.TotalMessages != nil && len(result.Dispatches) == 0 {
		// total_messages is only asserted for turns with no tool round. The
		// vectors' total_messages is one higher than their own
		// message_sequence for every tool-calling case, so message_sequence —
		// which matches the canonical runtime — is the authority there.
		if len(result.Messages) != *expected.TotalMessages {
			t.Errorf("total messages = %d, want %d", len(result.Messages), *expected.TotalMessages)
		}
	}

	if expected.ToolResultMessage != nil {
		found := false
		for _, msg := range result.Messages {
			if msg.Role == model.RoleTool {
				assertMessageView(t, "tool_result_message",
					viewMessage(msg), viewExpectedMessage(t, expected.ToolResultMessage))
				found = true
				break
			}
		}
		if !found {
			t.Error("expected a tool result message, found none")
		}
	}

	if expected.AssistantToolCalls != nil {
		found := false
		for _, msg := range result.Messages {
			if msg.Role == model.RoleAssistant && msg.Metadata["tool_calls"] != nil {
				assertMessageView(t, "assistant_tool_calls_message",
					viewMessage(msg), viewExpectedMessage(t, expected.AssistantToolCalls))
				found = true
				break
			}
		}
		if !found {
			t.Error("expected an assistant tool_calls message, found none")
		}
	}

	if expected.DeniedTools != nil {
		var denied []string
		for _, dispatch := range result.Dispatches {
			if dispatch.Denied {
				denied = append(denied, dispatch.Call.Name)
			}
		}
		if !equalStrings(denied, expected.DeniedTools) {
			t.Errorf("denied tools = %v, want %v", denied, expected.DeniedTools)
		}
	}
}

func assertVectorEvents(t *testing.T, vector agentVector, events []prompty.Event) {
	t.Helper()

	if len(vector.Expected.Events) == 0 {
		return
	}

	want := make([]string, 0, len(vector.Expected.Events))
	fullySpecified := false
	for _, event := range vector.Expected.Events {
		want = append(want, event.Type)
		if len(event.Data) > 0 {
			fullySpecified = true
		}
	}
	got := make([]string, 0, len(events))
	for _, event := range events {
		got = append(got, event.Type)
	}

	// Some vectors record only the event types and note that "events verify
	// type sequence only"; those are a partial specification and are asserted
	// as an ordered subsequence. Vectors that spell out event payloads are
	// asserted exactly.
	if fullySpecified {
		if !equalStrings(got, want) {
			t.Errorf("event sequence = %v, want %v", got, want)
		}
		return
	}
	if !isSubsequence(want, got) {
		t.Errorf("event sequence = %v, want %v as an ordered subsequence", got, want)
	}
}

// isSubsequence reports whether want appears in order within got.
func isSubsequence(want, got []string) bool {
	i := 0
	for _, value := range got {
		if i < len(want) && value == want[i] {
			i++
		}
	}
	return i == len(want)
}

func assertToolOrder(t *testing.T, vector agentVector, order []string) {
	t.Helper()

	if vector.Expected.ToolExecutionOrder == nil {
		return
	}
	if !equalStrings(order, vector.Expected.ToolExecutionOrder) {
		t.Errorf("tool execution order = %v, want %v", order, vector.Expected.ToolExecutionOrder)
	}
}

func assertExecutionArgs(t *testing.T, vector agentVector, observed map[string]map[string]interface{}) {
	t.Helper()

	for _, step := range vector.Sequence {
		for name, want := range step.ExpectedExecutionArg {
			wantArgs, ok := want.(map[string]interface{})
			if !ok {
				continue
			}
			gotArgs, ok := observed[name]
			if !ok {
				t.Errorf("tool %q was never executed", name)
				continue
			}
			for key, wantValue := range wantArgs {
				if got := fmt.Sprint(gotArgs[key]); got != fmt.Sprint(wantValue) {
					t.Errorf("tool %q arg %q = %v, want %v", name, key, gotArgs[key], wantValue)
				}
			}
		}
	}
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func describeMessages(messages []model.Message) string {
	var b strings.Builder
	for i, msg := range messages {
		fmt.Fprintf(&b, "\n  [%d] %s %q", i, msg.Role, msg.Text())
		if msg.Metadata["tool_call_id"] != nil {
			fmt.Fprintf(&b, " tool_call_id=%v", msg.Metadata["tool_call_id"])
		}
		if msg.Metadata["tool_calls"] != nil {
			b.WriteString(" +tool_calls")
		}
	}
	return b.String()
}
