package prompty_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"testing"

	prompty "prompty"
	model "prompty/model"
	openai "prompty/openai"
)

func userMessage(text string) model.Message   { return model.NewUserMessage(text) }
func systemMessage(text string) model.Message { return model.NewSystemMessage(text) }

// ---------------------------------------------------------------------------
// Context trimming
// ---------------------------------------------------------------------------

func TestEstimateCharsCountsRolesPartsAndToolCalls(t *testing.T) {
	messages := []model.Message{systemMessage("You are helpful."), userMessage("Hello!")}
	// "system"(6) + 4 + 16 + "user"(4) + 4 + 6 = 40
	if got := prompty.EstimateChars(messages); got != 40 {
		t.Errorf("EstimateChars = %d, want 40", got)
	}
	if got := prompty.EstimateChars(nil); got != 0 {
		t.Errorf("EstimateChars(nil) = %d, want 0", got)
	}

	withCalls := userMessage("")
	withCalls.Metadata = map[string]interface{}{
		"tool_calls": []interface{}{
			map[string]interface{}{"name": "get_weather", "arguments": `{"city":"NY"}`},
		},
	}
	// Tool calls are sent to the provider verbatim and are frequently the
	// largest thing in an agent conversation, so they must be charged.
	if prompty.EstimateChars([]model.Message{withCalls}) <= 8 {
		t.Error("tool_calls metadata must contribute to the estimate")
	}
}

func TestEstimateChargesNonTextPartsAFlatCost(t *testing.T) {
	// A reference to an image says nothing about the context it will cost, so
	// it is charged a cross-runtime constant rather than its byte length.
	message := model.Message{
		Role:  model.RoleUser,
		Parts: []interface{}{model.ImagePart{Kind: "image", Source: "https://example.test/x.png"}},
	}
	if got := prompty.EstimateChars([]model.Message{message}); got != len("user")+4+200 {
		t.Errorf("EstimateChars = %d, want the flat non-text charge", got)
	}
}

func TestTrimLeavesAConversationInsideItsBudgetAlone(t *testing.T) {
	messages := []model.Message{systemMessage("sys"), userMessage("hi")}
	dropped, trimmed := prompty.TrimToContextWindow(messages, 100000)
	if len(dropped) != 0 {
		t.Errorf("dropped %d messages from a conversation inside budget", len(dropped))
	}
	if len(trimmed) != 2 {
		t.Errorf("trimmed to %d messages, want 2", len(trimmed))
	}
}

func TestTrimIsDisabledByANonPositiveBudget(t *testing.T) {
	messages := []model.Message{systemMessage("sys"), userMessage(strings.Repeat("A", 5000))}
	for _, budget := range []int{0, -1} {
		dropped, trimmed := prompty.TrimToContextWindow(messages, budget)
		if len(dropped) != 0 || len(trimmed) != len(messages) {
			t.Errorf("budget %d trimmed the conversation; a non-positive budget must disable trimming", budget)
		}
	}
}

func TestTrimDropsOldestAndPreservesEverySystemMessage(t *testing.T) {
	messages := []model.Message{
		systemMessage("sys1"),
		systemMessage("sys2"),
		userMessage(strings.Repeat("A", 2000)),
		userMessage(strings.Repeat("B", 100)),
		userMessage(strings.Repeat("C", 100)),
	}
	dropped, trimmed := prompty.TrimToContextWindow(messages, 500)

	if len(dropped) == 0 {
		t.Fatal("an over-budget conversation must drop something")
	}
	if !strings.HasPrefix(dropped[0].Text(), "A") {
		t.Errorf("dropped %q first, want the oldest non-system message", dropped[0].Text())
	}
	if trimmed[0].Role != model.RoleSystem || trimmed[0].Text() != "sys1" {
		t.Error("the first system message was not preserved")
	}
	if trimmed[1].Role != model.RoleSystem || trimmed[1].Text() != "sys2" {
		t.Error("every system message must be preserved regardless of budget")
	}
	// The summary sits directly after the system messages, so the model is
	// told history was elided rather than silently losing it.
	if !strings.Contains(trimmed[2].Text(), "messages omitted") {
		t.Errorf("message[2] = %q, want the context summary", trimmed[2].Text())
	}
	if trimmed[2].Role != model.RoleSystem {
		t.Errorf("summary role = %q, want system — a user-role placeholder would be answered as if the user typed it", trimmed[2].Role)
	}
	if len(trimmed) != len(messages)-len(dropped)+1 {
		t.Errorf("trimmed to %d messages, want %d kept plus one summary",
			len(trimmed), len(messages)-len(dropped))
	}
}

func TestTrimKeepsTheMinimumExchangeEvenAtAnImpossibleBudget(t *testing.T) {
	// A budget too small to leave the model anything to answer produces an
	// untouched conversation rather than an unanswerable one.
	messages := []model.Message{
		systemMessage("sys"),
		userMessage(strings.Repeat("A", 5000)),
		userMessage(strings.Repeat("B", 5000)),
	}
	dropped, trimmed := prompty.TrimToContextWindow(messages, 10)
	if len(dropped) != 0 {
		t.Errorf("dropped %d messages, want none — two non-system messages is the floor", len(dropped))
	}
	if len(trimmed) != 3 {
		t.Errorf("trimmed to %d messages, want the original 3", len(trimmed))
	}
}

func TestTrimNeverDropsTheMostRecentUserMessage(t *testing.T) {
	messages := []model.Message{systemMessage("sys")}
	for index := 0; index < 12; index++ {
		messages = append(messages, userMessage(strings.Repeat("x", 400)))
	}
	final := userMessage("what is the weather?")
	messages = append(messages, final)

	_, trimmed := prompty.TrimToContextWindow(messages, 500)
	if trimmed[len(trimmed)-1].Text() != final.Text() {
		t.Errorf("last message = %q, want the newest user turn", trimmed[len(trimmed)-1].Text())
	}
}

func TestTrimKeepsTheResultWithinTheBudgetPlusItsFloors(t *testing.T) {
	// The budget is a target, not a guarantee: the preserved system messages
	// and the two-message floor outrank it. What must NOT happen is the
	// summary blowing the budget on its own — an uncapped placeholder can be
	// thousands of characters regardless of how small the budget is.
	for _, budget := range []int{300, 500, 2000} {
		messages := []model.Message{systemMessage("sys")}
		for index := 0; index < 30; index++ {
			messages = append(messages, userMessage(strings.Repeat("x", 300)))
		}

		dropped, trimmed := prompty.TrimToContextWindow(messages, budget)
		if len(dropped) == 0 {
			t.Fatalf("budget %d dropped nothing from a %d-character conversation",
				budget, prompty.EstimateChars(messages))
		}

		// The floors are the system messages plus the newest turns trimming
		// refuses to drop.
		floor := prompty.EstimateChars(messages[:1]) +
			prompty.EstimateChars(messages[len(messages)-2:])
		reserve := budget / 20
		if reserve > 5000 {
			reserve = 5000
		}
		// The result may exceed the budget only up to the floor, plus the
		// reserved summary and a little slack for the surrounding
		// "[Summary ...] ... (N omitted)" text.
		ceiling := budget
		if floor > ceiling {
			ceiling = floor
		}
		const summaryChrome = 128
		limit := ceiling + reserve + summaryChrome

		if got := prompty.EstimateChars(trimmed); got > limit {
			t.Errorf("budget %d produced %d characters, want at most %d (ceiling %d + reserve %d)",
				budget, got, limit, ceiling, reserve)
		}
	}
}

func TestSummarizeDroppedTruncatesLongMessagesSafely(t *testing.T) {
	summary := prompty.SummarizeDropped([]model.Message{userMessage(strings.Repeat("x", 500))})
	if len(summary) >= 500 || !strings.HasSuffix(summary, "...") {
		t.Errorf("summary = %d chars, want it truncated with an ellipsis", len(summary))
	}
	if prompty.SummarizeDropped(nil) != "" {
		t.Error("summarizing nothing must produce nothing")
	}
}

func TestTrimAtADegenerateBudgetDoesNotLeaveABareEllipsis(t *testing.T) {
	// A budget under 20 reserves nothing for the summary. Reserving nothing
	// must mean nothing, not a three-character placeholder that still exceeds
	// the reserve it was held to.
	messages := []model.Message{systemMessage("s")}
	for index := 0; index < 6; index++ {
		messages = append(messages, userMessage(strings.Repeat("y", 50)))
	}
	dropped, trimmed := prompty.TrimToContextWindow(messages, 19)
	if len(dropped) == 0 {
		t.Skip("this budget did not trigger trimming; the invariant is covered elsewhere")
	}
	for _, message := range trimmed {
		if strings.Contains(message.Text(), "] ...\n") {
			continue // the "(N messages omitted)" suffix is expected
		}
		if strings.Contains(message.Text(), "...] ") {
			t.Errorf("summary %q leaks a bare ellipsis where nothing was reserved", message.Text())
		}
	}
}

func TestSummarizeDroppedDoesNotSplitAMultiByteRune(t *testing.T) {
	// Truncating on a byte boundary would emit invalid UTF-8, which some
	// providers reject outright.
	summary := prompty.SummarizeDropped([]model.Message{userMessage(strings.Repeat("é", 400))})
	for _, r := range summary {
		if r == '\uFFFD' {
			t.Fatal("truncation split a multi-byte rune")
		}
	}
}

func TestSummarizeDroppedLabelsEmptyMessages(t *testing.T) {
	summary := prompty.SummarizeDropped([]model.Message{{Role: model.RoleAssistant}})
	if summary != "[assistant message]" {
		t.Errorf("summary = %q, want a labelled placeholder", summary)
	}
}

func TestFormatDroppedMessagesShowsToolCalls(t *testing.T) {
	message := model.Message{Role: model.RoleAssistant}
	message.Metadata = map[string]interface{}{
		"tool_calls": []interface{}{
			map[string]interface{}{
				"function": map[string]interface{}{"name": "get_weather", "arguments": `{"city":"NY"}`},
			},
		},
	}
	formatted := prompty.FormatDroppedMessages([]model.Message{message})
	if !strings.Contains(formatted, "Called: get_weather") || !strings.Contains(formatted, "NY") {
		t.Errorf("formatted = %q, want the tool call rendered", formatted)
	}
	if prompty.FormatDroppedMessages(nil) != "" {
		t.Error("formatting nothing must produce nothing")
	}
}

func TestRunTrimsAndReportsWhatItDropped(t *testing.T) {
	// Trimming that is not reported turns a short answer caused by lost
	// history into one the caller blames on the model.
	agent := simplePolicyAgent(t)
	messages := []model.Message{systemMessage("sys")}
	for index := 0; index < 10; index++ {
		messages = append(messages, userMessage(strings.Repeat("x", 400)))
	}
	messages = append(messages, userMessage("and finally?"))

	var seen []model.Message
	result, err := prompty.RunMessages(context.Background(), agent, messages, prompty.RunOptions{
		ContextBudget: 500,
		Executor:      recordingExecutor(&seen, textResponse("ok")),
		Processor:     passthroughProcessor{},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if result.ContextDropped == 0 {
		t.Error("trimming must be reported on the result")
	}
	if len(seen) >= len(messages) {
		t.Errorf("the executor saw %d messages, want fewer than the original %d", len(seen), len(messages))
	}
	if seen[0].Role != model.RoleSystem {
		t.Error("the system message was not preserved into the provider call")
	}
}

func TestCompactionReplacesTheMechanicalSummary(t *testing.T) {
	agent := simplePolicyAgent(t)
	messages := []model.Message{systemMessage("sys")}
	for index := 0; index < 10; index++ {
		messages = append(messages, userMessage(strings.Repeat("x", 400)))
	}
	messages = append(messages, userMessage("and finally?"))

	var seen []model.Message
	_, err := prompty.RunMessages(context.Background(), agent, messages, prompty.RunOptions{
		// The budget is large enough that the summary reserve (5%) comfortably
		// holds the compacted text; the cap itself is covered separately.
		ContextBudget: 4000,
		Compaction: func(_ context.Context, dropped []model.Message) (string, error) {
			return fmt.Sprintf("the user repeated themselves %d times", len(dropped)), nil
		},
		Executor:  recordingExecutor(&seen, textResponse("ok")),
		Processor: passthroughProcessor{},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(seen[1].Text(), "the user repeated themselves") {
		t.Errorf("summary = %q, want the compaction result", seen[1].Text())
	}
}

func TestCompactionIsHeldToTheSummaryReserve(t *testing.T) {
	// A summariser that returns an essay must not be able to defeat the budget
	// the host asked for.
	agent := simplePolicyAgent(t)
	messages := []model.Message{systemMessage("sys")}
	for index := 0; index < 10; index++ {
		messages = append(messages, userMessage(strings.Repeat("x", 400)))
	}
	messages = append(messages, userMessage("and finally?"))

	var seen []model.Message
	_, err := prompty.RunMessages(context.Background(), agent, messages, prompty.RunOptions{
		ContextBudget: 500, // reserve is 25 characters
		Compaction: func(context.Context, []model.Message) (string, error) {
			return strings.Repeat("verbose ", 500), nil
		},
		Executor:  recordingExecutor(&seen, textResponse("ok")),
		Processor: passthroughProcessor{},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(seen[1].Text()) > 256 {
		t.Errorf("compacted summary is %d characters, want it held to the reserve", len(seen[1].Text()))
	}
	if !strings.Contains(seen[1].Text(), "messages omitted") {
		t.Errorf("summary = %q, want it to still say how much was elided", seen[1].Text())
	}
}

func TestCompactionFailureLeavesTheMechanicalSummary(t *testing.T) {
	// Compaction is an optimisation. Failing to improve a summary must never
	// fail the turn.
	agent := simplePolicyAgent(t)
	messages := []model.Message{systemMessage("sys")}
	for index := 0; index < 10; index++ {
		messages = append(messages, userMessage(strings.Repeat("x", 400)))
	}
	messages = append(messages, userMessage("and finally?"))

	var seen []model.Message
	_, err := prompty.RunMessages(context.Background(), agent, messages, prompty.RunOptions{
		ContextBudget: 500,
		Compaction: func(context.Context, []model.Message) (string, error) {
			return "", errors.New("summariser unavailable")
		},
		Executor:  recordingExecutor(&seen, textResponse("ok")),
		Processor: passthroughProcessor{},
	})
	if err != nil {
		t.Fatalf("a failed compaction must not fail the turn: %v", err)
	}
	if !strings.Contains(seen[1].Text(), "messages omitted") {
		t.Errorf("summary = %q, want the mechanical fallback", seen[1].Text())
	}
}

// ---------------------------------------------------------------------------
// Guardrails
// ---------------------------------------------------------------------------

func TestGuardrailsAllowWhenUnset(t *testing.T) {
	var guardrails *prompty.Guardrails
	agent := model.Prompty{}
	if !guardrails.CheckInput(context.Background(), nil, agent).Allowed {
		t.Error("a nil Guardrails must allow")
	}
	if !guardrails.CheckOutput(context.Background(), nil, agent).Allowed {
		t.Error("a nil Guardrails must allow output")
	}
	if !(&prompty.Guardrails{}).CheckTool(context.Background(), "x", nil, agent).Allowed {
		t.Error("an unset tool guardrail must allow")
	}
}

func TestInputGuardrailDenialStopsBeforeAnyProviderCall(t *testing.T) {
	agent := simplePolicyAgent(t)
	called := false

	_, err := prompty.RunMessages(context.Background(), agent, []model.Message{userMessage("hi")}, prompty.RunOptions{
		Guardrails: &prompty.Guardrails{
			Input: func(context.Context, []model.Message, model.Prompty) prompty.GuardrailResult {
				return prompty.DenyGuardrail("Contains PII")
			},
		},
		Executor: executorFunc(func(model.Prompty, []model.Message) (interface{}, error) {
			called = true
			return textResponse("never"), nil
		}),
		Processor: passthroughProcessor{},
	})

	var guardrailErr *prompty.GuardrailError
	if !errors.As(err, &guardrailErr) {
		t.Fatalf("err = %v, want a *GuardrailError", err)
	}
	if guardrailErr.Phase != prompty.GuardrailPhaseInput {
		t.Errorf("phase = %q, want input", guardrailErr.Phase)
	}
	if guardrailErr.Reason != "Contains PII" {
		t.Errorf("reason = %q, want the policy's reason", guardrailErr.Reason)
	}
	if !errors.Is(err, prompty.ErrGuardrailDenied) {
		t.Error("a guardrail error must unwrap to ErrGuardrailDenied")
	}
	if called {
		t.Error("a denied input must not be paid for at the provider")
	}
}

func TestOutputGuardrailDenialWithholdsTheAnswer(t *testing.T) {
	agent := simplePolicyAgent(t)
	result, err := prompty.RunMessages(context.Background(), agent, []model.Message{userMessage("hi")}, prompty.RunOptions{
		Guardrails: &prompty.Guardrails{
			Output: func(context.Context, interface{}, model.Prompty) prompty.GuardrailResult {
				return prompty.DenyGuardrail("Response contains harmful content")
			},
		},
		Executor:  scriptedResponses(textResponse("something harmful")),
		Processor: passthroughProcessor{},
	})

	var guardrailErr *prompty.GuardrailError
	if !errors.As(err, &guardrailErr) || guardrailErr.Phase != prompty.GuardrailPhaseOutput {
		t.Fatalf("err = %v, want an output *GuardrailError", err)
	}
	if result.Output != nil {
		t.Errorf("output = %#v, want nothing returned to the caller", result.Output)
	}
	// The model was still called: the vector's iteration count records that
	// the turn cost a provider round even though the answer was withheld.
	if result.Iterations != 1 {
		t.Errorf("iterations = %d, want 1", result.Iterations)
	}
}

func TestOutputGuardrailRewriteReplacesTheAnswer(t *testing.T) {
	agent := simplePolicyAgent(t)
	result, err := prompty.RunMessages(context.Background(), agent, []model.Message{userMessage("hi")}, prompty.RunOptions{
		Guardrails: &prompty.Guardrails{
			Output: func(context.Context, interface{}, model.Prompty) prompty.GuardrailResult {
				return prompty.RewriteGuardrail("redacted")
			},
		},
		Executor:  scriptedResponses(textResponse("secret")),
		Processor: passthroughProcessor{},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if result.Text() != "redacted" {
		t.Errorf("output = %q, want the rewrite", result.Text())
	}
}

func TestGuardrailDenialWithNoReasonStillExplainsItself(t *testing.T) {
	agent := simplePolicyAgent(t)
	_, err := prompty.RunMessages(context.Background(), agent, []model.Message{userMessage("hi")}, prompty.RunOptions{
		Guardrails: &prompty.Guardrails{
			Input: func(context.Context, []model.Message, model.Prompty) prompty.GuardrailResult {
				return prompty.GuardrailResult{Allowed: false}
			},
		},
		Executor:  scriptedResponses(textResponse("never")),
		Processor: passthroughProcessor{},
	})

	var guardrailErr *prompty.GuardrailError
	if !errors.As(err, &guardrailErr) {
		t.Fatalf("err = %v, want a *GuardrailError", err)
	}
	if guardrailErr.Reason == "" {
		t.Error("a denial with no reason must still produce something actionable")
	}
}

// ---------------------------------------------------------------------------
// Steering
// ---------------------------------------------------------------------------

func TestSteeringDrainsInFIFOOrderAndEmpties(t *testing.T) {
	queue := prompty.NewSteering()
	if !queue.IsEmpty() || queue.HasPending() {
		t.Error("a new queue must be empty")
	}
	queue.SendAll("A", "B")
	queue.Send("C")
	if queue.Len() != 3 {
		t.Fatalf("len = %d, want 3", queue.Len())
	}

	drained := queue.Drain()
	if len(drained) != 3 {
		t.Fatalf("drained %d messages, want 3", len(drained))
	}
	for index, want := range []string{"A", "B", "C"} {
		if drained[index].Text() != want {
			t.Errorf("drained[%d] = %q, want %q", index, drained[index].Text(), want)
		}
		if drained[index].Role != model.RoleUser {
			t.Errorf("drained[%d] role = %q, want user", index, drained[index].Role)
		}
	}
	if !queue.IsEmpty() {
		t.Error("Drain must empty the queue")
	}
	if queue.Drain() != nil {
		t.Error("draining an empty queue must produce nothing")
	}
}

func TestSteeringNilReceiverIsSafe(t *testing.T) {
	// RunOptions.Steering is optional, so the loop calls Drain unconditionally.
	var queue *prompty.Steering
	queue.Send("ignored")
	if queue.Drain() != nil || queue.Len() != 0 || queue.HasPending() {
		t.Error("a nil steering queue must behave as an empty one")
	}
}

func TestSteeringIsSafeUnderConcurrentSend(t *testing.T) {
	// The whole point of steering is that it arrives from outside the turn.
	queue := prompty.NewSteering()
	const senders = 32
	var wait sync.WaitGroup
	for index := 0; index < senders; index++ {
		wait.Add(1)
		go func(id int) {
			defer wait.Done()
			queue.Send(fmt.Sprintf("message-%d", id))
		}(index)
	}
	wait.Wait()
	if got := len(queue.Drain()); got != senders {
		t.Errorf("drained %d messages, want %d", got, senders)
	}
}

func TestSteeringIsNotDrainedBeforeTheFirstModelCall(t *testing.T) {
	// A message queued before the turn started is already part of the prompt
	// the caller built; injecting it again would duplicate it.
	agent := simplePolicyAgent(t)
	queue := prompty.NewSteering()
	queue.Send("do it in Celsius")

	var seen []model.Message
	result, err := prompty.RunMessages(context.Background(), agent, []model.Message{userMessage("weather?")}, prompty.RunOptions{
		Steering:  queue,
		Executor:  recordingExecutor(&seen, textResponse("sunny")),
		Processor: passthroughProcessor{},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(seen) != 1 {
		t.Errorf("the first model call saw %d messages, want only the original 1", len(seen))
	}
	if result.SteeringInjected != 0 {
		t.Errorf("injected %d messages before the first call, want 0", result.SteeringInjected)
	}
	if queue.Len() != 1 {
		t.Error("undrained steering must stay queued for the next iteration")
	}
}

func TestSteeringQueuedBeforeTheTurnIsInjectedAtTheNextIteration(t *testing.T) {
	// Text queued before the turn starts is not discarded: it is injected at
	// the next iteration boundary, which is what the shared steering vectors
	// describe. This pins the behaviour the loop's comment claims.
	agent := simplePolicyAgent(t)
	queue := prompty.NewSteering()
	queue.Send("answer in Celsius")

	tools := prompty.NewToolRegistry()
	tools.RegisterText("get_weather", func(context.Context, map[string]interface{}) (string, error) {
		return "72F", nil
	})

	var calls [][]model.Message
	result, err := prompty.RunMessages(
		context.Background(),
		agent,
		[]model.Message{userMessage("weather?")},
		prompty.RunOptions{
			Tools:    tools,
			Steering: queue,
			Executor: recordingEachCall(&calls,
				toolCallResponse("call-1", "get_weather", `{"city":"Paris"}`),
				chatTextResponse("22C and sunny"),
			),
			Processor: openaiProcessorForPolicies(),
		},
	)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if result.SteeringInjected != 1 {
		t.Fatalf("injected %d steering messages, want 1", result.SteeringInjected)
	}
	if len(calls) != 2 {
		t.Fatalf("the loop made %d model calls, want 2", len(calls))
	}
	if containsText(calls[0], "answer in Celsius") {
		t.Error("steering must not reach the first model call")
	}
	if !containsText(calls[1], "answer in Celsius") {
		t.Error("steering must reach the second model call")
	}
	if !queue.IsEmpty() {
		t.Error("the queue must be empty after injection")
	}
}

func containsText(messages []model.Message, want string) bool {
	for _, message := range messages {
		if strings.Contains(message.Text(), want) {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

func simplePolicyAgent(t *testing.T) model.Prompty {
	t.Helper()
	agent, err := prompty.LoadFrontmatter(map[string]interface{}{
		"name":         "policy-test",
		"instructions": "",
		"model":        map[string]interface{}{"id": "gpt-4o-mini", "provider": "openai", "apiType": "chat"},
	}, t.TempDir(), prompty.LoadOptions{})
	if err != nil {
		t.Fatalf("load agent: %v", err)
	}
	return agent
}

func textResponse(text string) interface{} { return text }

// executorFunc adapts a function to the emitted Executor contract.
type executorFunc func(agent model.Prompty, messages []model.Message) (interface{}, error)

func (f executorFunc) Execute(agent model.Prompty, messages []model.Message) (interface{}, error) {
	return f(agent, messages)
}

func (f executorFunc) ExecuteStream(model.Prompty, []model.Message) (interface{}, error) {
	return nil, errors.New("streaming is not supported by this test executor")
}

func (f executorFunc) FormatToolMessages(interface{}, []model.ToolCall, []string, *string) ([]model.Message, error) {
	return nil, nil
}

func scriptedResponses(responses ...interface{}) prompty.Executor {
	index := 0
	return executorFunc(func(model.Prompty, []model.Message) (interface{}, error) {
		if index >= len(responses) {
			return nil, fmt.Errorf("the test scripted %d responses but the loop asked for %d", len(responses), index+1)
		}
		response := responses[index]
		index++
		return response, nil
	})
}

// recordingExecutor captures the messages the loop actually sent, which is how
// the trimming and steering tests observe what reached the provider.
func recordingExecutor(seen *[]model.Message, response interface{}) prompty.Executor {
	return executorFunc(func(_ model.Prompty, messages []model.Message) (interface{}, error) {
		*seen = append([]model.Message(nil), messages...)
		return response, nil
	})
}

// passthroughProcessor returns the executor's value unchanged, so these tests
// exercise the loop's policy hooks rather than a provider's decoding.
type passthroughProcessor struct{}

func (passthroughProcessor) Process(_ model.Prompty, response interface{}) (interface{}, error) {
	return response, nil
}

func (passthroughProcessor) ProcessStream(interface{}) (interface{}, error) {
	return nil, errors.New("streaming is not supported by this test processor")
}

// openaiProcessorForPolicies decodes the OpenAI chat wire shape, so a test that
// needs a real tool round gets the real tool-call extraction.
func openaiProcessorForPolicies() prompty.Processor { return openai.NewProcessor() }

// recordingEachCall captures the messages of every model call in order and
// replays a script of responses, which is how the steering test proves which
// call saw the injected message.
func recordingEachCall(calls *[][]model.Message, responses ...interface{}) prompty.Executor {
	index := 0
	inner := openai.Executor{}
	return executorRecorder{
		record: func(messages []model.Message) {
			*calls = append(*calls, append([]model.Message(nil), messages...))
		},
		next: func() (interface{}, error) {
			if index >= len(responses) {
				return nil, fmt.Errorf("the test scripted %d responses but the loop asked for %d", len(responses), index+1)
			}
			response := responses[index]
			index++
			return response, nil
		},
		// Tool messages are formatted by the real OpenAI executor, so the
		// second call carries a provider-valid tool exchange.
		format: inner.FormatToolMessages,
	}
}

type executorRecorder struct {
	record func([]model.Message)
	next   func() (interface{}, error)
	format func(interface{}, []model.ToolCall, []string, *string) ([]model.Message, error)
}

func (e executorRecorder) Execute(_ model.Prompty, messages []model.Message) (interface{}, error) {
	e.record(messages)
	return e.next()
}

func (e executorRecorder) ExecuteStream(model.Prompty, []model.Message) (interface{}, error) {
	return nil, errors.New("streaming is not supported by this test executor")
}

func (e executorRecorder) FormatToolMessages(
	raw interface{},
	calls []model.ToolCall,
	results []string,
	text *string,
) ([]model.Message, error) {
	return e.format(raw, calls, results, text)
}

// chatTextResponse renders a plain assistant answer in the OpenAI chat shape.
func chatTextResponse(text string) map[string]interface{} {
	return map[string]interface{}{
		"object": "chat.completion",
		"choices": []interface{}{map[string]interface{}{
			"index":   0,
			"message": map[string]interface{}{"role": "assistant", "content": text},
		}},
	}
}

// toolCallResponse renders a single tool call in the OpenAI chat shape.
func toolCallResponse(id, name, arguments string) map[string]interface{} {
	return map[string]interface{}{
		"object": "chat.completion",
		"choices": []interface{}{map[string]interface{}{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": "",
				"tool_calls": []interface{}{map[string]interface{}{
					"id": id, "type": "function",
					"function": map[string]interface{}{"name": name, "arguments": arguments},
				}},
			},
		}},
	}
}
