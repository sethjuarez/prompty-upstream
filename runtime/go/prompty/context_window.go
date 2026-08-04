package prompty

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	model "prompty/model"
)

// Context window management (spec §13.3): estimate the size of a conversation
// and trim it to fit a character budget.
//
// The budget is expressed in characters, not tokens, on purpose. A tokenizer is
// provider- and model-specific and would drag a large table into every host; a
// character budget is a stable, provider-neutral proxy that every runtime
// computes identically, so the shared context vectors converge.

// contextPartCost is what a non-text part (image, file, audio) contributes to
// the estimate. Those parts are references, not prose — their in-band size says
// nothing about how much context the provider will spend on them — so they are
// charged a flat, cross-runtime constant.
const contextPartCost = 200

// contextRoleOverhead is the per-message formatting cost charged on top of the
// role name: separators, delimiters, and the provider's own envelope.
const contextRoleOverhead = 4

// contextSummaryTruncate caps how much of one dropped message is quoted in the
// generated summary.
const contextSummaryTruncate = 200

// contextSummaryCap bounds the whole generated summary.
const contextSummaryCap = 4000

// contextMinimumKept is the number of non-system messages trimming will never
// drop below. Falling under it would leave the model without the exchange it is
// answering.
const contextMinimumKept = 2

// EstimateChars estimates the character cost of a conversation.
//
// Each message costs the length of its role name plus a fixed formatting
// overhead, plus the length of every text part, plus a flat charge per non-text
// part, plus the encoded length of any tool_calls metadata — those are sent to
// the provider verbatim and are frequently the largest thing in an agent
// conversation.
func EstimateChars(messages []model.Message) int {
	total := 0
	for _, message := range messages {
		total += len(string(message.Role)) + contextRoleOverhead
		for _, part := range message.Parts {
			switch typed := part.(type) {
			case model.TextPart:
				total += len(typed.Value)
			case *model.TextPart:
				if typed != nil {
					total += len(typed.Value)
				}
			default:
				total += contextPartCost
			}
		}
		if toolCalls, ok := message.Metadata["tool_calls"]; ok && toolCalls != nil {
			if encoded, err := json.Marshal(toolCalls); err == nil {
				total += len(encoded)
			}
		}
	}
	return total
}

// SummarizeDropped renders the messages trimming removed into the placeholder
// text that takes their place.
//
// The summary is deliberately mechanical: it quotes each dropped message rather
// than paraphrasing, so it costs no model call and stays deterministic. A host
// that wants a real précis supplies RunOptions.Compaction, which replaces this
// text with a model- or function-generated one.
func SummarizeDropped(messages []model.Message) string {
	if len(messages) == 0 {
		return ""
	}
	lines := make([]string, 0, len(messages))
	for _, message := range messages {
		role := string(message.Role)
		text := message.Text()
		if text == "" {
			lines = append(lines, fmt.Sprintf("[%s message]", role))
			continue
		}
		lines = append(lines, fmt.Sprintf("[%s]: %s", role, truncateRunes(text, contextSummaryTruncate)))
	}
	return truncateRunes(strings.Join(lines, "\n"), contextSummaryCap)
}

// truncateRunes cuts text to at most limit bytes and appends an ellipsis,
// backing up to a rune boundary so a multi-byte character is never split.
//
// A limit of zero or less removes the text entirely rather than leaving a bare
// ellipsis, so a caller that reserved no space really gets none.
func truncateRunes(text string, limit int) string {
	if limit <= 0 {
		return ""
	}
	if len(text) <= limit {
		return text
	}
	cut := limit
	for cut > 0 && !utf8StartsRune(text[cut]) {
		cut--
	}
	return text[:cut] + "..."
}

// utf8StartsRune reports whether b can begin a UTF-8 encoded rune, i.e. it is
// not a continuation byte.
func utf8StartsRune(b byte) bool { return b&0xC0 != 0x80 }

// TrimToContextWindow trims a conversation towards a character budget and
// returns the dropped messages alongside the conversation to send.
//
// The rules, shared across runtimes:
//
//  1. Leading system messages are always preserved — every one of them, however
//     tight the budget. They carry the agent's instructions.
//  2. Non-system messages are dropped oldest-first.
//  3. At least contextMinimumKept non-system messages survive; a budget too
//     small to honour that leaves the conversation untouched rather than
//     producing something the model cannot answer.
//  4. When anything was dropped, a synthetic system message summarising it is
//     inserted directly after the preserved system messages, so the model is
//     told that history was elided instead of silently losing it. The summary
//     is capped at the space reserved for it — 5% of the budget, at most 5000
//     characters — so it cannot itself blow the budget.
//
// # The budget is a target, not a guarantee
//
// Rules 1 and 3 are floors that outrank the budget: a conversation whose system
// messages plus two surviving turns already exceed it will come back over
// budget, because the alternative is a prompt the model cannot answer. Callers
// sizing a budget should leave headroom for the system prompt and one exchange.
// This is deliberate and matches every other Prompty runtime, so the shared
// context vectors converge.
//
// A conversation already inside the budget is returned unchanged with no
// dropped messages, so the caller can cheaply test whether trimming happened.
func TrimToContextWindow(messages []model.Message, budgetChars int) (dropped []model.Message, trimmed []model.Message) {
	if budgetChars <= 0 || EstimateChars(messages) <= budgetChars {
		return nil, messages
	}

	systemCount := 0
	for systemCount < len(messages) && messages[systemCount].Role == model.RoleSystem {
		systemCount++
	}
	systemMessages := messages[:systemCount]
	rest := messages[systemCount:]

	if len(rest) <= contextMinimumKept {
		return nil, messages
	}

	systemChars := EstimateChars(systemMessages)
	// The summary is reserved a fixed share of the budget: 5%, capped at 5000
	// characters. Reserving it here is what stops the placeholder that replaces
	// the dropped history from pushing the result straight back over budget.
	summaryBudget := contextSummaryReserve(budgetChars)
	available := budgetChars - (systemChars + summaryBudget)
	if available < 0 {
		available = 0
	}

	dropCount := 0
	restChars := EstimateChars(rest)
	maxDrops := len(rest) - contextMinimumKept
	for restChars > available && dropCount < maxDrops {
		restChars -= EstimateChars(rest[dropCount : dropCount+1])
		dropCount++
	}
	if dropCount == 0 {
		return nil, messages
	}

	dropped = append([]model.Message(nil), rest[:dropCount]...)
	kept := rest[dropCount:]

	// The summary is held to the space the loop above already reserved for it.
	// Without this cap the placeholder — up to contextSummaryCap characters —
	// can dwarf a small budget, so trimming would reliably overshoot the number
	// it was given.
	summary := ContextSummaryMessage(truncateRunes(SummarizeDropped(dropped), summaryBudget), dropCount)

	trimmed = make([]model.Message, 0, len(systemMessages)+1+len(kept))
	trimmed = append(trimmed, systemMessages...)
	trimmed = append(trimmed, summary)
	trimmed = append(trimmed, kept...)
	return dropped, trimmed
}

// contextSummaryReserve is the space trimming sets aside for the summary that
// replaces dropped history: 5% of the budget, never more than 5000 characters.
// The share is small because the summary is a marker, not a substitute for the
// conversation it replaces.
func contextSummaryReserve(budgetChars int) int {
	reserve := budgetChars / 20
	if reserve > 5000 {
		reserve = 5000
	}
	return reserve
}

// ContextSummaryMessage builds the synthetic message that stands in for trimmed
// history.
//
// The role is system rather than user: it is the harness speaking about the
// conversation, not the user speaking in it, and a user-role placeholder would
// be answered as if the user had typed it.
func ContextSummaryMessage(summary string, droppedCount int) model.Message {
	text := fmt.Sprintf("[Summary of earlier conversation] %s\n... (%d messages omitted)", summary, droppedCount)
	message := model.NewSystemMessage(text)
	message.Metadata = map[string]interface{}{
		"promptyContextSummary": true,
		"droppedMessages":       droppedCount,
	}
	return message
}

// FormatDroppedMessages renders dropped messages as the transcript handed to a
// compaction strategy. Tool calls are shown as `Called: name(args)` so a
// summariser can see what the agent did, not just what it said.
func FormatDroppedMessages(messages []model.Message) string {
	var lines []string
	for _, message := range messages {
		role := string(message.Role)
		text := message.Text()

		if raw, ok := message.Metadata["tool_calls"]; ok && raw != nil {
			for _, call := range toolCallSummaries(raw) {
				lines = append(lines, fmt.Sprintf("[%s]: Called: %s", role, call))
			}
		}

		if text != "" {
			lines = append(lines, fmt.Sprintf("[%s]: %s", role, text))
		} else if len(lines) == 0 || !strings.HasPrefix(lines[len(lines)-1], "["+role+"]") {
			lines = append(lines, fmt.Sprintf("[%s message]", role))
		}
	}
	return strings.Join(lines, "\n")
}

// toolCallSummaries renders a tool_calls metadata value as `name(args)` strings.
// It tolerates both the flat shape ({name, arguments}) and the OpenAI nested
// shape ({function: {name, arguments}}) because both appear in host metadata.
func toolCallSummaries(raw interface{}) []string {
	list, ok := raw.([]interface{})
	if !ok {
		return nil
	}
	out := make([]string, 0, len(list))
	for _, item := range list {
		entry, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		if nested, ok := entry["function"].(map[string]interface{}); ok {
			entry = nested
		}
		name, _ := entry["name"].(string)
		if name == "" {
			name = "unknown"
		}
		arguments, _ := entry["arguments"].(string)
		if arguments == "" {
			arguments = "{}"
		}
		out = append(out, fmt.Sprintf("%s(%s)", name, arguments))
	}
	return out
}

// CompactionFunc replaces the mechanical summary of dropped messages with a
// better one — typically produced by a small model or a host-side summariser.
//
// It receives the messages that were dropped, already ordered oldest-first, and
// returns the replacement summary text. Returning an error leaves the
// mechanical summary in place: compaction is an optimisation, and failing to
// improve a summary must never fail the turn.
type CompactionFunc func(ctx context.Context, dropped []model.Message) (string, error)

// applyContextPolicy trims a conversation for one iteration and reports what it
// removed. It is the single place the loop calls, so trimming and compaction
// stay in step.
func applyContextPolicy(
	ctx context.Context,
	messages []model.Message,
	budget int,
	compaction CompactionFunc,
) (trimmed []model.Message, droppedCount int) {
	if budget <= 0 {
		return messages, 0
	}
	dropped, result := TrimToContextWindow(messages, budget)
	if len(dropped) == 0 {
		return result, 0
	}
	if compaction != nil {
		if summary, err := compaction(ctx, dropped); err == nil && summary != "" {
			// A compacted summary is held to the same reserve as the
			// mechanical one; a summariser that returns an essay must not be
			// able to defeat the budget the host asked for.
			replacement := ContextSummaryMessage(truncateRunes(summary, contextSummaryReserve(budget)), len(dropped))
			// The summary always sits directly after the preserved system
			// messages, which is where TrimToContextWindow put it.
			for index := range result {
				if isContextSummaryMessage(result[index]) {
					result[index] = replacement
					break
				}
			}
		}
	}
	return result, len(dropped)
}

func isContextSummaryMessage(message model.Message) bool {
	flag, ok := message.Metadata["promptyContextSummary"].(bool)
	return ok && flag
}
