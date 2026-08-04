package prompty_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	model "prompty/model"
	openai "prompty/openai"
)

// specRoot walks up to the repository's shared spec/ directory.
func specRoot(t *testing.T) string {
	t.Helper()

	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		candidate := filepath.Join(dir, "spec")
		if info, err := os.Stat(filepath.Join(candidate, "vectors")); err == nil && info.IsDir() {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skip("shared spec/ directory not found; skipping shared-vector tests")
			return ""
		}
		dir = parent
	}
}

func readVectorFile(t *testing.T, relative string, target interface{}) {
	t.Helper()

	path := filepath.Join(specRoot(t), "vectors", filepath.FromSlash(relative))
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(raw, target); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}

// scriptedExecutor replays a recorded sequence of provider responses.
//
// It is a real Executor in every respect except transport: FormatToolMessages
// is delegated to the genuine OpenAI executor, so the message shapes the loop
// produces are the ones a live provider would receive, not a test fiction.
type scriptedExecutor struct {
	responses []interface{}
	calls     int
	// onCall runs before each response is handed back, so a test can cancel
	// mid-turn.
	onCall func(call int)
	// seen records the conversation as it was at each call, which is how the
	// message-sequence assertions inspect what the provider would have seen.
	seen [][]model.Message

	delegate openai.Executor
}

func (e *scriptedExecutor) ExecuteContext(ctx context.Context, agent model.Prompty, messages []model.Message) (interface{}, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if e.onCall != nil {
		e.onCall(e.calls)
	}
	e.seen = append(e.seen, append([]model.Message(nil), messages...))
	if e.calls >= len(e.responses) {
		return nil, errScriptExhausted
	}
	response := e.responses[e.calls]
	e.calls++
	return response, nil
}

func (e *scriptedExecutor) ExecuteStreamContext(context.Context, model.Prompty, []model.Message) (interface{}, error) {
	return nil, errScriptExhausted
}

func (e *scriptedExecutor) Execute(agent model.Prompty, messages []model.Message) (interface{}, error) {
	return e.ExecuteContext(context.Background(), agent, messages)
}

func (e *scriptedExecutor) ExecuteStream(model.Prompty, []model.Message) (interface{}, error) {
	return nil, errScriptExhausted
}

func (e *scriptedExecutor) FormatToolMessages(raw interface{}, calls []model.ToolCall, results []string, text *string) ([]model.Message, error) {
	return e.delegate.FormatToolMessages(raw, calls, results, text)
}

var errScriptExhausted = &scriptError{"scripted provider ran out of responses"}

type scriptError struct{ message string }

func (e *scriptError) Error() string { return e.message }

// messageView is the flattened shape the agent vectors describe a message in.
type messageView struct {
	Role     string
	Content  string
	ToolCall string
	Calls    []interface{}
}

func viewMessage(msg model.Message) messageView {
	view := messageView{Role: string(msg.Role), Content: msg.Text()}
	if id, ok := msg.Metadata["tool_call_id"].(string); ok {
		view.ToolCall = id
	}
	if calls, ok := msg.Metadata["tool_calls"].([]interface{}); ok {
		view.Calls = calls
	}
	return view
}

func viewExpectedMessage(t *testing.T, raw interface{}) messageView {
	t.Helper()

	entry, ok := raw.(map[string]interface{})
	if !ok {
		t.Fatalf("expected message is not an object: %T", raw)
	}
	view := messageView{}
	view.Role, _ = entry["role"].(string)

	switch content := entry["content"].(type) {
	case string:
		view.Content = content
	case []interface{}:
		// tool_result_message spells content as typed blocks.
		for _, item := range content {
			if block, ok := item.(map[string]interface{}); ok {
				if text, ok := block["text"].(string); ok {
					view.Content += text
				}
			}
		}
	}

	if metadata, ok := entry["metadata"].(map[string]interface{}); ok {
		if id, ok := metadata["tool_call_id"].(string); ok {
			view.ToolCall = id
		}
		if calls, ok := metadata["tool_calls"].([]interface{}); ok {
			view.Calls = calls
		}
	}
	return view
}

func assertMessageView(t *testing.T, label string, got, want messageView) {
	t.Helper()

	if got.Role != want.Role {
		t.Errorf("%s: role = %q, want %q", label, got.Role, want.Role)
	}
	if got.Content != want.Content {
		t.Errorf("%s: content = %q, want %q", label, got.Content, want.Content)
	}
	if got.ToolCall != want.ToolCall {
		t.Errorf("%s: tool_call_id = %q, want %q", label, got.ToolCall, want.ToolCall)
	}
	if want.Calls != nil {
		gotJSON := mustJSON(t, got.Calls)
		wantJSON := mustJSON(t, want.Calls)
		if !reflect.DeepEqual(gotJSON, wantJSON) {
			t.Errorf("%s: tool_calls mismatch\ngot  %s\nwant %s",
				label, mustEncode(t, gotJSON), mustEncode(t, wantJSON))
		}
	}
}

func mustJSON(t *testing.T, v interface{}) interface{} {
	t.Helper()

	encoded, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded interface{}
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return decoded
}

func mustEncode(t *testing.T, v interface{}) string {
	t.Helper()

	encoded, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(encoded)
}
