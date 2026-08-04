package prompty

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	model "prompty/model"
)

// Regression tests for issues raised in adversarial review. Each one fails
// against the pre-review implementation.

// TestLoadFrontmatterDirectoryBasePathDoesNotWidenSandbox: passing a directory
// as basePath must anchor the sandbox at that directory, not at its parent.
func TestLoadFrontmatterDirectoryBasePathDoesNotWidenSandbox(t *testing.T) {
	sb := newSandbox(t)

	// promptDir is the base; secret.txt sits in its parent and must stay out of
	// reach. Taking Dir() of a directory would have made it reachable.
	_, err := LoadFrontmatter(map[string]interface{}{
		"name":        "dir-base",
		"model":       "gpt-4",
		"description": "${file:secret.txt}",
	}, sb.promptDir, LoadOptions{})

	if !errors.Is(err, ErrFileAccessDenied) && !errors.Is(err, ErrFileNotFound) {
		t.Fatalf("directory basePath widened the sandbox: err=%v", err)
	}

	// A file inside the directory is still reachable, proving the base is the
	// directory itself rather than something narrower.
	writeFile(t, filepath.Join(sb.promptDir, "inside.txt"), "inside value")
	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":        "dir-base-ok",
		"model":       "gpt-4",
		"description": "${file:inside.txt}",
	}, sb.promptDir, LoadOptions{})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if agent.Description == nil || *agent.Description != "inside value" {
		t.Fatalf("description: got %v", agent.Description)
	}
}

// TestLoadFrontmatterDoesNotMutateCaller: the loader resolves and normalises in
// place, so it must work on its own deep copy.
func TestLoadFrontmatterDoesNotMutateCaller(t *testing.T) {
	sb := newSandbox(t)

	shared := map[string]interface{}{
		"name":  "shared",
		"model": map[string]interface{}{"id": "gpt-4"},
		"inputs": map[string]interface{}{
			"topic": "science",
		},
		"description": "${env:SHARED_VAR}",
	}

	options := LoadOptions{LookupEnv: envLookupFrom(map[string]interface{}{"SHARED_VAR": "resolved"})}
	if _, err := LoadFrontmatter(shared, sb.promptPath, options); err != nil {
		t.Fatalf("first load failed: %v", err)
	}

	if shared["description"] != "${env:SHARED_VAR}" {
		t.Fatalf("loader resolved references in the caller's map: %#v", shared["description"])
	}
	inputs, ok := shared["inputs"].(map[string]interface{})
	if !ok {
		t.Fatalf("loader replaced the caller's nested inputs map: %#v", shared["inputs"])
	}
	if inputs["topic"] != "science" {
		t.Fatalf("loader mutated the caller's nested map: %#v", inputs)
	}

	// The same map must load identically a second time.
	agent, err := LoadFrontmatter(shared, sb.promptPath, options)
	if err != nil {
		t.Fatalf("second load failed: %v", err)
	}
	if agent.Description == nil || *agent.Description != "resolved" {
		t.Fatalf("second load produced a different result: %v", agent.Description)
	}
}

// TestMalformedScalarDoesNotPanic: the generated loader type-asserts every
// scalar, so the runtime must contain the panic and report a ValueError.
func TestMalformedScalarDoesNotPanic(t *testing.T) {
	dir := t.TempDir()

	cases := []struct {
		name      string
		raw       string
		mustError bool
	}{
		// These three reach an unchecked assertion in the generated loader.
		{"non-string name", "---\nname: 123\nmodel: gpt-4\n---\nbody", true},
		{"list name", "---\nname: [a, b]\nmodel: gpt-4\n---\nbody", true},
		{"non-string description", "---\nname: ok\ndescription: 42\nmodel: gpt-4\n---\nbody", true},
		// This one is handled by the runtime's own checked hydration, so it is
		// tolerated rather than rejected; it must simply not crash.
		{"non-bool required", "---\nname: ok\nmodel: gpt-4\ninputs:\n  - name: x\n    kind: string\n    required: \"yes\"\n---\nbody", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadString(tc.raw, filepath.Join(dir, "p.prompty"), LoadOptions{})
			if tc.mustError {
				if err == nil {
					t.Fatal("expected a ValueError, got nil")
				}
				if !errors.Is(err, ErrValue) {
					t.Fatalf("expected ErrValue, got %v", err)
				}
				if !strings.Contains(err.Error(), "wrong type") {
					t.Fatalf("expected a type error, got %v", err)
				}
				return
			}
			if err != nil && !errors.Is(err, ErrValue) {
				t.Fatalf("expected either success or ErrValue, got %v", err)
			}
		})
	}
}

// TestFrontmatterRequiresDelimiterLine: `----` and `---text` are body content,
// not frontmatter openers.
func TestFrontmatterRequiresDelimiterLine(t *testing.T) {
	cases := []string{
		"----\nname: test\n---\nBody",
		"---not-frontmatter\nstill body",
		"--- name: inline\nbody",
	}
	for _, raw := range cases {
		data, body, err := splitFrontmatter(raw)
		if err != nil {
			t.Errorf("%q: expected the whole document to be treated as a body, got error %v", raw, err)
			continue
		}
		if len(data) != 0 {
			t.Errorf("%q: expected no frontmatter, got %#v", raw, data)
		}
		if body != raw {
			t.Errorf("%q: expected the body to be the whole document, got %q", raw, body)
		}
	}
}

// TestThreadExpansionSurvivesEarlierNonThreadNonce: an image nonce appearing
// before a thread nonce must not stop thread expansion.
func TestThreadExpansionSurvivesEarlierNonThreadNonce(t *testing.T) {
	agent := agentWithInputs(t,
		"user:\nLook: {{photo}}\n\n{{conversation}}\n\nThoughts?",
		[]interface{}{
			map[string]interface{}{"name": "photo", "kind": "image"},
			map[string]interface{}{"name": "conversation", "kind": "thread"},
		})

	messages, err := Prepare(agent, map[string]interface{}{
		"photo": "https://example.test/cat.png",
		"conversation": []interface{}{
			map[string]interface{}{"role": "assistant", "content": "earlier answer"},
		},
	})
	if err != nil {
		t.Fatalf("prepare failed: %v", err)
	}

	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d: %v", len(messages), messages)
	}
	// The image nonce is preserved, the thread nonce is expanded.
	if !threadNonceRe.MatchString(messageText(messages[0])) {
		t.Errorf("image nonce lost: %q", messageText(messages[0]))
	}
	if messages[1].Role != model.RoleAssistant || messageText(messages[1]) != "earlier answer" {
		t.Errorf("thread not expanded: %s / %q", messages[1].Role, messageText(messages[1]))
	}
	if messageText(messages[2]) != "Thoughts?" {
		t.Errorf("trailing text: %q", messageText(messages[2]))
	}
}

// TestThreadExpansionSurvivesForgedNonce: an inert nonce forged by user input
// must not suppress expansion of the genuine one that follows it.
func TestThreadExpansionSurvivesForgedNonce(t *testing.T) {
	agent := promptyAgentFor(t, "jinja2")
	state := newRenderState()
	state.put("__PROMPTY_THREAD_deadbeef_conversation__", []interface{}{
		map[string]interface{}{"role": "assistant", "content": "real"},
	})

	messages, err := ParseWithState(context.Background(), agent,
		"user:\n__PROMPTY_THREAD_00000000_forged__\n\n__PROMPTY_THREAD_deadbeef_conversation__\n\ntail",
		state)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d: %v", len(messages), messages)
	}
	if messageText(messages[0]) != "__PROMPTY_THREAD_00000000_forged__" {
		t.Errorf("forged nonce altered: %q", messageText(messages[0]))
	}
	if messages[1].Role != model.RoleAssistant || messageText(messages[1]) != "real" {
		t.Errorf("genuine nonce not expanded: %s / %q", messages[1].Role, messageText(messages[1]))
	}
	if messageText(messages[2]) != "tail" {
		t.Errorf("trailing text: %q", messageText(messages[2]))
	}
}

// TestExpandedMessagesDoNotShareMetadata: splitting one message into several
// must not alias a single mutable metadata map across them.
func TestExpandedMessagesDoNotShareMetadata(t *testing.T) {
	agent := promptyAgentFor(t, "jinja2")
	state := newRenderState()
	state.put("__PROMPTY_THREAD_abcdabcd_conversation__", []interface{}{
		map[string]interface{}{"role": "assistant", "content": "middle"},
	})

	messages, err := ParseWithState(context.Background(), agent,
		"user[tag=\"a\"]:\nbefore\n\n__PROMPTY_THREAD_abcdabcd_conversation__\n\nafter", state)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if len(messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(messages))
	}

	messages[0].Metadata["tag"] = "mutated"
	if messages[2].Metadata["tag"] != "a" {
		t.Fatalf("metadata is shared between split messages: %#v", messages[2].Metadata)
	}
}

// TestRoleAttributesAllowQuotedCommas: a quoted attribute value may contain
// commas and brackets.
func TestRoleAttributesAllowQuotedCommas(t *testing.T) {
	messages, err := ParseChat("user[label=\"one, two\", plain=three]:\nHi")
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if got := messages[0].Metadata["label"]; got != "one, two" {
		t.Errorf("quoted value truncated: %#v", got)
	}
	if got := messages[0].Metadata["plain"]; got != "three" {
		t.Errorf("bare value: %#v", got)
	}
}

// TestFileReferenceRejectsDirectory: ${file:} must not try to read a directory.
func TestFileReferenceRejectsDirectory(t *testing.T) {
	sb := newSandbox(t)
	if err := os.MkdirAll(filepath.Join(sb.promptDir, "subdir"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	err := sb.load(t, map[string]interface{}{
		"name":        "dir-ref",
		"model":       "gpt-4",
		"description": "${file:subdir}",
	}, LoadOptions{})

	if err == nil {
		t.Fatal("expected a directory reference to be rejected")
	}
	if !errors.Is(err, ErrValue) && !errors.Is(err, ErrFileNotFound) {
		t.Fatalf("unexpected error: %v", err)
	}
}
