package prompty

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeFile creates a file, making parent directories as needed.
func writeFile(t *testing.T, path, content string) string {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	return path
}

// sandbox builds a workspace with a prompt directory and a sibling directory
// that is deliberately outside the prompt's default sandbox.
type sandbox struct {
	root       string
	promptDir  string
	promptPath string
	sharedDir  string
}

func newSandbox(t *testing.T) sandbox {
	t.Helper()
	root := t.TempDir()
	sb := sandbox{
		root:      root,
		promptDir: filepath.Join(root, "prompts"),
		sharedDir: filepath.Join(root, "shared"),
	}
	sb.promptPath = filepath.Join(sb.promptDir, "agent.prompty")
	if err := os.MkdirAll(sb.promptDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.MkdirAll(sb.sharedDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	writeFile(t, filepath.Join(root, "secret.txt"), "TOP SECRET")
	writeFile(t, filepath.Join(sb.sharedDir, "shared.txt"), "shared value")
	return sb
}

func (sb sandbox) load(t *testing.T, frontmatter map[string]interface{}, options LoadOptions) error {
	t.Helper()
	_, err := LoadFrontmatter(frontmatter, sb.promptPath, options)
	return err
}

// TestFileReferenceRejectsTraversal is the core sandbox guarantee: frontmatter
// cannot walk out of the prompt's own directory with `..`.
func TestFileReferenceRejectsTraversal(t *testing.T) {
	sb := newSandbox(t)

	err := sb.load(t, map[string]interface{}{
		"name":        "traversal",
		"model":       "gpt-4",
		"description": "${file:../secret.txt}",
	}, LoadOptions{})

	if err == nil {
		t.Fatal("expected traversal to be rejected")
	}
	if !errors.Is(err, ErrFileAccessDenied) {
		t.Fatalf("expected ErrFileAccessDenied, got %v", err)
	}
	if strings.Contains(err.Error(), "TOP SECRET") {
		t.Fatal("error message leaked the file contents")
	}
}

// TestFileReferenceRejectsAbsolutePath covers the second half of the §2.11 rule:
// an absolute path is refused even when it happens to exist.
func TestFileReferenceRejectsAbsolutePath(t *testing.T) {
	sb := newSandbox(t)
	secret := filepath.Join(sb.root, "secret.txt")

	err := sb.load(t, map[string]interface{}{
		"name":        "absolute",
		"model":       "gpt-4",
		"description": "${file:" + secret + "}",
	}, LoadOptions{})

	if !errors.Is(err, ErrFileAccessDenied) {
		t.Fatalf("expected ErrFileAccessDenied, got %v", err)
	}
}

// TestFileReferenceAllowsExplicitRoot shows the host escape hatch working: the
// same reference that failed above succeeds once the application opts in.
func TestFileReferenceAllowsExplicitRoot(t *testing.T) {
	sb := newSandbox(t)

	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":        "explicit-root",
		"model":       "gpt-4",
		"description": "${file:../shared/shared.txt}",
	}, sb.promptPath, LoadOptions{AllowedFileRoots: []string{sb.sharedDir}})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if agent.Description == nil || *agent.Description != "shared value" {
		t.Fatalf("description: got %v, want %q", agent.Description, "shared value")
	}
}

// TestFileReferenceAllowsAbsolutePathInsideExplicitRoot confirms absolute paths
// are honoured only inside a host-supplied root, never inside the implicit
// prompt-directory root.
func TestFileReferenceAllowsAbsolutePathInsideExplicitRoot(t *testing.T) {
	sb := newSandbox(t)
	shared := filepath.Join(sb.sharedDir, "shared.txt")

	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":        "absolute-allowed",
		"model":       "gpt-4",
		"description": "${file:" + shared + "}",
	}, sb.promptPath, LoadOptions{AllowedFileRoots: []string{sb.sharedDir}})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if agent.Description == nil || *agent.Description != "shared value" {
		t.Fatalf("description: got %v", agent.Description)
	}
}

// TestFileReferenceRejectsSymlinkEscape covers the case a purely lexical check
// would miss: a link that lives inside the sandbox but points outside it.
func TestFileReferenceRejectsSymlinkEscape(t *testing.T) {
	sb := newSandbox(t)
	link := filepath.Join(sb.promptDir, "escape.txt")
	if err := os.Symlink(filepath.Join(sb.root, "secret.txt"), link); err != nil {
		t.Skipf("symlinks unavailable on this host: %v", err)
	}

	err := sb.load(t, map[string]interface{}{
		"name":        "symlink",
		"model":       "gpt-4",
		"description": "${file:escape.txt}",
	}, LoadOptions{})

	if !errors.Is(err, ErrFileAccessDenied) {
		t.Fatalf("expected ErrFileAccessDenied, got %v", err)
	}
}

// TestFileReferenceRecursesIntoLoadedContent covers the recursion requirement:
// a referenced JSON document is itself scanned for references.
func TestFileReferenceRecursesIntoLoadedContent(t *testing.T) {
	sb := newSandbox(t)
	writeFile(t, filepath.Join(sb.promptDir, "connection.json"),
		`{"kind":"key","endpoint":"https://example.test","apiKey":"${env:NESTED_KEY}"}`)

	agent, err := LoadFrontmatter(map[string]interface{}{
		"name": "recursive",
		"model": map[string]interface{}{
			"id":         "gpt-4",
			"connection": "${file:connection.json}",
		},
	}, sb.promptPath, LoadOptions{
		LookupEnv: envLookupFrom(map[string]interface{}{"NESTED_KEY": "resolved-secret"}),
	})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	saved := saveAgent(agent)
	conn := vectorMap(vectorMap(saved, "model"), "connection")
	if got := vectorString(conn, "apiKey"); got != "resolved-secret" {
		t.Fatalf("nested ${env:} not resolved: got %q", got)
	}
}

// TestFileReferenceDetectsCycle proves the recursion above terminates.
func TestFileReferenceDetectsCycle(t *testing.T) {
	sb := newSandbox(t)
	writeFile(t, filepath.Join(sb.promptDir, "a.json"), `{"next":"${file:b.json}"}`)
	writeFile(t, filepath.Join(sb.promptDir, "b.json"), `{"next":"${file:a.json}"}`)

	err := sb.load(t, map[string]interface{}{
		"name":     "cycle",
		"model":    "gpt-4",
		"metadata": map[string]interface{}{"chain": "${file:a.json}"},
	}, LoadOptions{})

	if err == nil {
		t.Fatal("expected a cycle to be reported")
	}
	if !strings.Contains(err.Error(), "Circular file reference") {
		t.Fatalf("expected a circular-reference error, got %v", err)
	}
}

// TestFileReferenceParsesByExtension covers the §2.11 extension rules.
func TestFileReferenceParsesByExtension(t *testing.T) {
	sb := newSandbox(t)
	writeFile(t, filepath.Join(sb.promptDir, "conf.json"), `{"a":1}`)
	writeFile(t, filepath.Join(sb.promptDir, "conf.yaml"), "b: two\n")
	writeFile(t, filepath.Join(sb.promptDir, "note.txt"), "plain text")

	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":  "extensions",
		"model": "gpt-4",
		"metadata": map[string]interface{}{
			"json": "${file:conf.json}",
			"yaml": "${file:conf.yaml}",
			"text": "${file:note.txt}",
		},
	}, sb.promptPath, LoadOptions{})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	jsonValue, ok := agent.Metadata["json"].(map[string]interface{})
	if !ok {
		t.Fatalf("json reference did not parse into a mapping: %#v", agent.Metadata["json"])
	}
	if !scalarEqual(jsonValue["a"], int64(1)) {
		t.Fatalf("json value: got %#v, want 1", jsonValue["a"])
	}

	yamlValue, ok := agent.Metadata["yaml"].(map[string]interface{})
	if !ok {
		t.Fatalf("yaml reference did not parse into a mapping: %#v", agent.Metadata["yaml"])
	}
	if yamlValue["b"] != "two" {
		t.Fatalf("yaml value: got %#v", yamlValue["b"])
	}

	if agent.Metadata["text"] != "plain text" {
		t.Fatalf("text reference: got %#v", agent.Metadata["text"])
	}
}

// TestFileReferenceMissingFile checks the error type the spec pins for a missing
// ${file:} target.
func TestFileReferenceMissingFile(t *testing.T) {
	sb := newSandbox(t)

	err := sb.load(t, map[string]interface{}{
		"name":        "missing",
		"model":       "gpt-4",
		"description": "${file:nope.json}",
	}, LoadOptions{})

	if !errors.Is(err, ErrFileNotFound) {
		t.Fatalf("expected ErrFileNotFound, got %v", err)
	}
}

// TestEnvReferenceDefaultKeepsColons guards the "everything after the second
// colon" rule, which is what makes a URL default work at all.
func TestEnvReferenceDefaultKeepsColons(t *testing.T) {
	sb := newSandbox(t)

	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":        "env-default",
		"model":       "gpt-4",
		"description": "${env:UNSET_ENDPOINT:https://api.openai.com/v1}",
	}, sb.promptPath, LoadOptions{
		LookupEnv: envLookupFrom(map[string]interface{}{}),
	})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if agent.Description == nil || *agent.Description != "https://api.openai.com/v1" {
		t.Fatalf("description: got %v", agent.Description)
	}
}

// TestEnvReferenceEmptyDefault documents that an explicitly empty default is a
// default, not a missing one.
func TestEnvReferenceEmptyDefault(t *testing.T) {
	sb := newSandbox(t)

	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":        "env-empty-default",
		"model":       "gpt-4",
		"description": "${env:UNSET_VAR:}",
	}, sb.promptPath, LoadOptions{LookupEnv: envLookupFrom(map[string]interface{}{})})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if agent.Description == nil || *agent.Description != "" {
		t.Fatalf("description: got %v, want empty string", agent.Description)
	}
}

// TestUnknownProtocolIsLeftAlone: an unrecognised ${...} is data, not an error.
func TestUnknownProtocolIsLeftAlone(t *testing.T) {
	sb := newSandbox(t)

	agent, err := LoadFrontmatter(map[string]interface{}{
		"name":        "unknown-protocol",
		"model":       "gpt-4",
		"description": "${vault:secret/path}",
	}, sb.promptPath, LoadOptions{})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if agent.Description == nil || *agent.Description != "${vault:secret/path}" {
		t.Fatalf("description: got %v", agent.Description)
	}
}

// TestJinja2RendererCannotReadFiles asserts the template sandbox: an include can
// never reach the host filesystem.
func TestJinja2RendererCannotReadFiles(t *testing.T) {
	sb := newSandbox(t)
	secret := filepath.ToSlash(filepath.Join(sb.root, "secret.txt"))

	renderer := NewJinja2Renderer()
	for _, template := range []string{
		`{% include "` + secret + `" %}`,
		`{% extends "` + secret + `" %}`,
		`{% include "../secret.txt" %}`,
	} {
		out, err := renderer.Render(promptyAgentFor(t, "jinja2"), template, nil)
		if err == nil && strings.Contains(out, "TOP SECRET") {
			t.Fatalf("template %q leaked file contents", template)
		}
		if strings.Contains(out, "TOP SECRET") {
			t.Fatalf("template %q leaked file contents", template)
		}
	}
}

// TestMustacheRendererCannotReadFiles asserts the same for mustache partials,
// whose library default is a filesystem provider.
func TestMustacheRendererCannotReadFiles(t *testing.T) {
	sb := newSandbox(t)
	writeFile(t, filepath.Join(sb.promptDir, "secret.mustache"), "TOP SECRET")

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(sb.promptDir); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	t.Cleanup(func() { _ = os.Chdir(cwd) })

	out, err := NewMustacheRenderer().Render(promptyAgentFor(t, "mustache"), "{{> secret}}", nil)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}
	if strings.Contains(out, "TOP SECRET") {
		t.Fatal("mustache partial leaked file contents")
	}
}

// TestMustacheRendererDoesNotEscapeHTML: .prompty templates are not HTML.
func TestMustacheRendererDoesNotEscapeHTML(t *testing.T) {
	out, err := NewMustacheRenderer().Render(
		promptyAgentFor(t, "mustache"),
		"{{content}}",
		map[string]interface{}{"content": "<b>bold</b> & \"quoted\""},
	)
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}
	if out != `<b>bold</b> & "quoted"` {
		t.Fatalf("mustache escaped output: got %q", out)
	}
}
