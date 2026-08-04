package prompty

import (
	"strings"

	"gopkg.in/yaml.v3"
)

// frontmatterDelimiters are the two delimiter forms recognised at the head of a
// .prompty file (spec §2.1). The opening and closing delimiter need not match.
var frontmatterDelimiters = [...]string{"---", "+++"}

// splitFrontmatter splits raw .prompty content into its parsed YAML frontmatter
// mapping and its markdown body (spec §2.2).
//
// Behaviour, in the order the spec fixes it:
//
//  1. No delimiter at the start (ignoring leading whitespace) -> no frontmatter,
//     the entire content is the body.
//  2. Opening delimiter with no closing delimiter -> ValueError.
//  3. Frontmatter that is not a YAML mapping -> ValueError.
//
// The body is returned untrimmed; the caller decides how to normalise it.
func splitFrontmatter(raw string) (map[string]interface{}, string, error) {
	trimmed := strings.TrimLeft(raw, " \t\r\n")
	if !hasFrontmatterDelimiter(trimmed) {
		return map[string]interface{}{}, raw, nil
	}

	// Skip past the opening delimiter line.
	newline := strings.IndexByte(trimmed[3:], '\n')
	if newline < 0 {
		// A lone delimiter with no newline: empty frontmatter, empty body.
		return map[string]interface{}{}, "", nil
	}
	rest := trimmed[3+newline+1:]

	closeAt, closeLen := findClosingDelimiter(rest)
	if closeAt < 0 {
		return nil, "", newValueError("invalid frontmatter: opening delimiter without a closing match")
	}

	yamlText := rest[:closeAt]
	body := ""
	if after := rest[closeAt+closeLen:]; after != "" {
		if nl := strings.IndexByte(after, '\n'); nl >= 0 {
			body = after[nl+1:]
		}
	}

	data, err := parseFrontmatterYAML(yamlText)
	if err != nil {
		return nil, "", err
	}
	return data, body, nil
}

func hasFrontmatterDelimiter(s string) bool {
	// The delimiter must be alone on the first line: `----` and `---x` are
	// ordinary body text, not frontmatter openers.
	line := s
	if nl := strings.IndexByte(s, '\n'); nl >= 0 {
		line = s[:nl]
	}
	return isDelimiterLine(line)
}

// findClosingDelimiter returns the byte offset of the closing delimiter line and
// the length of that line, or (-1, 0) when there is none. The delimiter must be
// alone on its line, modulo surrounding whitespace.
func findClosingDelimiter(text string) (int, int) {
	offset := 0
	for {
		lineEnd := strings.IndexByte(text[offset:], '\n')
		var line string
		if lineEnd < 0 {
			line = text[offset:]
		} else {
			line = text[offset : offset+lineEnd]
		}
		if isDelimiterLine(line) {
			return offset, len(line)
		}
		if lineEnd < 0 {
			return -1, 0
		}
		offset += lineEnd + 1
	}
}

func isDelimiterLine(line string) bool {
	t := strings.TrimSpace(line)
	for _, d := range frontmatterDelimiters {
		if t == d {
			return true
		}
	}
	return false
}

// parseFrontmatterYAML parses the frontmatter block into a plain Go mapping.
// An empty block is a valid empty mapping (spec §2.2 vector 3).
func parseFrontmatterYAML(text string) (map[string]interface{}, error) {
	if strings.TrimSpace(text) == "" {
		return map[string]interface{}{}, nil
	}

	var node interface{}
	if err := yaml.Unmarshal([]byte(text), &node); err != nil {
		return nil, &ValueError{Message: "invalid frontmatter YAML: " + err.Error(), Err: err}
	}
	if node == nil {
		return map[string]interface{}{}, nil
	}

	normalized := normalizeYAMLValue(node)
	m, ok := normalized.(map[string]interface{})
	if !ok {
		return nil, newValueError("invalid frontmatter: frontmatter must be a YAML mapping")
	}
	return m, nil
}

// normalizeYAMLValue rewrites the yaml.v3 decode result into the shape the
// emitted loaders expect: map[string]interface{} everywhere, never
// map[interface{}]interface{}.
func normalizeYAMLValue(v interface{}) interface{} {
	switch t := v.(type) {
	case map[string]interface{}:
		out := make(map[string]interface{}, len(t))
		for k, val := range t {
			out[k] = normalizeYAMLValue(val)
		}
		return out
	case map[interface{}]interface{}:
		out := make(map[string]interface{}, len(t))
		for k, val := range t {
			out[toStringKey(k)] = normalizeYAMLValue(val)
		}
		return out
	case []interface{}:
		out := make([]interface{}, len(t))
		for i, val := range t {
			out[i] = normalizeYAMLValue(val)
		}
		return out
	default:
		return v
	}
}
