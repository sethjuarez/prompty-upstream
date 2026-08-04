package prompty

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"regexp"
	"strconv"
	"strings"

	model "prompty/model"
)

// roleBoundaryRe is the normative role boundary regex from spec §6.2. A role
// marker must occupy its whole line (leading whitespace and a leading `#` for
// markdown-heading compatibility are allowed) and role names are
// case-insensitive.
//
// Only system, user and assistant are role markers. `developer:` and `tool:`
// exist as Message roles but are not recognised in template bodies, so a line
// like "tool: use this carefully" stays ordinary prose.
var roleBoundaryRe = regexp.MustCompile(`(?i)^\s*#?\s*(system|user|assistant)(\[(\w+\s*=\s*"?[^"]*"?\s*,?\s*)+\])?\s*:\s*$`)

// roleAttrRe extracts key=value pairs from a role marker attribute block. The
// quoted alternative comes first so a quoted value may contain commas and
// brackets, which the bare form must stop at.
var roleAttrRe = regexp.MustCompile(`(\w+)\s*=\s*(?:"([^"]*)"|([^",\]]*))`)

// parserNonceKey is the context key under which the pre-render nonce travels
// from PreRender to Parse.
const parserNonceKey = "nonce"

// PromptyParser is the built-in role-marker parser registered under "prompty"
// (spec §6). It converts rendered text into an ordered []model.Message.
//
// It also implements the §6.3 injection defense. PreRender rewrites every role
// marker in the template to carry a fresh nonce; Parse then rejects any role
// marker whose nonce does not match, which is exactly the set of markers that
// arrived through interpolated input rather than the template itself. That check
// only runs when the caller opts in (Template.Format.Strict), because the shared
// render vectors require role markers to pass through untouched by default.
type PromptyParser struct{}

// NewPromptyParser returns a parser safe for concurrent use.
func NewPromptyParser() *PromptyParser { return &PromptyParser{} }

// PreRender implements model.Parser. It returns a *interface{} holding a
// map[string]interface{} with the sanitized template and the nonce, matching the
// emitted signature.
func (p *PromptyParser) PreRender(template string) *interface{} {
	sanitized, nonce := p.preRender(template)
	var out interface{} = map[string]interface{}{
		"template":     sanitized,
		parserNonceKey: nonce,
	}
	return &out
}

// preRender is the typed form of PreRender used inside this package.
func (p *PromptyParser) preRender(template string) (string, string) {
	nonce := newNonce(8)
	lines := strings.Split(template, "\n")
	for i, line := range lines {
		match := roleBoundaryRe.FindStringSubmatch(line)
		if match == nil {
			continue
		}
		role := strings.ToLower(match[1])
		attrs := parseRoleAttrs(match[2])
		delete(attrs, parserNonceKey)
		lines[i] = renderRoleMarker(role, nonce, attrs)
	}
	return strings.Join(lines, "\n"), nonce
}

// Parse implements model.Parser.
func (p *PromptyParser) Parse(agent model.Prompty, rendered string, parserContext *map[string]interface{}) ([]model.Message, error) {
	return p.ParseContext(context.Background(), agent, rendered, parserContext)
}

// ParseContext implements ContextParser.
func (p *PromptyParser) ParseContext(ctx context.Context, _ model.Prompty, rendered string, parserContext *map[string]interface{}) ([]model.Message, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	expected := ""
	if parserContext != nil {
		if v, ok := (*parserContext)[parserNonceKey].(string); ok {
			expected = v
		}
	}
	return parseChat(rendered, expected)
}

// ParseChat splits rendered text at role markers with no nonce validation.
func ParseChat(rendered string) ([]model.Message, error) {
	return parseChat(rendered, "")
}

// parseChat is the §6.4 parsing algorithm.
//
// Content accumulates verbatim between markers; only leading and trailing
// newlines are trimmed, so indentation and trailing spaces inside a message
// survive (spec §6.6, parse vector content_trimmed). A message with no content
// between two markers becomes an empty TextPart rather than being dropped.
func parseChat(rendered string, expectedNonce string) ([]model.Message, error) {
	var (
		messages    []model.Message
		currentRole = model.RoleSystem
		currentAttr map[string]interface{}
		buffer      []string
		sawMarker   bool
	)

	flush := func() error {
		// Before the first marker there is nothing to flush unless the body had
		// leading content, which becomes a system message (spec §6.4 step 4).
		if !sawMarker && len(buffer) == 0 {
			return nil
		}
		msg, err := buildMessage(currentRole, joinAndTrim(buffer), currentAttr, sawMarker, expectedNonce)
		if err != nil {
			return err
		}
		messages = append(messages, msg)
		buffer = nil
		currentAttr = nil
		return nil
	}

	for _, line := range strings.Split(rendered, "\n") {
		match := roleBoundaryRe.FindStringSubmatch(line)
		if match == nil {
			buffer = append(buffer, line)
			continue
		}
		if err := flush(); err != nil {
			return nil, err
		}
		currentRole = model.Role(strings.ToLower(match[1]))
		currentAttr = parseRoleAttrs(match[2])
		sawMarker = true
	}

	if err := flush(); err != nil {
		return nil, err
	}
	return messages, nil
}

func buildMessage(role model.Role, content string, attrs map[string]interface{}, sawMarker bool, expectedNonce string) (model.Message, error) {
	if expectedNonce != "" && sawMarker {
		actual := ""
		if v, ok := attrs[parserNonceKey]; ok {
			actual = stringifyAttr(v)
		}
		if actual != expectedNonce {
			return model.Message{}, newValueError("Role marker nonce mismatch (possible injection)")
		}
	}

	// The nonce is an internal transport detail and never reaches metadata.
	var metadata map[string]interface{}
	for k, v := range attrs {
		if k == parserNonceKey {
			continue
		}
		if metadata == nil {
			metadata = map[string]interface{}{}
		}
		metadata[k] = v
	}

	return model.Message{
		Role:     role,
		Parts:    []interface{}{model.TextPart{Kind: "text", Value: content}},
		Metadata: metadata,
	}, nil
}

// joinAndTrim joins the buffered lines and strips leading/trailing newlines
// only. Spaces and tabs at the edges are content.
func joinAndTrim(lines []string) string {
	return strings.Trim(strings.Join(lines, "\n"), "\n")
}

// parseRoleAttrs extracts the `[key=value, key2="value2"]` block from a role
// marker, coercing booleans and numbers the way the other runtimes do.
func parseRoleAttrs(block string) map[string]interface{} {
	if block == "" {
		return nil
	}
	var attrs map[string]interface{}
	for _, match := range roleAttrRe.FindAllStringSubmatch(block, -1) {
		key := match[1]
		// Group 2 is the quoted form, group 3 the bare form; exactly one is
		// non-empty for a non-empty value, and both are empty for `key=""`.
		raw := match[2]
		if raw == "" {
			raw = match[3]
		}
		raw = strings.TrimSpace(raw)
		if attrs == nil {
			attrs = map[string]interface{}{}
		}
		attrs[key] = coerceAttrValue(raw)
	}
	return attrs
}

func coerceAttrValue(raw string) interface{} {
	switch strings.ToLower(raw) {
	case "true":
		return true
	case "false":
		return false
	}
	if i, err := strconv.ParseInt(raw, 10, 64); err == nil {
		return i
	}
	if f, err := strconv.ParseFloat(raw, 64); err == nil {
		return f
	}
	return raw
}

func stringifyAttr(v interface{}) string {
	switch t := v.(type) {
	case string:
		return t
	case int64:
		return strconv.FormatInt(t, 10)
	case float64:
		return strconv.FormatFloat(t, 'f', -1, 64)
	case bool:
		return strconv.FormatBool(t)
	default:
		return ""
	}
}

func renderRoleMarker(role, nonce string, attrs map[string]interface{}) string {
	var b strings.Builder
	b.WriteString(role)
	b.WriteString(`[nonce="`)
	b.WriteString(nonce)
	b.WriteString(`"`)
	for _, key := range sortedAttrKeys(attrs) {
		b.WriteString(", ")
		b.WriteString(key)
		b.WriteString(`="`)
		b.WriteString(stringifyAttr(attrs[key]))
		b.WriteString(`"`)
	}
	b.WriteString("]:")
	return b.String()
}

// newNonce returns n cryptographically random hex characters. Falling back to a
// non-random value would silently disable both the injection defense and thread
// nonce uniqueness, so a failure of the system CSPRNG panics instead.
func newNonce(hexChars int) string {
	buf := make([]byte, (hexChars+1)/2)
	if _, err := rand.Read(buf); err != nil {
		panic("prompty: crypto/rand unavailable: " + err.Error())
	}
	return hex.EncodeToString(buf)[:hexChars]
}

var (
	_ Parser        = (*PromptyParser)(nil)
	_ ContextParser = (*PromptyParser)(nil)
)
