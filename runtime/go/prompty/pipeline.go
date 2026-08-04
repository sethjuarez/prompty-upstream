package prompty

import (
	"context"
	"regexp"
	"strings"
	"sync"

	model "prompty/model"
)

// RichKinds are the input property kinds whose values are structured data that
// must never be handed to a template engine directly (spec §5.2, §12.6). During
// rendering each one is replaced by a nonce; `thread` nonces are expanded back
// into messages during parsing, the rest are resolved during wire conversion.
var RichKinds = map[string]bool{
	"thread": true,
	"image":  true,
	"file":   true,
	"audio":  true,
}

// noncePrefix and nonceHexLen define the nonce format from spec §12.6:
//
//	__PROMPTY_THREAD_<hex8>_<propertyName>__
const (
	noncePrefix  = "__PROMPTY_THREAD_"
	nonceHexLen  = 8
	nonceSuffix  = "__"
	nonceJoinSep = "_"
)

// threadNonceRe matches an emitted nonce anywhere inside message text.
var threadNonceRe = regexp.MustCompile(`__PROMPTY_THREAD_[0-9a-fA-F]+_[A-Za-z_][A-Za-z0-9_]*__`)

// RenderState carries the per-render nonce table and parser context from the
// render stage to the parse stage (spec §5.4 step 5).
//
// It is safe for concurrent use: a host may render on one goroutine and parse on
// another, and a single state may be read by several goroutines at once.
type RenderState struct {
	mu            sync.RWMutex
	nonces        map[string]interface{}
	parserContext map[string]interface{}
}

func newRenderState() *RenderState {
	return &RenderState{nonces: map[string]interface{}{}}
}

func (s *RenderState) put(nonce string, value interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nonces[nonce] = value
}

// Lookup returns the original value behind a nonce.
func (s *RenderState) Lookup(nonce string) (interface{}, bool) {
	if s == nil {
		return nil, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.nonces[nonce]
	return v, ok
}

// Nonces returns a copy of the nonce table.
func (s *RenderState) Nonces() map[string]interface{} {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string]interface{}, len(s.nonces))
	for k, v := range s.nonces {
		out[k] = v
	}
	return out
}

// ParserContext returns the context the parser needs, or nil when the render did
// not produce one (non-strict mode).
func (s *RenderState) ParserContext() *map[string]interface{} {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.parserContext == nil {
		return nil
	}
	out := make(map[string]interface{}, len(s.parserContext))
	for k, v := range s.parserContext {
		out[k] = v
	}
	return &out
}

// ValidateInputs applies the agent's declared input schema to user inputs
// (spec §12.2).
//
// Rules, all of them load-bearing:
//   - A declared input with no supplied value falls back to its `default`.
//   - A required input with neither value nor default is an error.
//   - An optional input with neither is omitted, not defaulted and not an error.
//   - `example` is documentation only and is never used as a runtime value.
//   - Values are not type-checked and undeclared inputs are passed through.
func ValidateInputs(agent model.Prompty, inputs map[string]interface{}) (map[string]interface{}, error) {
	validated := make(map[string]interface{}, len(inputs)+len(agent.Inputs))
	for k, v := range inputs {
		validated[k] = v
	}

	for _, prop := range AgentInputs(agent) {
		if prop.Name == "" {
			continue
		}
		if _, ok := validated[prop.Name]; ok {
			continue
		}
		if prop.Default != nil {
			validated[prop.Name] = *prop.Default
			continue
		}
		if prop.Required {
			return nil, &ValueError{
				Message:    "Missing required input: " + prop.Name,
				Field:      prop.Name,
				Constraint: "required",
			}
		}
	}
	return validated, nil
}

// Render substitutes inputs into the agent's instructions (spec §5).
func Render(agent model.Prompty, inputs map[string]interface{}) (string, error) {
	rendered, _, err := RenderWithState(context.Background(), agent, inputs)
	return rendered, err
}

// RenderWithContext is Render with cancellation.
func RenderWithContext(ctx context.Context, agent model.Prompty, inputs map[string]interface{}) (string, error) {
	rendered, _, err := RenderWithState(ctx, agent, inputs)
	return rendered, err
}

// RenderWithState renders and also returns the state the parser needs to expand
// thread nonces and validate role markers. Callers that go on to parse should
// use this and pass the state to ParseWithState; Prepare does exactly that.
func RenderWithState(ctx context.Context, agent model.Prompty, inputs map[string]interface{}) (string, *RenderState, error) {
	if err := ctx.Err(); err != nil {
		return "", nil, err
	}

	validated, err := ValidateInputs(agent, inputs)
	if err != nil {
		return "", nil, err
	}

	state := newRenderState()
	prepared := make(map[string]interface{}, len(validated))
	for k, v := range validated {
		prepared[k] = v
	}

	// Replace rich-kind values with nonces so structured data never reaches the
	// template engine (spec §5.4 step 2).
	for _, prop := range AgentInputs(agent) {
		if prop.Name == "" || !RichKinds[prop.Kind] {
			continue
		}
		value, ok := validated[prop.Name]
		if !ok {
			continue
		}
		nonce := noncePrefix + newNonce(nonceHexLen) + nonceJoinSep + prop.Name + nonceSuffix
		prepared[prop.Name] = nonce
		state.put(nonce, value)
	}

	template := instructionsOf(agent)

	// Strict mode stamps every role marker in the template with a nonce so the
	// parser can tell template-authored markers from interpolated ones.
	if isStrict(agent) {
		parser, err := parserFor(agent)
		if err != nil {
			return "", nil, err
		}
		if sanitized, nonce, ok := preRenderTemplate(parser, template); ok {
			template = sanitized
			state.parserContext = map[string]interface{}{parserNonceKey: nonce}
		}
	}

	renderer, err := rendererFor(agent)
	if err != nil {
		return "", nil, err
	}

	rendered, err := renderWith(ctx, renderer, agent, template, prepared)
	if err != nil {
		return "", nil, err
	}
	return rendered, state, nil
}

// Parse splits rendered text into messages (spec §6). Thread nonces are left as
// literal text because without render state there is nothing to expand them to;
// use Prepare or ParseWithState for the full pipeline.
func Parse(agent model.Prompty, rendered string) ([]model.Message, error) {
	return ParseWithState(context.Background(), agent, rendered, nil)
}

// ParseWithState parses rendered text and expands thread nonces recorded in
// state.
func ParseWithState(ctx context.Context, agent model.Prompty, rendered string, state *RenderState) ([]model.Message, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	parser, err := parserFor(agent)
	if err != nil {
		return nil, err
	}

	messages, err := parseWith(ctx, parser, agent, rendered, state.ParserContext())
	if err != nil {
		return nil, err
	}
	return expandThreads(messages, state), nil
}

// Prepare runs render + parse + thread expansion and returns messages ready for
// a provider (spec §1.4).
func Prepare(agent model.Prompty, inputs map[string]interface{}) ([]model.Message, error) {
	return PrepareWithContext(context.Background(), agent, inputs)
}

// PrepareWithContext is Prepare with cancellation. ctx is checked at each stage
// boundary and forwarded to renderers and parsers that accept one.
func PrepareWithContext(ctx context.Context, agent model.Prompty, inputs map[string]interface{}) ([]model.Message, error) {
	rendered, state, err := RenderWithState(ctx, agent, inputs)
	if err != nil {
		return nil, err
	}
	return ParseWithState(ctx, agent, rendered, state)
}

// renderWith prefers the cancellation-aware interface when the renderer offers
// one, and otherwise falls back to the emitted contract.
func renderWith(ctx context.Context, renderer Renderer, agent model.Prompty, template string, inputs map[string]interface{}) (string, error) {
	if cr, ok := renderer.(ContextRenderer); ok {
		return cr.RenderContext(ctx, agent, template, inputs)
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	return renderer.Render(agent, template, inputs)
}

func parseWith(ctx context.Context, parser Parser, agent model.Prompty, rendered string, parserContext *map[string]interface{}) ([]model.Message, error) {
	if cp, ok := parser.(ContextParser); ok {
		return cp.ParseContext(ctx, agent, rendered, parserContext)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return parser.Parse(agent, rendered, parserContext)
}

// preRenderTemplate calls the emitted PreRender hook and unpacks its untyped
// result. A parser that does not implement pre-rendering returns nil, and the
// template is used unchanged.
func preRenderTemplate(parser Parser, template string) (string, string, bool) {
	result := parser.PreRender(template)
	if result == nil || *result == nil {
		return "", "", false
	}
	m, ok := (*result).(map[string]interface{})
	if !ok {
		return "", "", false
	}
	sanitized, ok := m["template"].(string)
	if !ok {
		return "", "", false
	}
	nonce, _ := m[parserNonceKey].(string)
	return sanitized, nonce, true
}

func instructionsOf(agent model.Prompty) string {
	if agent.Instructions == nil {
		return ""
	}
	return *agent.Instructions
}

func isStrict(agent model.Prompty) bool {
	return agent.Template != nil && agent.Template.Format.Strict != nil && *agent.Template.Format.Strict
}

// expandThreads replaces thread nonces with the conversation history they stand
// for (spec §6.7).
//
// Text before the nonce becomes a message with the containing message's role,
// the thread's own messages are spliced in with their original roles, and text
// after the nonce becomes another message with the containing role. A nonce with
// no thread behind it stays literal text.
func expandThreads(messages []model.Message, state *RenderState) []model.Message {
	if state == nil || len(messages) == 0 {
		return messages
	}
	if len(state.Nonces()) == 0 {
		return messages
	}

	out := make([]model.Message, 0, len(messages))
	for _, msg := range messages {
		out = append(out, expandMessage(msg, state)...)
	}
	return out
}

func expandMessage(msg model.Message, state *RenderState) []model.Message {
	text, ok := soleTextValue(msg)
	if !ok {
		return []model.Message{msg}
	}

	var (
		out       []model.Message
		pending   strings.Builder
		remaining = text
		expanded  bool
	)

	flushPending := func() {
		if trimmed := strings.Trim(pending.String(), "\n"); trimmed != "" {
			out = append(out, textMessage(msg.Role, trimmed, msg.Metadata))
		}
		pending.Reset()
	}

	for {
		loc := threadNonceRe.FindStringIndex(remaining)
		if loc == nil {
			pending.WriteString(remaining)
			break
		}

		nonce := remaining[loc[0]:loc[1]]
		value, found := state.Lookup(nonce)
		var thread []model.Message
		isThread := false
		if found {
			thread, isThread = coerceThreadMessages(value)
		}

		if !isThread {
			// Either a nonce we never issued (so it came from user input and
			// must stay inert literal text) or a non-thread rich kind whose
			// nonce wire conversion resolves later (spec §12.6). Keep it and
			// keep scanning: a real thread nonce may follow it.
			pending.WriteString(remaining[:loc[1]])
			remaining = remaining[loc[1]:]
			continue
		}

		expanded = true
		pending.WriteString(remaining[:loc[0]])
		flushPending()
		out = append(out, thread...)
		remaining = remaining[loc[1]:]
	}

	if !expanded {
		return []model.Message{msg}
	}
	flushPending()
	return out
}

// soleTextValue returns the message text when the message is a single TextPart.
// Multimodal messages are never split around a nonce.
func soleTextValue(msg model.Message) (string, bool) {
	if len(msg.Parts) != 1 {
		return "", false
	}
	switch p := msg.Parts[0].(type) {
	case model.TextPart:
		return p.Value, true
	case *model.TextPart:
		return p.Value, true
	default:
		return "", false
	}
}

// textMessage builds a single-TextPart message. Metadata is copied rather than
// aliased: one source message can expand into several, and they must not share
// a mutable map.
func textMessage(role model.Role, text string, metadata map[string]interface{}) model.Message {
	var copied map[string]interface{}
	if metadata != nil {
		copied = make(map[string]interface{}, len(metadata))
		for k, v := range metadata {
			copied[k] = v
		}
	}
	return model.Message{
		Role:     role,
		Parts:    []interface{}{model.TextPart{Kind: "text", Value: text}},
		Metadata: copied,
	}
}

// coerceThreadMessages accepts every shape a thread input can plausibly arrive
// in: already-typed messages, decoded JSON/YAML maps, or the
// `{_kind: thread, messages: [...]}` envelope used by the shared vectors.
func coerceThreadMessages(v interface{}) ([]model.Message, bool) {
	switch t := v.(type) {
	case nil:
		return nil, false
	case []model.Message:
		return t, true
	case []interface{}:
		out := make([]model.Message, 0, len(t))
		for _, item := range t {
			msg, ok := coerceMessage(item)
			if !ok {
				return nil, false
			}
			out = append(out, msg)
		}
		return out, true
	case map[string]interface{}:
		if inner, ok := t["messages"]; ok {
			return coerceThreadMessages(inner)
		}
		return nil, false
	default:
		return nil, false
	}
}

func coerceMessage(v interface{}) (model.Message, bool) {
	switch t := v.(type) {
	case model.Message:
		return t, true
	case *model.Message:
		return *t, true
	case map[string]interface{}:
		role, _ := t["role"].(string)
		if role == "" {
			return model.Message{}, false
		}
		parts, ok := coerceParts(firstPresent(t, "parts", "content"))
		if !ok {
			return model.Message{}, false
		}
		metadata, _ := t["metadata"].(map[string]interface{})
		return model.Message{Role: model.Role(role), Parts: parts, Metadata: metadata}, true
	default:
		return model.Message{}, false
	}
}

func coerceParts(v interface{}) ([]interface{}, bool) {
	switch t := v.(type) {
	case nil:
		return []interface{}{model.TextPart{Kind: "text", Value: ""}}, true
	case string:
		return []interface{}{model.TextPart{Kind: "text", Value: t}}, true
	case []interface{}:
		out := make([]interface{}, 0, len(t))
		for _, item := range t {
			switch p := item.(type) {
			case string:
				out = append(out, model.TextPart{Kind: "text", Value: p})
			case map[string]interface{}:
				// Delegate discrimination to the emitted loader so text, image,
				// audio and file parts all land on the canonical types.
				loaded, err := model.LoadContentPart(p, model.NewLoadContext())
				if err != nil {
					return nil, false
				}
				out = append(out, loaded)
			default:
				out = append(out, item)
			}
		}
		return out, true
	default:
		return nil, false
	}
}

func firstPresent(m map[string]interface{}, keys ...string) interface{} {
	for _, key := range keys {
		if v, ok := m[key]; ok {
			return v
		}
	}
	return nil
}
