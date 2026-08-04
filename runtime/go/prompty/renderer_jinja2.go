package prompty

import (
	"context"

	"github.com/nikolalohinski/gonja/v2/builtins"
	"github.com/nikolalohinski/gonja/v2/config"
	"github.com/nikolalohinski/gonja/v2/exec"
	"github.com/nikolalohinski/gonja/v2/loaders"

	model "prompty/model"
)

// templateName is the in-memory name every prompt template is loaded under.
// It must start with "/" for gonja's memory loader.
const templateName = "/prompt"

// Jinja2Renderer is the default renderer, registered under both "jinja2" and
// "nunjucks" (spec §11.3). It is backed by gonja, a Jinja2-compatible engine,
// and covers the full §5.5 conformance floor: variables, dotted access,
// if/elif/else, for, comments, and the default/upper/lower/join/length/trim
// filters.
//
// Two deliberate configuration choices:
//
//   - AutoEscape is off and StrictUndefined is off. .prompty templates are not
//     HTML, and an undefined variable renders as the empty string (spec §5.5,
//     render vectors html_not_escaped and missing_variable_renders_empty).
//   - Templates are loaded from an in-memory loader holding only this template,
//     so {% include %} and {% extends %} cannot reach the filesystem. Combined
//     with gonja's globals — none of which touch the filesystem or network —
//     this satisfies the §5.5 sandboxing requirement.
//
// The zero value is not usable; construct with NewJinja2Renderer.
type Jinja2Renderer struct {
	config *config.Config
}

// NewJinja2Renderer returns a renderer safe for concurrent use.
func NewJinja2Renderer() *Jinja2Renderer {
	cfg := config.New()
	cfg.AutoEscape = false
	cfg.StrictUndefined = false
	return &Jinja2Renderer{config: cfg}
}

// Render implements model.Renderer.
func (r *Jinja2Renderer) Render(agent model.Prompty, template string, inputs map[string]interface{}) (string, error) {
	return r.RenderContext(context.Background(), agent, template, inputs)
}

// RenderContext implements ContextRenderer.
func (r *Jinja2Renderer) RenderContext(ctx context.Context, _ model.Prompty, template string, inputs map[string]interface{}) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}

	loader, err := loaders.NewMemoryLoader(map[string]string{templateName: template})
	if err != nil {
		return "", &ValueError{Message: "Template syntax error: " + err.Error(), Err: err}
	}

	// A fresh environment per render keeps concurrent renders from sharing
	// mutable context state with each other or with gonja's package globals.
	env := &exec.Environment{
		Context:           exec.EmptyContext().Update(builtins.GlobalFunctions).Update(builtins.GlobalVariables),
		Filters:           builtins.Filters,
		Tests:             builtins.Tests,
		ControlStructures: builtins.ControlStructures,
		Methods:           builtins.Methods,
	}

	tpl, err := exec.NewTemplate(templateName, r.config, loader, env)
	if err != nil {
		return "", &ValueError{Message: "Template syntax error: " + err.Error(), Err: err}
	}

	if inputs == nil {
		inputs = map[string]interface{}{}
	}
	out, err := tpl.ExecuteToString(exec.NewContext(inputs))
	if err != nil {
		return "", &ValueError{Message: "Template render error: " + err.Error(), Err: err}
	}
	return out, nil
}

var (
	_ Renderer        = (*Jinja2Renderer)(nil)
	_ ContextRenderer = (*Jinja2Renderer)(nil)
)
