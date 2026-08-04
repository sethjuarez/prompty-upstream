package prompty

import (
	"context"

	"github.com/cbroglie/mustache"

	model "prompty/model"
)

// MustacheRenderer implements the optional "mustache" template format
// (spec §11.3). It supports variables, truthy sections, inverted sections and
// list iteration with the implicit `{{.}}` operator.
//
// Two deliberate configuration choices, mirroring Jinja2Renderer:
//
//   - Raw mode is forced, so `{{value}}` is never HTML-escaped. .prompty
//     templates are not HTML documents.
//   - Partials resolve through an empty static provider, so `{{> partial}}`
//     cannot read files off disk. The library default is a filesystem provider.
//
// The zero value is not usable; construct with NewMustacheRenderer.
type MustacheRenderer struct {
	partials mustache.PartialProvider
}

// NewMustacheRenderer returns a renderer safe for concurrent use.
func NewMustacheRenderer() *MustacheRenderer {
	return &MustacheRenderer{partials: &mustache.StaticProvider{}}
}

// Render implements model.Renderer.
func (r *MustacheRenderer) Render(agent model.Prompty, template string, inputs map[string]interface{}) (string, error) {
	return r.RenderContext(context.Background(), agent, template, inputs)
}

// RenderContext implements ContextRenderer.
func (r *MustacheRenderer) RenderContext(ctx context.Context, _ model.Prompty, template string, inputs map[string]interface{}) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}

	tpl, err := mustache.ParseStringPartialsRaw(template, r.partials, true)
	if err != nil {
		return "", &ValueError{Message: "Template syntax error: " + err.Error(), Err: err}
	}
	if inputs == nil {
		inputs = map[string]interface{}{}
	}
	out, err := tpl.Render(inputs)
	if err != nil {
		return "", &ValueError{Message: "Template render error: " + err.Error(), Err: err}
	}
	return out, nil
}

var (
	_ Renderer        = (*MustacheRenderer)(nil)
	_ ContextRenderer = (*MustacheRenderer)(nil)
)
