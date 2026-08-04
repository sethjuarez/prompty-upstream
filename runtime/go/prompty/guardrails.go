package prompty

import (
	"context"
	"errors"
	"fmt"

	model "prompty/model"
)

// Guardrails (spec §13.4) are optional host policy hooks around the turn loop:
// one before the model is called, one on the model's final answer, and one
// before each tool executes.
//
// They are deliberately separate from RunOptions.Permit. Permit answers "is
// this caller allowed to run this tool"; a guardrail answers "should this
// content be allowed at all". A host commonly has both, from different owners,
// and collapsing them would force one team's policy through the other's hook.

// GuardrailResult is the emitted guardrail verdict (model.GuardrailResult).
//
// Allowed reports the verdict; Reason explains a denial and is surfaced to the
// caller verbatim; Rewrite optionally replaces the checked value.
type GuardrailResult = model.GuardrailResult

// AllowGuardrail is the verdict that permits the operation unchanged.
func AllowGuardrail() GuardrailResult { return model.NewAllowGuardrailResult() }

// DenyGuardrail is the verdict that blocks the operation with a reason.
func DenyGuardrail(reason string) GuardrailResult { return model.NewDenyGuardrailResult(reason) }

// RewriteGuardrail allows the operation but substitutes a replacement value.
// Only the output guardrail acts on a rewrite; the input and tool guardrails
// ignore it, because rewriting a prompt or a tool's arguments behind the
// model's back produces a conversation that no longer replays.
func RewriteGuardrail(replacement interface{}) GuardrailResult {
	return model.NewRewriteGuardrailResult(replacement)
}

// GuardrailPhase names which hook produced a denial.
type GuardrailPhase string

const (
	GuardrailPhaseInput  GuardrailPhase = "input"
	GuardrailPhaseOutput GuardrailPhase = "output"
	GuardrailPhaseTool   GuardrailPhase = "tool"
)

// ErrGuardrailDenied marks any turn stopped by a guardrail, so a caller can
// test for the class with errors.Is without naming the phase.
var ErrGuardrailDenied = errors.New("prompty: guardrail denied")

// GuardrailError reports a guardrail denial and the phase that produced it.
//
// An input or output denial is fatal to the turn: the host said this content
// must not be sent or returned, and continuing anyway would defeat the policy.
// A tool denial is not fatal — it becomes a model-visible tool result, so the
// model can react to being refused.
type GuardrailError struct {
	Reason string
	Phase  GuardrailPhase
}

func (e *GuardrailError) Error() string {
	return fmt.Sprintf("Guardrail denied: %s", e.Reason)
}

func (e *GuardrailError) Unwrap() error { return ErrGuardrailDenied }

// InputGuardrail checks the conversation about to be sent to the model. It runs
// on every iteration, after steering injection and context trimming, so it sees
// exactly what the provider will see.
type InputGuardrail func(ctx context.Context, messages []model.Message, agent model.Prompty) GuardrailResult

// OutputGuardrail checks the model's final answer before it is returned. It
// does not run on intermediate tool-calling rounds — those are not answers.
type OutputGuardrail func(ctx context.Context, output interface{}, agent model.Prompty) GuardrailResult

// ToolGuardrail checks one tool call before it executes. Args are the effective
// arguments after binding injection, so a policy sees the values the tool will
// actually receive rather than what the model asked for.
type ToolGuardrail func(ctx context.Context, name string, args map[string]interface{}, agent model.Prompty) GuardrailResult

// Guardrails bundles the optional hooks. A nil Guardrails, or any nil field,
// allows everything — a host installs only the policies it has.
type Guardrails struct {
	Input  InputGuardrail
	Output OutputGuardrail
	Tool   ToolGuardrail
}

// CheckInput runs the input guardrail, allowing when none is configured.
func (g *Guardrails) CheckInput(ctx context.Context, messages []model.Message, agent model.Prompty) GuardrailResult {
	if g == nil || g.Input == nil {
		return AllowGuardrail()
	}
	return g.Input(ctx, messages, agent)
}

// CheckOutput runs the output guardrail, allowing when none is configured.
func (g *Guardrails) CheckOutput(ctx context.Context, output interface{}, agent model.Prompty) GuardrailResult {
	if g == nil || g.Output == nil {
		return AllowGuardrail()
	}
	return g.Output(ctx, output, agent)
}

// CheckTool runs the tool guardrail, allowing when none is configured.
func (g *Guardrails) CheckTool(ctx context.Context, name string, args map[string]interface{}, agent model.Prompty) GuardrailResult {
	if g == nil || g.Tool == nil {
		return AllowGuardrail()
	}
	return g.Tool(ctx, name, args, agent)
}

// guardrailReason returns the denial reason, falling back to a phase-specific
// default so a policy that denies without explaining still produces something
// actionable rather than an empty string.
func guardrailReason(result GuardrailResult, fallback string) string {
	if result.Reason != nil && *result.Reason != "" {
		return *result.Reason
	}
	return fallback
}

// guardrailPermission composes a tool guardrail with the host's own permission
// function into the single PermissionFunc the dispatcher understands.
//
// The guardrail runs first: a content policy that forbids a tool should not be
// overridable by a permission callback that happens to say yes. A denial from
// either becomes the same model-visible refusal, so the model always learns
// that its call was rejected and why.
func guardrailPermission(guardrails *Guardrails, agent model.Prompty, permit PermissionFunc) PermissionFunc {
	if guardrails == nil || guardrails.Tool == nil {
		return permit
	}
	return func(ctx context.Context, call model.ToolCall, args map[string]interface{}) PermissionDecision {
		verdict := guardrails.CheckTool(ctx, call.Name, args, agent)
		if !verdict.Allowed {
			return Deny(guardrailReason(verdict, "Tool denied"))
		}
		if permit == nil {
			return Allow()
		}
		return permit(ctx, call, args)
	}
}
