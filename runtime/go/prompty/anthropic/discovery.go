package anthropic

import (
	"fmt"

	prompty "prompty"
	model "prompty/model"
)

// Model discovery for Anthropic (spec §11.6).
//
// Anthropic's /v1/models returns capability fields directly — display name,
// context length, modalities — so a mapped ModelInfo is usually complete and
// the shared capability dataset has nothing left to fill. That is the intended
// end state: the dataset exists for providers that return less, and enrichment
// never overwrites what a provider said.

// ownerAnthropic is the owner every Anthropic model is attributed to. The API
// does not return an owner field because there is only one possible answer, but
// a cross-provider listing that leaves it blank forces callers to special-case
// Anthropic, so it is filled in here.
const ownerAnthropic = "anthropic"

// MapModel maps one entry of an Anthropic /v1/models response onto ModelInfo.
//
// The raw object is preserved in AdditionalProperties so fields the contract
// does not model — the `type` discriminator, future capability flags — stay
// reachable without a second HTTP call.
func MapModel(raw map[string]interface{}) (model.ModelInfo, error) {
	id, ok := raw["id"].(string)
	if !ok || id == "" {
		return model.ModelInfo{}, fmt.Errorf("anthropic: model entry has no id")
	}

	owner := ownerAnthropic
	info := model.ModelInfo{
		Id:                   id,
		OwnedBy:              &owner,
		AdditionalProperties: raw,
	}
	if displayName, ok := raw["display_name"].(string); ok && displayName != "" {
		info.DisplayName = &displayName
	}
	if window := asInt32(raw["context_length"]); window != nil {
		info.ContextWindow = window
	}
	if modalities := stringList(raw["input_modalities"]); modalities != nil {
		info.InputModalities = modalities
	}
	if modalities := stringList(raw["output_modalities"]); modalities != nil {
		info.OutputModalities = modalities
	}
	return info, nil
}

// MapModels maps a whole Anthropic model listing. A malformed entry fails the
// listing rather than being dropped, so a short result is never mistaken for a
// provider that offers fewer models.
func MapModels(entries []interface{}) ([]model.ModelInfo, error) {
	out := make([]model.ModelInfo, 0, len(entries))
	for index, entry := range entries {
		raw, ok := entry.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("anthropic: model listing entry %d is not an object", index)
		}
		info, err := MapModel(raw)
		if err != nil {
			return nil, fmt.Errorf("anthropic: model listing entry %d: %w", index, err)
		}
		out = append(out, info)
	}
	return out, nil
}

func asInt32(raw interface{}) *int32 {
	switch value := raw.(type) {
	case int:
		converted := int32(value)
		return &converted
	case int32:
		converted := value
		return &converted
	case int64:
		converted := int32(value)
		return &converted
	case float64:
		converted := int32(value)
		return &converted
	default:
		return nil
	}
}

// stringList converts a JSON array of strings, preserving the difference
// between an absent list (nil) and a present-but-empty one, which enrichment
// depends on.
func stringList(raw interface{}) []string {
	switch value := raw.(type) {
	case []string:
		return value
	case []interface{}:
		out := make([]string, 0, len(value))
		for _, item := range value {
			if text, ok := item.(string); ok {
				out = append(out, text)
			}
		}
		return out
	default:
		return nil
	}
}

// ModelListFunc fetches a raw Anthropic model listing. It is the seam a live
// HTTP client plugs into and a test replaces with a recorded response.
type ModelListFunc func(connection interface{}) ([]interface{}, error)

// ModelLister lists Anthropic models. It satisfies prompty.ModelLister.
type ModelLister struct{ Fetch ModelListFunc }

// ListModels fetches and maps the provider's listing.
//
// Capability enrichment is applied by prompty.ListModels rather than here, so
// every provider gets the identical fill-only-missing rule.
func (l ModelLister) ListModels(connection interface{}) ([]model.ModelInfo, error) {
	if l.Fetch == nil {
		return nil, fmt.Errorf("anthropic: model lister has no fetch function")
	}
	entries, err := l.Fetch(connection)
	if err != nil {
		return nil, err
	}
	return MapModels(entries)
}

// RegisterModelLister installs a model lister under the anthropic provider key.
func RegisterModelLister(lister ModelLister) {
	prompty.RegisterModelLister(Provider, lister)
}
