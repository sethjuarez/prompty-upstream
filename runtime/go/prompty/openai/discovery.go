package openai

import (
	"fmt"

	prompty "prompty"
	model "prompty/model"
)

// Model discovery for OpenAI, Azure OpenAI and Azure AI Foundry (spec §11.6).
//
// The three endpoints return three different shapes, and this file is the whole
// of the difference: MapModel for the OpenAI /v1/models list, MapDeployment for
// a Foundry/Azure deployment, and MapCatalogModel for a Foundry catalog entry.
// Everything after the mapping — capability enrichment from the shared dataset —
// is the runtime's, so all providers converge on one rule.
//
// Every mapper preserves the provider's raw object in
// model.ModelInfo.AdditionalProperties. A discovery result that drops the
// fields the contract does not model would force a host back to the raw HTTP
// call for anything Prompty had not anticipated.

// MapModel maps one entry of an OpenAI /v1/models response onto ModelInfo.
//
// OpenAI returns ids and ownership and nothing else — no context window, no
// modalities — which is exactly why the shared capability dataset exists.
func MapModel(raw map[string]interface{}) (model.ModelInfo, error) {
	id, ok := raw["id"].(string)
	if !ok || id == "" {
		return model.ModelInfo{}, fmt.Errorf("openai: model entry has no id")
	}
	info := model.ModelInfo{Id: id, AdditionalProperties: raw}
	if owner, ok := raw["owned_by"].(string); ok && owner != "" {
		info.OwnedBy = &owner
	}
	return info, nil
}

// MapDeployment maps an Azure/Foundry model deployment onto ModelInfo.
//
// Two shapes are accepted because both are live: the flat data-plane shape
// ({name, modelName, modelPublisher, ...}) and the nested ARM control-plane
// shape ({name, properties: {model: {...}, capabilities: {...}}}). The
// deployment name is the id in both, because that is the name an agent puts in
// its model.id — not the underlying model's name, which several deployments
// may share.
func MapDeployment(raw map[string]interface{}) (model.ModelInfo, error) {
	name, ok := raw["name"].(string)
	if !ok || name == "" {
		return model.ModelInfo{}, fmt.Errorf("openai: deployment has no name")
	}
	info := model.ModelInfo{Id: name, AdditionalProperties: raw}

	properties, _ := raw["properties"].(map[string]interface{})
	nestedModel, _ := properties["model"].(map[string]interface{})

	if displayName := firstString(raw["modelName"], nestedModel["name"]); displayName != "" {
		info.DisplayName = &displayName
	}
	if owner := firstString(raw["modelPublisher"], nestedModel["publisher"]); owner != "" {
		info.OwnedBy = &owner
	}
	if window := firstInt32(raw["maxContextLength"], nestedModel["maxContextLength"]); window != nil {
		info.ContextWindow = window
	}

	capabilities, _ := properties["capabilities"].(map[string]interface{})
	if modalities := stringList(capabilities["supportedInputModalities"]); modalities != nil {
		info.InputModalities = modalities
	}
	if modalities := stringList(capabilities["supportedOutputModalities"]); modalities != nil {
		info.OutputModalities = modalities
	}
	return info, nil
}

// MapCatalogModel maps a Foundry catalog model onto ModelInfo. The catalog is
// the OpenAI-compatible listing shape with an Azure context-length field
// bolted on.
func MapCatalogModel(raw map[string]interface{}) (model.ModelInfo, error) {
	info, err := MapModel(raw)
	if err != nil {
		return model.ModelInfo{}, err
	}
	if window := firstInt32(raw["maxContextLength"], raw["context_length"]); window != nil {
		info.ContextWindow = window
	}
	return info, nil
}

// MapModels maps a whole OpenAI /v1/models list. A malformed entry fails the
// listing rather than being dropped: a discovery result that is silently short
// looks to a caller exactly like a provider that does not offer the model.
func MapModels(entries []interface{}) ([]model.ModelInfo, error) {
	return mapEach(entries, MapModel)
}

// MapDeployments maps a whole deployment listing.
func MapDeployments(entries []interface{}) ([]model.ModelInfo, error) {
	return mapEach(entries, MapDeployment)
}

// MapCatalogModels maps a whole Foundry catalog listing.
func MapCatalogModels(entries []interface{}) ([]model.ModelInfo, error) {
	return mapEach(entries, MapCatalogModel)
}

func mapEach(
	entries []interface{},
	mapper func(map[string]interface{}) (model.ModelInfo, error),
) ([]model.ModelInfo, error) {
	out := make([]model.ModelInfo, 0, len(entries))
	for index, entry := range entries {
		raw, ok := entry.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("openai: model listing entry %d is not an object", index)
		}
		info, err := mapper(raw)
		if err != nil {
			return nil, fmt.Errorf("openai: model listing entry %d: %w", index, err)
		}
		out = append(out, info)
	}
	return out, nil
}

func firstString(candidates ...interface{}) string {
	for _, candidate := range candidates {
		if value, ok := candidate.(string); ok && value != "" {
			return value
		}
	}
	return ""
}

func firstInt32(candidates ...interface{}) *int32 {
	for _, candidate := range candidates {
		switch value := candidate.(type) {
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
		}
	}
	return nil
}

// stringList converts a JSON array of strings, preserving the difference
// between an absent list (nil) and a present-but-empty one. Enrichment relies
// on that difference: an empty provider list wins over the dataset.
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

// ModelListFunc fetches a raw provider model listing. It is the seam a live
// HTTP client plugs into and a test replaces with a recorded response, so
// discovery is testable without a network.
type ModelListFunc func(connection interface{}) ([]interface{}, error)

// ModelLister lists models for an OpenAI-shaped provider.
//
// Fetch does the transport; Map turns one raw entry into a ModelInfo. Leaving
// Map nil selects MapModel, which is the OpenAI shape; a Foundry lister sets it
// to MapDeployment or MapCatalogModel.
type ModelLister struct {
	Fetch ModelListFunc
	Map   func(map[string]interface{}) (model.ModelInfo, error)
}

// ListModels satisfies prompty.ModelLister (model.ModelLister).
//
// Capability enrichment is not applied here: prompty.ListModels applies it for
// every provider, so a lister that is called directly returns exactly what the
// provider said, and a lister resolved through the registry gets the shared
// fill-only-missing rule.
func (l ModelLister) ListModels(connection interface{}) ([]model.ModelInfo, error) {
	if l.Fetch == nil {
		return nil, fmt.Errorf("openai: model lister has no fetch function")
	}
	entries, err := l.Fetch(connection)
	if err != nil {
		return nil, err
	}
	mapper := l.Map
	if mapper == nil {
		mapper = MapModel
	}
	return mapEach(entries, mapper)
}

// RegisterModelLister installs a model lister under the OpenAI provider key.
// Registration is explicit, like every other provider component: a host that
// never discovers models never links a discovery client.
func RegisterModelLister(lister ModelLister) {
	prompty.RegisterModelLister(ProviderOpenAI, lister)
}

// RegisterFoundryModelLister installs a deployment lister under both the
// foundry and azure provider keys.
func RegisterFoundryModelLister(lister ModelLister) {
	if lister.Map == nil {
		lister.Map = MapDeployment
	}
	prompty.RegisterModelLister(ProviderFoundry, lister)
	prompty.RegisterModelLister(ProviderAzure, lister)
}
