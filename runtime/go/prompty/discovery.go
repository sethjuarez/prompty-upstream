package prompty

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"sync"

	model "prompty/model"
)

// Cross-runtime model-capability enrichment for provider discovery (spec §11.6).
//
// Provider /models endpoints vary in richness: some (Anthropic, Foundry) return
// capability fields directly, while others (OpenAI) return only ids. To keep
// discovery results consistent across providers *and* across runtimes, Prompty
// ships one shared, provider-keyed capability dataset and applies one rule:
//
//	Provider-supplied fields always win. Dataset entries only fill fields the
//	provider left empty (fill-only-missing). Matching is by longest prefix on
//	the model id, applied only at token boundaries.
//
// Canonical source vs. vendored copy. The cross-runtime source of truth is
// spec/data/model_capabilities.json. A Go module can only embed files inside
// the module tree, so this package embeds a vendored copy at
// runtime/go/prompty/data/model_capabilities.json. The two files MUST stay
// byte-identical; TestVendoredCapabilityDatasetMatchesSpec enforces this
// whenever the repository layout is available, and is a no-op otherwise.
//
// To refresh: edit spec/data/model_capabilities.json, then copy it over the
// vendored file (the drift test fails until you do).
//
// The dataset is deliberately not emitted from TypeSpec: it is volatile
// provider data (context windows, modalities, new model families) refreshed as
// a snapshot, whereas TypeSpec owns the structural model.ModelInfo contract.

//go:embed data/model_capabilities.json
var capabilityDatasetJSON []byte

// CapabilityDatasetJSON returns the embedded shared capability dataset exactly
// as vendored, so a host can inspect or diff it without reading the repository.
func CapabilityDatasetJSON() []byte {
	out := make([]byte, len(capabilityDatasetJSON))
	copy(out, capabilityDatasetJSON)
	return out
}

// ModelCapabilities are the fallback capability fields for one model, as looked
// up from the shared dataset.
//
// Every field is optional. A nil ContextWindow means the dataset does not
// supply one. A nil modality slice means "not supplied"; a non-nil empty slice
// means "supplied and intentionally empty" — embeddings, for instance, declare
// no output modality. The distinction is load-bearing: an empty dataset list is
// a valid fill, and an empty *provider* list wins over a non-empty dataset one.
type ModelCapabilities struct {
	ContextWindow    *int32
	InputModalities  []string
	OutputModalities []string
}

type capabilityEntry struct {
	prefix       string
	capabilities ModelCapabilities
}

// capabilityTable is immutable once built, so concurrent lookups need no lock.
type capabilityTable struct {
	providers map[string][]capabilityEntry
}

var (
	capabilityTableOnce sync.Once
	capabilityTableData *capabilityTable
	capabilityTableErr  error
)

// capabilities parses the embedded dataset exactly once and hands back an
// immutable table. Parsing is deferred rather than done in an init function so
// a malformed vendored file surfaces as an error at the first discovery call
// instead of killing every program that links the package.
func capabilities() (*capabilityTable, error) {
	capabilityTableOnce.Do(func() {
		capabilityTableData, capabilityTableErr = parseCapabilityTable(capabilityDatasetJSON)
	})
	return capabilityTableData, capabilityTableErr
}

func parseCapabilityTable(raw []byte) (*capabilityTable, error) {
	var document struct {
		Providers map[string][]map[string]interface{} `json:"providers"`
	}
	if err := json.Unmarshal(raw, &document); err != nil {
		return nil, fmt.Errorf("prompty: model capability dataset is not valid JSON: %w", err)
	}

	table := &capabilityTable{providers: make(map[string][]capabilityEntry, len(document.Providers))}
	for provider, entries := range document.Providers {
		parsed := make([]capabilityEntry, 0, len(entries))
		for _, entry := range entries {
			prefix, ok := entry["prefix"].(string)
			if !ok || prefix == "" {
				continue
			}
			parsed = append(parsed, capabilityEntry{
				prefix: prefix,
				capabilities: ModelCapabilities{
					ContextWindow:    datasetInt32(entry["contextWindow"]),
					InputModalities:  datasetModalities(entry["inputModalities"]),
					OutputModalities: datasetModalities(entry["outputModalities"]),
				},
			})
		}
		// Longest prefix first so the first match is the most specific one.
		// Stable so equal-length prefixes keep the order authored in the file.
		sort.SliceStable(parsed, func(i, j int) bool {
			return len(parsed[i].prefix) > len(parsed[j].prefix)
		})
		table.providers[provider] = parsed
	}
	return table, nil
}

func datasetInt32(value interface{}) *int32 {
	number, ok := value.(float64)
	if !ok {
		return nil
	}
	converted := int32(number)
	return &converted
}

// datasetModalities preserves the difference between an absent list (nil) and a
// present-but-empty one (non-nil, length zero).
func datasetModalities(value interface{}) []string {
	list, ok := value.([]interface{})
	if !ok {
		return nil
	}
	out := make([]string, 0, len(list))
	for _, item := range list {
		if text, ok := item.(string); ok {
			out = append(out, text)
		}
	}
	return out
}

// prefixMatches reports whether id is matched by a dataset prefix under the
// cross-runtime rule.
//
// A prefix matches only at a token boundary: id must equal prefix exactly, or
// the character immediately after the prefix must be a separator (any
// non-ASCII-alphanumeric character). Real ids keep matching — gpt-4 matches
// gpt-4-0613, gpt-4o matches gpt-4o-2024-05-13 — while an accidental substring
// hit is rejected: gpt-4 must NOT match a future gpt-45. Every runtime
// implements this same rule so the shared enrichment vectors converge.
func prefixMatches(id string, prefix string) bool {
	if len(id) < len(prefix) || id[:len(prefix)] != prefix {
		return false
	}
	if len(id) == len(prefix) {
		return true
	}
	next := id[len(prefix)]
	isAlphanumeric := (next >= '0' && next <= '9') ||
		(next >= 'a' && next <= 'z') ||
		(next >= 'A' && next <= 'Z')
	return !isAlphanumeric
}

// LookupModelCapabilities returns the dataset fallback for a model id within a
// provider, and whether the provider has any entry that matches.
//
// The returned value owns its slices, so a caller may retain or mutate them
// without corrupting the shared table.
func LookupModelCapabilities(provider string, id string) (ModelCapabilities, bool) {
	table, err := capabilities()
	if err != nil || table == nil {
		return ModelCapabilities{}, false
	}
	for _, entry := range table.providers[provider] {
		if prefixMatches(id, entry.prefix) {
			return entry.capabilities.clone(), true
		}
	}
	return ModelCapabilities{}, false
}

func (c ModelCapabilities) clone() ModelCapabilities {
	return ModelCapabilities{
		ContextWindow:    copyInt32Ptr(c.ContextWindow),
		InputModalities:  copyStrings(c.InputModalities),
		OutputModalities: copyStrings(c.OutputModalities),
	}
}

func copyInt32Ptr(value *int32) *int32 {
	if value == nil {
		return nil
	}
	copied := *value
	return &copied
}

// copyStrings preserves nil-ness: a nil input stays nil ("not supplied") and an
// empty input stays a non-nil empty slice ("supplied and empty").
func copyStrings(values []string) []string {
	if values == nil {
		return nil
	}
	out := make([]string, len(values))
	copy(out, values)
	return out
}

// EnrichModelInfo fills a model.ModelInfo in place from the shared capability
// dataset, applying the cross-runtime fill-only-missing rule.
//
// A dataset field is written only where the ModelInfo left it empty: a nil
// ContextWindow, or a nil modality slice. A provider-supplied value is never
// overwritten — including a provider-supplied empty modality list, which is a
// deliberate statement and wins over a non-empty dataset entry. Enrichment of
// an unknown id, or of a provider with no dataset section, is a no-op.
//
// A nil info is ignored so callers can enrich optional results without a guard.
func EnrichModelInfo(provider string, info *model.ModelInfo) {
	if info == nil {
		return
	}
	fallback, ok := LookupModelCapabilities(provider, info.Id)
	if !ok {
		return
	}
	if info.ContextWindow == nil && fallback.ContextWindow != nil {
		info.ContextWindow = fallback.ContextWindow
	}
	if info.InputModalities == nil && fallback.InputModalities != nil {
		info.InputModalities = fallback.InputModalities
	}
	if info.OutputModalities == nil && fallback.OutputModalities != nil {
		info.OutputModalities = fallback.OutputModalities
	}
}

// EnrichModelInfos enriches every entry of a discovery result in place and
// returns the same slice, so a lister can end with a single call.
func EnrichModelInfos(provider string, infos []model.ModelInfo) []model.ModelInfo {
	for index := range infos {
		EnrichModelInfo(provider, &infos[index])
	}
	return infos
}

// ModelLister is the emitted model-discovery contract (model.ModelLister).
//
// Listing is kept behind this interface on purpose: a lister is the only part
// of discovery that talks to a network, so a host can register a fake in tests
// and the enrichment rules above stay exercised without a live provider.
type ModelLister = model.ModelLister

// ContextModelLister is the optional cancellation-aware extension of
// ModelLister. Discovery prefers it when a registered lister implements it.
type ContextModelLister interface {
	ListModelsContext(ctx context.Context, connection interface{}) ([]model.ModelInfo, error)
}

var modelListers = newRegistry[ModelLister]("model lister", nil)

// RegisterModelLister registers (or replaces) a model lister under a provider
// key. Registration is explicit, exactly like executors and processors: a host
// that never discovers models never links a discovery client.
func RegisterModelLister(key string, lister ModelLister) { modelListers.register(key, lister) }

// UnregisterModelLister removes the model lister registered under key.
func UnregisterModelLister(key string) { modelListers.unregister(key) }

// GetModelLister returns the model lister registered under key, or an
// *InvokerError naming the missing provider.
func GetModelLister(key string) (ModelLister, error) { return modelListers.get(key) }

// HasModelLister reports whether a model lister is registered under key.
func HasModelLister(key string) bool { return modelListers.has(key) }

// ModelListerKeys lists the registered model lister keys in sorted order.
func ModelListerKeys() []string { return modelListers.keys() }

// ListModels discovers the models a provider connection exposes and enriches
// them from the shared capability dataset before returning.
//
// Enrichment is applied here rather than inside each lister so every provider
// gets the identical fill-only-missing rule; a lister only has to map its own
// wire shape onto model.ModelInfo.
func ListModels(provider string, connection interface{}) ([]model.ModelInfo, error) {
	lister, err := GetModelLister(provider)
	if err != nil {
		return nil, err
	}
	infos, err := lister.ListModels(connection)
	if err != nil {
		return nil, err
	}
	return EnrichModelInfos(provider, infos), nil
}

// ListModelsContext is ListModels with cancellation, used when the registered
// lister implements ContextModelLister. A lister that does not is called
// without the context, so a plain implementation still works.
func ListModelsContext(ctx context.Context, provider string, connection interface{}) ([]model.ModelInfo, error) {
	lister, err := GetModelLister(provider)
	if err != nil {
		return nil, err
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	var infos []model.ModelInfo
	if aware, ok := lister.(ContextModelLister); ok {
		infos, err = aware.ListModelsContext(ctx, connection)
	} else {
		infos, err = lister.ListModels(connection)
	}
	if err != nil {
		return nil, err
	}
	return EnrichModelInfos(provider, infos), nil
}
