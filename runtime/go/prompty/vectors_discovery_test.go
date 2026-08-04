package prompty_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	prompty "prompty"
	anthropic "prompty/anthropic"
	model "prompty/model"
	openai "prompty/openai"
)

// discoveryVectorFile is the shared provider-wire-to-ModelInfo vector file.
type discoveryVectorFile struct {
	Description string            `json:"description"`
	Vectors     []discoveryVector `json:"vectors"`
}

type discoveryVector struct {
	Name     string                 `json:"name"`
	Provider string                 `json:"provider"`
	Shape    string                 `json:"shape"`
	Input    map[string]interface{} `json:"input"`
	Expected map[string]interface{} `json:"expected"`
}

// TestDiscoveryVectors drives every shared discovery vector through the real
// provider mappers, asserting the canonical emitted ModelInfo projection.
//
// The comparison is on ModelInfo.Save rather than on Go struct fields, because
// the contract the vectors pin is the wire shape every runtime emits, not any
// one runtime's internal representation.
func TestDiscoveryVectors(t *testing.T) {
	var file discoveryVectorFile
	readVectorFile(t, "discovery/discovery_vectors.json", &file)
	if len(file.Vectors) == 0 {
		t.Fatal("no discovery vectors found")
	}

	executed := 0
	for _, vector := range file.Vectors {
		vector := vector
		executed++
		t.Run(vector.Name, func(t *testing.T) {
			info, err := mapDiscoveryVector(vector)
			if err != nil {
				t.Fatalf("map %s/%s: %v", vector.Provider, vector.Shape, err)
			}
			assertModelInfoMatches(t, info, vector.Expected)
		})
	}
	t.Logf("discovery vectors: %d executed, %d total", executed, len(file.Vectors))
}

// mapDiscoveryVector routes a vector to the provider mapper its provider and
// shape name.
func mapDiscoveryVector(vector discoveryVector) (model.ModelInfo, error) {
	switch vector.Provider {
	case "openai":
		return openai.MapModel(vector.Input)
	case "anthropic":
		return anthropic.MapModel(vector.Input)
	case "foundry":
		switch vector.Shape {
		case "deployment":
			return openai.MapDeployment(vector.Input)
		case "catalog":
			return openai.MapCatalogModel(vector.Input)
		default:
			return model.ModelInfo{}, fmt.Errorf("unknown foundry shape %q", vector.Shape)
		}
	default:
		return model.ModelInfo{}, fmt.Errorf("unknown provider %q", vector.Provider)
	}
}

// assertModelInfoMatches compares a mapped ModelInfo against a vector's
// expectation.
//
// Every key the vector names is asserted exactly. Keys it does not name must be
// absent or empty, so a mapper cannot pass by inventing a field the contract
// says the provider did not supply.
func assertModelInfoMatches(t *testing.T, info model.ModelInfo, expected map[string]interface{}) {
	t.Helper()

	actual := info.Save(model.NewSaveContext())
	for _, field := range []string{
		"id", "displayName", "ownedBy", "contextWindow",
		"inputModalities", "outputModalities", "additionalProperties",
	} {
		want, wanted := expected[field]
		got, present := actual[field]

		if !wanted {
			if present && !isAbsentModelField(got) {
				t.Errorf("%s = %#v, want absent", field, got)
			}
			continue
		}
		if !present {
			t.Errorf("%s is missing, want %#v", field, want)
			continue
		}
		if !jsonEqual(want, got) {
			t.Errorf("%s = %#v, want %#v", field, got, want)
		}
	}
}

// isAbsentModelField reports whether a serialized field means "not supplied".
// The emitted Save always writes the modality lists, so a nil one is how it
// spells absent.
func isAbsentModelField(value interface{}) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Slice, reflect.Map:
		return reflected.IsNil()
	default:
		return false
	}
}

// jsonEqual compares through JSON so a vector's float64 numbers and Go's int32
// fields compare on value rather than on type.
func jsonEqual(want, got interface{}) bool {
	wantJSON, wantErr := json.Marshal(want)
	gotJSON, gotErr := json.Marshal(got)
	if wantErr != nil || gotErr != nil {
		return false
	}
	var wantValue, gotValue interface{}
	if json.Unmarshal(wantJSON, &wantValue) != nil || json.Unmarshal(gotJSON, &gotValue) != nil {
		return false
	}
	return reflect.DeepEqual(wantValue, gotValue)
}

// ---------------------------------------------------------------------------
// Enrichment
// ---------------------------------------------------------------------------

type enrichmentVectorFile struct {
	Description string             `json:"description"`
	Vectors     []enrichmentVector `json:"vectors"`
}

type enrichmentVector struct {
	Name     string                 `json:"name"`
	Provider string                 `json:"provider"`
	Input    map[string]interface{} `json:"input"`
	Expected map[string]interface{} `json:"expected"`
}

// TestEnrichmentVectors drives every shared enrichment vector through
// prompty.EnrichModelInfo, validating the fill-only-missing rule independently
// of any provider's wire mapping.
func TestEnrichmentVectors(t *testing.T) {
	var file enrichmentVectorFile
	readVectorFile(t, "discovery/enrichment_vectors.json", &file)
	if len(file.Vectors) == 0 {
		t.Fatal("no enrichment vectors found")
	}

	executed := 0
	for _, vector := range file.Vectors {
		vector := vector
		executed++
		t.Run(vector.Name, func(t *testing.T) {
			info, err := model.LoadModelInfo(vector.Input, model.NewLoadContext())
			if err != nil {
				t.Fatalf("load base ModelInfo: %v", err)
			}
			// The distinction between an absent modality list and a
			// provider-supplied empty one is the whole point of several
			// vectors, so it is asserted on the loaded base before enriching.
			assertModalityPresence(t, "input", vector.Input["inputModalities"], info.InputModalities)
			assertModalityPresence(t, "output", vector.Input["outputModalities"], info.OutputModalities)

			prompty.EnrichModelInfo(vector.Provider, &info)
			assertModelInfoMatches(t, info, vector.Expected)
		})
	}
	t.Logf("enrichment vectors: %d executed, %d total", executed, len(file.Vectors))
}

func assertModalityPresence(t *testing.T, label string, raw interface{}, loaded []string) {
	t.Helper()
	_, supplied := raw.([]interface{})
	if supplied && loaded == nil {
		t.Fatalf("%s modalities were supplied by the vector but loaded as absent", label)
	}
	if !supplied && loaded != nil {
		t.Fatalf("%s modalities were absent in the vector but loaded as present", label)
	}
}

// TestVendoredCapabilityDatasetMatchesSpec guards the vendored copy of the
// shared capability dataset against drift from the canonical one in spec/.
//
// Go can only embed files inside the module, so the dataset is duplicated. The
// two must stay byte-identical or the shared enrichment vectors stop meaning
// the same thing in Go as in every other runtime.
func TestVendoredCapabilityDatasetMatchesSpec(t *testing.T) {
	canonical := filepath.Join(specRootForTest(t), "data", "model_capabilities.json")
	raw, err := os.ReadFile(canonical) // #nosec G304 -- derived from the repo layout.
	if err != nil {
		t.Skipf("canonical dataset not available at %s: %v", canonical, err)
	}
	if !jsonBytesEqual(t, raw, prompty.CapabilityDatasetJSON()) {
		t.Fatalf("runtime/go/prompty/data/model_capabilities.json is out of sync with %s — re-copy the canonical file", canonical)
	}
}

func jsonBytesEqual(t *testing.T, left, right []byte) bool {
	t.Helper()
	var leftValue, rightValue interface{}
	if err := json.Unmarshal(left, &leftValue); err != nil {
		t.Fatalf("canonical dataset is not valid JSON: %v", err)
	}
	if err := json.Unmarshal(right, &rightValue); err != nil {
		t.Fatalf("vendored dataset is not valid JSON: %v", err)
	}
	return reflect.DeepEqual(leftValue, rightValue)
}

// ---------------------------------------------------------------------------
// Focused discovery unit tests
// ---------------------------------------------------------------------------

func TestEnrichUnknownModelIsNoop(t *testing.T) {
	info := model.ModelInfo{Id: "ft:custom-model:acme::xyz"}
	prompty.EnrichModelInfo("openai", &info)

	if info.ContextWindow != nil {
		t.Errorf("contextWindow = %v, want nil for an unknown id", *info.ContextWindow)
	}
	if info.InputModalities != nil || info.OutputModalities != nil {
		t.Errorf("modalities = %v/%v, want nil for an unknown id", info.InputModalities, info.OutputModalities)
	}
}

func TestEnrichUnknownProviderIsNoop(t *testing.T) {
	// A provider with no dataset section must not fall through to another
	// provider's table: gpt-4o is an OpenAI id, and a Bedrock deployment that
	// happens to reuse the name must not inherit OpenAI's capabilities.
	info := model.ModelInfo{Id: "gpt-4o"}
	prompty.EnrichModelInfo("bedrock", &info)

	if info.ContextWindow != nil {
		t.Errorf("contextWindow = %v, want nil for an unknown provider", *info.ContextWindow)
	}
}

func TestLookupCapabilitiesRespectsTokenBoundary(t *testing.T) {
	cases := []struct {
		id    string
		found bool
	}{
		{"gpt-4", true},
		{"gpt-4-0613", true},
		{"gpt-4o-mini-2024-07-18", true},
		{"gpt-45", false},
		{"gpt-45-future", false},
		{"gpt-4omini", false},
		{"", false},
	}
	for _, testCase := range cases {
		_, found := prompty.LookupModelCapabilities("openai", testCase.id)
		if found != testCase.found {
			t.Errorf("LookupModelCapabilities(openai, %q) found = %t, want %t", testCase.id, found, testCase.found)
		}
	}
}

func TestLookupCapabilitiesPrefersLongestPrefix(t *testing.T) {
	// gpt-4o-mini and gpt-4o both match; the more specific one must win even
	// though they carry the same context window, because a future divergence
	// between them must not depend on file order.
	mini, ok := prompty.LookupModelCapabilities("openai", "gpt-4o-mini-2024-07-18")
	if !ok {
		t.Fatal("gpt-4o-mini-2024-07-18 should match the dataset")
	}
	if mini.ContextWindow == nil || *mini.ContextWindow != 128000 {
		t.Errorf("contextWindow = %v, want 128000", mini.ContextWindow)
	}
	if got := fmt.Sprint(mini.InputModalities); got != "[text image]" {
		t.Errorf("inputModalities = %s, want [text image]", got)
	}
}

func TestLookupCapabilitiesReturnsIndependentSlices(t *testing.T) {
	// The table is shared and immutable; a caller that mutates a lookup result
	// must not corrupt it for every other caller.
	first, ok := prompty.LookupModelCapabilities("openai", "gpt-4o")
	if !ok || len(first.InputModalities) == 0 {
		t.Fatal("gpt-4o should match with input modalities")
	}
	first.InputModalities[0] = "mutated"

	second, _ := prompty.LookupModelCapabilities("openai", "gpt-4o")
	if second.InputModalities[0] != "text" {
		t.Errorf("shared table was mutated through a lookup result: %v", second.InputModalities)
	}
}

func TestCapabilityLookupIsConcurrencySafe(t *testing.T) {
	// The table is built under a sync.Once and never mutated afterwards, so
	// concurrent first-use must neither race nor return a half-built table.
	const goroutines = 32
	var (
		wait    sync.WaitGroup
		results = make([]bool, goroutines)
	)
	for index := 0; index < goroutines; index++ {
		wait.Add(1)
		go func(slot int) {
			defer wait.Done()
			capabilities, ok := prompty.LookupModelCapabilities("openai", "gpt-4o-2024-05-13")
			results[slot] = ok && capabilities.ContextWindow != nil && *capabilities.ContextWindow == 128000
		}(index)
	}
	wait.Wait()

	for index, ok := range results {
		if !ok {
			t.Fatalf("goroutine %d saw an incomplete capability table", index)
		}
	}
}

func TestEnrichModelInfoIgnoresNil(t *testing.T) {
	// A lister enriching an optional result should not need a nil guard.
	prompty.EnrichModelInfo("openai", nil)
}

func TestListModelsEnrichesThroughRegistry(t *testing.T) {
	t.Cleanup(func() { prompty.UnregisterModelLister("fake-openai") })

	prompty.RegisterModelLister("fake-openai", openai.ModelLister{
		Fetch: func(interface{}) ([]interface{}, error) {
			return []interface{}{
				map[string]interface{}{"id": "gpt-4o", "owned_by": "system"},
				map[string]interface{}{"id": "ft:custom:acme"},
			}, nil
		},
	})

	// The lister is registered under a provider key with no dataset section, so
	// nothing is filled: enrichment keys off the *provider*, not the id.
	infos, err := prompty.ListModels("fake-openai", nil)
	if err != nil {
		t.Fatalf("list models: %v", err)
	}
	if len(infos) != 2 {
		t.Fatalf("listed %d models, want 2", len(infos))
	}
	if infos[0].ContextWindow != nil {
		t.Errorf("an unknown provider key must not inherit another provider's capabilities")
	}

	// Registered under the real key, the same listing is enriched.
	prompty.RegisterModelLister(openai.ProviderOpenAI, openai.ModelLister{
		Fetch: func(interface{}) ([]interface{}, error) {
			return []interface{}{map[string]interface{}{"id": "gpt-4o", "owned_by": "system"}}, nil
		},
	})
	t.Cleanup(func() { prompty.UnregisterModelLister(openai.ProviderOpenAI) })

	enriched, err := prompty.ListModels(openai.ProviderOpenAI, nil)
	if err != nil {
		t.Fatalf("list models: %v", err)
	}
	if len(enriched) != 1 || enriched[0].ContextWindow == nil || *enriched[0].ContextWindow != 128000 {
		t.Fatalf("registry listing was not enriched: %#v", enriched)
	}
}

func TestListModelsUnknownProviderIsAnInvokerError(t *testing.T) {
	if _, err := prompty.ListModels("nobody-registered-this", nil); err == nil {
		t.Fatal("expected an error for an unregistered provider")
	}
}

func TestListModelsContextHonoursCancellation(t *testing.T) {
	prompty.RegisterModelLister("cancel-test", openai.ModelLister{
		Fetch: func(interface{}) ([]interface{}, error) {
			t.Error("a cancelled listing must not reach the provider")
			return nil, nil
		},
	})
	t.Cleanup(func() { prompty.UnregisterModelLister("cancel-test") })

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := prompty.ListModelsContext(ctx, "cancel-test", nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestModelListerPropagatesFetchFailure(t *testing.T) {
	// A discovery failure must surface, not become an empty listing: an empty
	// list and a failed call are indistinguishable to a caller that only sees
	// the models, and one of them means "retry".
	failure := errors.New("provider unreachable")
	prompty.RegisterModelLister("broken", openai.ModelLister{
		Fetch: func(interface{}) ([]interface{}, error) { return nil, failure },
	})
	t.Cleanup(func() { prompty.UnregisterModelLister("broken") })

	if _, err := prompty.ListModels("broken", nil); !errors.Is(err, failure) {
		t.Fatalf("err = %v, want the fetch failure", err)
	}
}

func TestModelListerRejectsMalformedEntry(t *testing.T) {
	prompty.RegisterModelLister("malformed", openai.ModelLister{
		Fetch: func(interface{}) ([]interface{}, error) {
			return []interface{}{map[string]interface{}{"object": "model"}}, nil
		},
	})
	t.Cleanup(func() { prompty.UnregisterModelLister("malformed") })

	if _, err := prompty.ListModels("malformed", nil); err == nil {
		t.Fatal("a model entry with no id must fail the listing rather than be dropped")
	}
}

func TestAnthropicListerFillsOwner(t *testing.T) {
	anthropic.RegisterModelLister(anthropic.ModelLister{
		Fetch: func(interface{}) ([]interface{}, error) {
			return []interface{}{map[string]interface{}{"id": "claude-3-haiku-20240307", "type": "model"}}, nil
		},
	})
	t.Cleanup(func() { prompty.UnregisterModelLister(anthropic.Provider) })

	infos, err := prompty.ListModels(anthropic.Provider, nil)
	if err != nil {
		t.Fatalf("list models: %v", err)
	}
	if len(infos) != 1 || infos[0].OwnedBy == nil || *infos[0].OwnedBy != "anthropic" {
		t.Fatalf("anthropic listing did not attribute an owner: %#v", infos)
	}
	// Anthropic has no dataset section, so a sparse entry stays sparse.
	if infos[0].ContextWindow != nil {
		t.Errorf("contextWindow = %v, want nil", *infos[0].ContextWindow)
	}
}

func TestModelListerRegistryKeysAreSorted(t *testing.T) {
	lister := openai.ModelLister{Fetch: func(interface{}) ([]interface{}, error) { return nil, nil }}
	prompty.RegisterModelLister("zeta", lister)
	prompty.RegisterModelLister("alpha", lister)
	t.Cleanup(func() {
		prompty.UnregisterModelLister("zeta")
		prompty.UnregisterModelLister("alpha")
	})

	keys := prompty.ModelListerKeys()
	seenAlpha, seenZeta := -1, -1
	for index, key := range keys {
		switch key {
		case "alpha":
			seenAlpha = index
		case "zeta":
			seenZeta = index
		}
	}
	if seenAlpha < 0 || seenZeta < 0 || seenAlpha > seenZeta {
		t.Errorf("ModelListerKeys() = %v, want sorted order", keys)
	}
	if !prompty.HasModelLister("alpha") {
		t.Error("HasModelLister should report a registered key")
	}
}

// specRootForTest locates the repository's spec directory from the test's
// working directory.
func specRootForTest(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	start := dir
	for {
		candidate := filepath.Join(dir, "spec")
		if info, err := os.Stat(filepath.Join(candidate, "vectors")); err == nil && info.IsDir() {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skipf("shared spec/ directory not found above %s", start)
			return ""
		}
		dir = parent
	}
}
