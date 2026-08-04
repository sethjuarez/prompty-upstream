package prompty

import (
	"context"
	"os"
	"path/filepath"
	"strings"

	model "prompty/model"
)

// SourcePathKey is the metadata key under which Load records the absolute path
// of the .prompty file it came from. PromptyTool resolution and relative asset
// lookups need the origin after the agent has been detached from disk.
const SourcePathKey = "__source_path"

// LoadOptions configures a load. The zero value is the safe default: file
// references are confined to the .prompty file's own directory and environment
// variables come from the process environment.
type LoadOptions struct {
	// AllowedFileRoots lists additional directories that ${file:...} references
	// may read from. This is a host capability: it can only be supplied by the
	// calling application, never by frontmatter (spec §2.11).
	AllowedFileRoots []string

	// LookupEnv overrides environment variable resolution. Leave nil to use
	// os.LookupEnv. Primarily a testing seam — it also lets a host scope a load
	// to a curated variable set instead of the whole process environment.
	LookupEnv func(string) (string, bool)
}

func (o LoadOptions) lookupEnv() func(string) (string, bool) {
	if o.LookupEnv != nil {
		return o.LookupEnv
	}
	return os.LookupEnv
}

// Load reads a .prompty file and returns the typed agent (spec §4).
//
// Loading never reads a .env file: populating the environment is the
// application's job (spec §4.3).
func Load(path string) (model.Prompty, error) {
	return LoadWithOptions(path, LoadOptions{})
}

// LoadWithContext is Load with cancellation. The file is read eagerly, so ctx is
// checked at the stage boundaries rather than mid-read.
func LoadWithContext(ctx context.Context, path string, options LoadOptions) (model.Prompty, error) {
	if err := ctx.Err(); err != nil {
		return model.Prompty{}, err
	}
	return LoadWithOptions(path, options)
}

// LoadWithOptions reads a .prompty file using explicit load options.
func LoadWithOptions(path string, options LoadOptions) (model.Prompty, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return model.Prompty{}, newFileNotFound(path, err, "Prompty file not found: %s", path)
	}

	raw, err := os.ReadFile(abs)
	if err != nil {
		return model.Prompty{}, newFileNotFound(abs, err, "Prompty file not found: %s", abs)
	}

	agent, err := buildAgent(string(raw), abs, options)
	if err != nil {
		return model.Prompty{}, err
	}
	return agent, nil
}

// LoadString parses .prompty content that did not come from disk. basePath is
// the path the content should be treated as living at; its directory anchors
// ${file:...} resolution. An existing directory is used as-is.
func LoadString(raw string, basePath string, options LoadOptions) (model.Prompty, error) {
	abs, err := filepath.Abs(basePath)
	if err != nil {
		return model.Prompty{}, newValueError("Invalid base path: %s", basePath)
	}
	return buildAgent(raw, abs, options)
}

// LoadFrontmatter builds an agent directly from an already-parsed frontmatter
// mapping. Shared test vectors express cases this way, and hosts that generate
// prompts programmatically need the same entry point.
//
// The mapping is deep-copied before use, so a caller may safely reuse or share
// the map it passes in.
func LoadFrontmatter(data map[string]interface{}, basePath string, options LoadOptions) (model.Prompty, error) {
	abs, err := filepath.Abs(basePath)
	if err != nil {
		return model.Prompty{}, newValueError("Invalid base path: %s", basePath)
	}
	cloned, _ := deepCloneValue(data).(map[string]interface{})
	if cloned == nil {
		cloned = map[string]interface{}{}
	}
	return buildAgentFromData(cloned, abs, options)
}

// buildAgent runs the load algorithm from spec §4.2 over raw file content.
func buildAgent(raw string, filePath string, options LoadOptions) (model.Prompty, error) {
	// Normalise line endings first so every downstream offset, regex and
	// trimming rule sees the same text on every platform.
	raw = strings.ReplaceAll(raw, "\r\n", "\n")

	data, body, err := splitFrontmatter(raw)
	if err != nil {
		return model.Prompty{}, err
	}

	// The markdown body becomes `instructions`. Trailing newlines are editor
	// noise; leading and internal whitespace is meaningful to the template.
	body = strings.TrimRight(body, "\n")
	if body != "" {
		data["instructions"] = body
	}

	return buildAgentFromData(data, filePath, options)
}

func buildAgentFromData(data map[string]interface{}, filePath string, options LoadOptions) (model.Prompty, error) {
	agentDir := baseDirFor(filePath)

	resolver, err := newReferenceResolver(agentDir, options.AllowedFileRoots, options.lookupEnv())
	if err != nil {
		return model.Prompty{}, err
	}
	resolved, err := resolver.resolveTree(data, 0)
	if err != nil {
		return model.Prompty{}, err
	}
	data, _ = resolved.(map[string]interface{})

	normalized, err := normalizeFrontmatter(data)
	if err != nil {
		return model.Prompty{}, err
	}

	agent, err := loadTypedAgent(normalized)
	if err != nil {
		return model.Prompty{}, err
	}

	if err := hydrateAgent(&agent, normalized); err != nil {
		return model.Prompty{}, err
	}

	if agent.Metadata == nil {
		agent.Metadata = map[string]interface{}{}
	}
	agent.Metadata[SourcePathKey] = filePath
	return agent, nil
}

// baseDirFor returns the directory that anchors ${file:...} resolution.
//
// Callers may pass either a .prompty file path or, for in-memory documents, the
// directory the document should be treated as living in. Taking Dir() of a
// directory would silently widen the sandbox by one level, so the two cases are
// distinguished explicitly.
func baseDirFor(path string) string {
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		return path
	}
	return filepath.Dir(path)
}

// loadTypedAgent calls the emitted loader behind a recover barrier.
//
// The generated model.LoadPrompty uses unchecked type assertions
// (val.(string), val.(bool), ...) on every scalar field, so a document with
// `name: 123` panics instead of returning an error. Generated code cannot be
// edited here, and a malformed prompt file must never take the host process
// down, so the panic is converted into the ValueError the spec calls for.
func loadTypedAgent(normalized map[string]interface{}) (agent model.Prompty, err error) {
	defer func() {
		if r := recover(); r != nil {
			agent = model.Prompty{}
			err = newValueError("Invalid frontmatter: a field has the wrong type (%v)", r)
		}
	}()
	return model.LoadPrompty(normalized, model.NewLoadContext())
}

// hydrateAgent repopulates the parts of the agent that the emitted loader drops.
//
// Generated files are off limits, so the runtime re-derives inputs, outputs and
// function-tool parameters from the same normalised maps that were handed to
// model.LoadPrompty. Everything else on the agent is exactly what the emitter
// produced.
func hydrateAgent(agent *model.Prompty, data map[string]interface{}) error {
	inputs, err := hydrateProperties(data["inputs"])
	if err != nil {
		return err
	}
	if inputs != nil {
		agent.Inputs = inputs
	}

	outputs, err := hydrateProperties(data["outputs"])
	if err != nil {
		return err
	}
	if outputs != nil {
		agent.Outputs = outputs
	}

	rawTools, _ := data["tools"].([]interface{})
	for i := range agent.Tools {
		if i >= len(rawTools) {
			break
		}
		toolMap, ok := rawTools[i].(map[string]interface{})
		if !ok {
			continue
		}
		params, err := hydrateProperties(toolMap["parameters"])
		if err != nil {
			return err
		}
		if params == nil {
			continue
		}
		if fn, ok := agent.Tools[i].(model.FunctionTool); ok {
			fn.Parameters = params
			agent.Tools[i] = fn
		}
	}
	return nil
}

func cloneStringMap(m map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{}, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// deepCloneValue recursively copies maps and slices so the loader's in-place
// resolution and normalisation never reach data the caller still owns. Leaf
// values are shared, which is safe because they are treated as immutable.
func deepCloneValue(v interface{}) interface{} {
	switch t := v.(type) {
	case map[string]interface{}:
		out := make(map[string]interface{}, len(t))
		for k, val := range t {
			out[k] = deepCloneValue(val)
		}
		return out
	case []interface{}:
		out := make([]interface{}, len(t))
		for i, val := range t {
			out[i] = deepCloneValue(val)
		}
		return out
	default:
		return v
	}
}

// AgentInputs returns the agent's declared input properties as read-only views,
// skipping anything that is not a recognised emitted property type.
func AgentInputs(agent model.Prompty) []PropertyView {
	out := make([]PropertyView, 0, len(agent.Inputs))
	for _, raw := range agent.Inputs {
		if view, ok := viewProperty(raw); ok {
			out = append(out, view)
		}
	}
	return out
}

// SourcePath returns the absolute path the agent was loaded from, if known.
func SourcePath(agent model.Prompty) (string, bool) {
	if agent.Metadata == nil {
		return "", false
	}
	path, ok := agent.Metadata[SourcePathKey].(string)
	return path, ok && path != ""
}
