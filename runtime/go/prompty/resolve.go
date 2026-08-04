package prompty

import (
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"gopkg.in/yaml.v3"
)

// maxReferenceDepth bounds ${file:...} recursion. A referenced file may itself
// contain references; without a bound, a.json -> b.json -> a.json would spin.
// The visited set already breaks exact cycles; the depth bound also stops
// unbounded fan-out through distinct files.
const maxReferenceDepth = 16

// referenceResolver resolves ${env:...} and ${file:...} references across a
// frontmatter tree (spec §2.11, §4.2 step 5).
//
// File access is sandboxed. The canonical target must be contained by the
// .prompty file's own directory or by one of the host-supplied allowed roots.
// Frontmatter cannot add roots — only the caller's LoadOptions can.
type referenceResolver struct {
	// agentDir is the canonical directory of the .prompty file. It is both the
	// base for relative references and an implicit allowed root.
	agentDir string
	// roots is the canonical allowed-root set, agentDir first.
	roots []string
	// hostRoots is the canonical set of roots supplied explicitly by the host.
	// Absolute references are only honoured inside these.
	hostRoots []string
	// visiting guards against ${file:} reference cycles.
	visiting map[string]bool
	// lookupEnv is injected so tests can drive resolution without mutating the
	// process environment.
	lookupEnv func(string) (string, bool)
}

func newReferenceResolver(agentDir string, allowedRoots []string, lookupEnv func(string) (string, bool)) (*referenceResolver, error) {
	if lookupEnv == nil {
		lookupEnv = os.LookupEnv
	}
	canonicalAgentDir, err := canonicalizeDir(agentDir)
	if err != nil {
		return nil, newFileNotFound(agentDir, err, "Prompt directory not accessible: %s", agentDir)
	}

	r := &referenceResolver{
		agentDir:  canonicalAgentDir,
		roots:     []string{canonicalAgentDir},
		visiting:  map[string]bool{},
		lookupEnv: lookupEnv,
	}
	for _, root := range allowedRoots {
		canonical, err := canonicalizeDir(root)
		if err != nil {
			return nil, newFileNotFound(root, err, "Allowed file root not accessible: %s", root)
		}
		r.roots = append(r.roots, canonical)
		r.hostRoots = append(r.hostRoots, canonical)
	}
	return r, nil
}

// resolveTree walks every value in the tree, replacing reference strings in
// place. Resolution is recursive: a resolved ${file:...} payload is itself
// scanned for references (spec §2.11 "walk the entire dict recursively").
func (r *referenceResolver) resolveTree(v interface{}, depth int) (interface{}, error) {
	switch t := v.(type) {
	case string:
		return r.resolveString(t, depth)
	case map[string]interface{}:
		for k, val := range t {
			resolved, err := r.resolveTree(val, depth)
			if err != nil {
				return nil, err
			}
			t[k] = resolved
		}
		return t, nil
	case []interface{}:
		for i, val := range t {
			resolved, err := r.resolveTree(val, depth)
			if err != nil {
				return nil, err
			}
			t[i] = resolved
		}
		return t, nil
	default:
		return v, nil
	}
}

// resolveString resolves a single value. A reference must span the whole string
// (`${protocol:content}`); embedded references are left alone so ordinary prose
// containing a `${` is never mangled.
func (r *referenceResolver) resolveString(s string, depth int) (interface{}, error) {
	if !strings.HasPrefix(s, "${") || !strings.HasSuffix(s, "}") {
		return s, nil
	}
	inner := s[2 : len(s)-1]
	colon := strings.IndexByte(inner, ':')
	if colon < 0 {
		return s, nil
	}
	protocol := strings.ToLower(inner[:colon])
	content := inner[colon+1:]

	switch protocol {
	case "env":
		return r.resolveEnv(content)
	case "file":
		return r.resolveFile(content, depth)
	default:
		// Unknown protocol: leave the value untouched (spec §4.2 default case).
		return s, nil
	}
}

// resolveEnv handles ${env:VAR} and ${env:VAR:default}. The default is
// everything after the second colon, so ${env:URL:https://x} keeps its scheme.
func (r *referenceResolver) resolveEnv(content string) (interface{}, error) {
	name := content
	hasDefault := false
	def := ""
	if idx := strings.IndexByte(content, ':'); idx >= 0 {
		name = content[:idx]
		def = content[idx+1:]
		hasDefault = true
	}
	if value, ok := r.lookupEnv(name); ok {
		return value, nil
	}
	if hasDefault {
		return def, nil
	}
	return nil, &ValueError{
		Message:    "Environment variable '" + name + "' not set",
		Field:      name,
		Constraint: "env",
	}
}

// resolveFile handles ${file:relative/path}. Extension drives parsing: .json is
// parsed as JSON, .yaml/.yml as YAML, everything else is loaded as raw text
// (spec §2.11).
func (r *referenceResolver) resolveFile(ref string, depth int) (interface{}, error) {
	if depth >= maxReferenceDepth {
		return nil, newValueError("File reference nesting exceeds %d levels at '%s'", maxReferenceDepth, ref)
	}

	target, err := r.resolveFilePath(ref)
	if err != nil {
		return nil, err
	}
	if r.visiting[target] {
		return nil, newValueError("Circular file reference detected: %s", target)
	}

	raw, err := r.readConfined(target, ref)
	if err != nil {
		return nil, err
	}

	var value interface{}
	switch strings.ToLower(filepath.Ext(target)) {
	case ".json":
		value, err = decodeJSONValue(raw)
		if err != nil {
			return nil, &ValueError{Message: "Referenced file is not valid JSON: " + ref, Field: ref, Err: err}
		}
	case ".yaml", ".yml":
		var node interface{}
		if err := yaml.Unmarshal(raw, &node); err != nil {
			return nil, &ValueError{Message: "Referenced file is not valid YAML: " + ref, Field: ref, Err: err}
		}
		value = normalizeYAMLValue(node)
	default:
		// Preserve text exactly, minus platform line endings.
		return strings.ReplaceAll(string(raw), "\r\n", "\n"), nil
	}

	r.visiting[target] = true
	defer delete(r.visiting, target)
	return r.resolveTree(value, depth+1)
}

// readConfined reads a file that resolveFilePath has already cleared, then
// re-checks that the bytes came from the file that was cleared.
//
// resolveFilePath validates a *path*; this reads through a *handle*. Between the
// two, an attacker with write access to the sandbox could replace the path with
// a link pointing elsewhere. Go exposes no portable openat2/O_NOFOLLOW, so the
// window cannot be closed entirely; comparing the open handle against the
// validated path with os.SameFile narrows it to the interval between Open and
// Stat and turns a successful swap into an error rather than a silent read.
func (r *referenceResolver) readConfined(target, ref string) ([]byte, error) {
	f, err := os.Open(target)
	if err != nil {
		return nil, newFileNotFound(target, err, "Referenced file not found: %s", ref)
	}
	defer f.Close()

	opened, err := f.Stat()
	if err != nil {
		return nil, newFileNotFound(target, err, "Referenced file not readable: %s", ref)
	}
	if opened.IsDir() {
		return nil, newValueError("File reference '%s' points at a directory", ref)
	}
	validated, err := os.Stat(target)
	if err != nil || !os.SameFile(opened, validated) {
		return nil, r.accessDenied(ref, target)
	}

	raw, err := io.ReadAll(f)
	if err != nil {
		return nil, newFileNotFound(target, err, "Referenced file not readable: %s", ref)
	}
	return raw, nil
}

// resolveFilePath turns a reference into a canonical, sandbox-checked absolute
// path. It performs the containment check twice on purpose: once lexically
// (cheap, rejects `..` before touching the filesystem) and once after symlink
// evaluation (rejects a symlink inside the sandbox pointing out of it).
func (r *referenceResolver) resolveFilePath(ref string) (string, error) {
	if ref == "" {
		return "", newValueError("Empty ${file:} reference")
	}

	candidate := ref
	absoluteRef := filepath.IsAbs(ref)
	if !absoluteRef {
		candidate = filepath.Join(r.agentDir, ref)
	}
	candidate = filepath.Clean(candidate)

	// By default absolute references are rejected outright; they are only
	// honoured inside a root the host explicitly opted into (spec §2.11).
	allowed := r.roots
	if absoluteRef {
		allowed = r.hostRoots
	}

	if !containedByAny(candidate, allowed) {
		return "", r.accessDenied(ref, candidate)
	}

	resolved, err := filepath.EvalSymlinks(candidate)
	if err != nil {
		return "", newFileNotFound(candidate, err, "Referenced file not found: %s", ref)
	}
	if !containedByAny(resolved, allowed) {
		return "", r.accessDenied(ref, resolved)
	}
	return resolved, nil
}

func (r *referenceResolver) accessDenied(ref, target string) error {
	return &ValueError{
		Message:    "File reference '" + ref + "' resolves outside the allowed roots: " + target,
		Field:      ref,
		Constraint: "allowedFileRoots",
		Err:        ErrFileAccessDenied,
	}
}

// canonicalizeDir makes a directory absolute and symlink-free so containment
// checks compare like with like.
func canonicalizeDir(dir string) (string, error) {
	if dir == "" {
		dir = "."
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	return filepath.Clean(resolved), nil
}

// containedByAny reports whether target is root itself or lives beneath it.
//
// Comparison is path-segment based rather than string-prefix based, so
// "/data/prompts-secret" is not treated as being inside "/data/prompts".
func containedByAny(target string, roots []string) bool {
	for _, root := range roots {
		if isContained(target, root) {
			return true
		}
	}
	return false
}

func isContained(target, root string) bool {
	target = filepath.Clean(target)
	root = filepath.Clean(root)
	if pathEqual(target, root) {
		return true
	}
	rel, err := filepath.Rel(root, target)
	if err != nil {
		return false
	}
	if rel == "." {
		return true
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return false
	}
	return !filepath.IsAbs(rel)
}

// pathEqual compares two canonical paths using the same case rules
// filepath.Rel applies on this platform, so the equality fast path and the
// containment path can never disagree.
func pathEqual(a, b string) bool {
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}
