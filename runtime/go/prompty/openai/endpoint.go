package openai

import (
	"net/url"
	"os"
	"strings"

	model "prompty/model"
	wire "prompty/wire"
)

// Dialect selects how an endpoint URL and its auth header are constructed.
type Dialect string

const (
	// DialectOpenAI is api.openai.com and any OpenAI-compatible gateway:
	// `{base}/v1/{path}` with an `Authorization: Bearer` header.
	DialectOpenAI Dialect = "openai"
	// DialectAzure is Azure OpenAI:
	// `{base}/openai/deployments/{deployment}/{path}?api-version=...` with an
	// `api-key` header.
	DialectAzure Dialect = "azure"
	// DialectFoundry is Azure AI Foundry's OpenAI-compatible surface:
	// `{resource}.openai.azure.com/openai/v1/{path}` with a bearer token.
	DialectFoundry Dialect = "foundry"
)

// DefaultAzureAPIVersion matches the canonical Rust provider. Override it per
// agent with model.options.additionalProperties.apiVersion.
const DefaultAzureAPIVersion = "2025-04-01-preview"

// DefaultOpenAIEndpoint is used when neither the connection nor the environment
// names one.
const DefaultOpenAIEndpoint = "https://api.openai.com"

// Env is the environment lookup used to resolve endpoints and credentials.
// It is injectable so tests never touch the real process environment and so a
// host can supply secrets from its own vault instead.
type Env func(key string) (string, bool)

func (e Env) lookup(key string) string {
	if e == nil {
		value, _ := os.LookupEnv(key)
		return value
	}
	value, _ := e(key)
	return value
}

// Environment variables consulted, in the order the resolvers try them.
const (
	EnvOpenAIAPIKey        = "OPENAI_API_KEY"
	EnvOpenAIBaseURL       = "OPENAI_BASE_URL"
	EnvAzureEndpoint       = "AZURE_OPENAI_ENDPOINT"
	EnvAzureAPIKey         = "AZURE_OPENAI_API_KEY"
	EnvAzureDeployment     = "AZURE_OPENAI_DEPLOYMENT"
	EnvAzureAPIVersion     = "AZURE_OPENAI_API_VERSION"
	EnvInferenceCredential = "AZURE_INFERENCE_CREDENTIAL"
)

// Request is a fully resolved provider call: everything the transport needs and
// nothing it has to work out for itself.
//
// Header carries the credential. It is deliberately separate from every other
// field so that URL and Body can be logged, diffed and asserted on in tests
// while the secret stays in one place that no diagnostic path reads.
type Request struct {
	Method string
	URL    string
	Header map[string]string
	Body   map[string]interface{}
	// AuthHeader names the header carrying the credential, so callers can
	// redact exactly that one without guessing.
	AuthHeader string
}

// RedactedHeader returns the headers with the credential masked, for logging.
func (r Request) RedactedHeader() map[string]string {
	out := make(map[string]string, len(r.Header))
	for key, value := range r.Header {
		if strings.EqualFold(key, r.AuthHeader) {
			out[key] = wire.Redacted
			continue
		}
		out[key] = value
	}
	return out
}

// String renders the request for diagnostics with the credential removed.
func (r Request) String() string {
	return r.Method + " " + wire.RedactURL(r.URL)
}

// pathFor maps an API type to its endpoint path suffix.
func pathFor(apiType string) (string, bool) {
	switch apiType {
	case APITypeChat, APITypeAgent:
		return "chat/completions", true
	case APITypeResponses:
		return "responses", true
	case APITypeEmbedding:
		return "embeddings", true
	case APITypeImage:
		return "images/generations", true
	default:
		return "", false
	}
}

// ResolveDialect decides which endpoint shape an agent's connection implies.
//
// The connection wins over the registered fallback, and deliberately so: the
// connection is the agent author's statement of what the endpoint actually is,
// and getting it wrong produces a 404 or a 401. A host that registers this
// provider under several keys therefore cannot force an agent whose connection
// says `foundry` down the Azure deployment path.
//
// fallback is the executor's configured dialect, used only when the connection
// says nothing. An empty fallback resolves to DialectOpenAI.
func ResolveDialect(fallback Dialect, conn map[string]interface{}) Dialect {
	switch asciiFold(wire.ConnectionString(conn, "kind")) {
	case "foundry":
		return DialectFoundry
	case "azure", "azure_openai", "azureopenai":
		return DialectAzure
	}
	if isAzureHost(wire.ConnectionString(conn, "endpoint")) {
		return DialectAzure
	}
	if fallback != "" {
		return fallback
	}
	return DialectOpenAI
}

// asciiFold lowercases ASCII letters without the byte-length surprises
// strings.ToLower can produce on non-ASCII input.
func asciiFold(s string) string {
	var folded []byte
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < 'A' || c > 'Z' {
			continue
		}
		if folded == nil {
			folded = []byte(s)
		}
		folded[i] = c + ('a' - 'A')
	}
	if folded == nil {
		return s
	}
	return string(folded)
}

var azureHostSuffixes = []string{
	".openai.azure.com",
	".services.ai.azure.com",
	".cognitiveservices.azure.com",
	".azure-api.net",
}

func isAzureHost(endpoint string) bool {
	if endpoint == "" {
		return false
	}
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return false
	}
	host := strings.ToLower(parsed.Hostname())
	for _, suffix := range azureHostSuffixes {
		if strings.HasSuffix(host, suffix) {
			return true
		}
	}
	return false
}

// BuildProviderRequest resolves the URL, headers and body for one call.
//
// It performs no I/O beyond reading the injected environment, so an endpoint or
// header regression is caught by a unit test rather than by a live 404.
func BuildProviderRequest(
	agent model.Prompty,
	messages []model.Message,
	explicit Dialect,
	env Env,
	stream bool,
) (Request, error) {
	apiType := wire.APIType(agent)
	path, ok := pathFor(apiType)
	if !ok {
		return Request{}, &wire.SchemaError{Message: "unsupported OpenAI apiType: " + apiType}
	}
	if stream && apiType != APITypeChat && apiType != APITypeAgent && apiType != APITypeResponses {
		return Request{}, &wire.SchemaError{Message: "streaming is not supported for apiType: " + apiType}
	}

	body, err := BuildRequest(agent, messages)
	if err != nil {
		return Request{}, err
	}
	if stream {
		EnableStreaming(body, apiType)
	}

	conn := wire.ConnectionMap(agent)
	dialect := ResolveDialect(explicit, conn)

	endpoint, err := buildURL(agent, conn, dialect, path, env)
	if err != nil {
		return Request{}, err
	}

	authName, authValue, err := authHeader(conn, dialect, env)
	if err != nil {
		return Request{}, err
	}

	header := map[string]string{
		"Content-Type": "application/json",
		authName:       authValue,
	}
	if stream {
		header["Accept"] = "text/event-stream"
	}

	return Request{
		Method:     "POST",
		URL:        endpoint,
		Header:     header,
		Body:       body,
		AuthHeader: authName,
	}, nil
}

func buildURL(agent model.Prompty, conn map[string]interface{}, dialect Dialect, path string, env Env) (string, error) {
	switch dialect {
	case DialectFoundry:
		base := wire.ConnectionString(conn, "endpoint")
		if base == "" {
			base = env.lookup(EnvAzureEndpoint)
		}
		if base == "" {
			return "", missingEndpointError("foundry", EnvAzureEndpoint)
		}
		return strings.TrimSuffix(foundryBase(base), "/") + "/" + path, nil

	case DialectAzure:
		base := wire.ConnectionString(conn, "endpoint")
		if base == "" {
			base = env.lookup(EnvAzureEndpoint)
		}
		if base == "" {
			return "", missingEndpointError("azure", EnvAzureEndpoint)
		}
		deployment := wire.ModelID(agent, env.lookup(EnvAzureDeployment))
		if deployment == "" {
			return "", &wire.ProviderError{
				Provider: "openai",
				Op:       "resolve deployment",
				Body:     "no deployment name found; set model.id or " + EnvAzureDeployment,
			}
		}
		return strings.TrimSuffix(base, "/") +
			"/openai/deployments/" + url.PathEscape(deployment) + "/" + path +
			"?api-version=" + url.QueryEscape(azureAPIVersion(agent, env)), nil

	default:
		base := wire.ConnectionString(conn, "endpoint")
		if base == "" {
			base = env.lookup(EnvOpenAIBaseURL)
		}
		if base == "" {
			base = DefaultOpenAIEndpoint
		}
		base = strings.TrimSuffix(base, "/")
		// A base that already ends in /v1 — a proxy configured as
		// https://gateway.example.com/openai/v1 — must not get a second one.
		if strings.HasSuffix(base, "/v1") {
			return base + "/" + path, nil
		}
		return base + "/v1/" + path, nil
	}
}

func missingEndpointError(dialect, envVar string) error {
	return &wire.ProviderError{
		Provider: "openai",
		Op:       "resolve endpoint",
		Body:     "no endpoint found for " + dialect + " connection; set model.connection.endpoint or " + envVar,
	}
}

// foundryBase rewrites a Foundry project endpoint into the OpenAI-compatible
// inference endpoint.
//
// A Foundry project URL looks like
// https://res.services.ai.azure.com/api/projects/my-project; the inference
// surface lives at https://res.openai.azure.com/openai/v1. An endpoint that is
// already in inference form is returned unchanged so this is idempotent.
func foundryBase(endpoint string) string {
	base := endpoint
	if idx := strings.Index(base, "/api/projects"); idx >= 0 {
		base = base[:idx]
	}
	base = strings.TrimSuffix(base, "/")

	scheme, rest, found := strings.Cut(base, "://")
	if !found {
		return base
	}
	authority, _, _ := strings.Cut(rest, "/")

	host, port := authority, ""
	if h, p, ok := strings.Cut(authority, ":"); ok && isAllDigits(p) {
		host, port = h, ":"+p
	}
	if resource, ok := strings.CutSuffix(host, ".services.ai.azure.com"); ok {
		host = resource + ".openai.azure.com"
	}

	// The inference path is always /openai/v1; an endpoint that already carries
	// it lands on the same string, which makes this idempotent.
	return scheme + "://" + host + port + "/openai/v1"
}

func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

func azureAPIVersion(agent model.Prompty, env Env) string {
	if extra := wire.AdditionalProperties(agent); extra != nil {
		if version, ok := extra["apiVersion"].(string); ok && version != "" {
			return version
		}
	}
	if version := env.lookup(EnvAzureAPIVersion); version != "" {
		return version
	}
	return DefaultAzureAPIVersion
}

// authHeader resolves the credential header for a dialect.
//
// The returned value is a secret. It is never placed into an error, a log line
// or a Request field other than Header, and RedactedHeader is the only
// supported way to render headers for diagnostics.
func authHeader(conn map[string]interface{}, dialect Dialect, env Env) (string, string, error) {
	switch dialect {
	case DialectFoundry:
		// Foundry authenticates with a bearer token (an Entra ID token or a
		// Foundry inference credential). AZURE_OPENAI_API_KEY is deliberately
		// not a fallback here: an Azure OpenAI key is not a bearer token, and
		// sending it as one turns a misconfiguration into a confusing 401
		// instead of the clear error below.
		token := foundryToken(conn, env)
		if token == "" {
			return "", "", missingCredentialError("foundry", EnvInferenceCredential)
		}
		return "Authorization", "Bearer " + token, nil

	case DialectAzure:
		key := wire.ConnectionString(conn, "apiKey", "api_key", "key")
		if key == "" {
			key = env.lookup(EnvAzureAPIKey)
		}
		if key == "" {
			return "", "", missingCredentialError("azure", EnvAzureAPIKey)
		}
		// Azure OpenAI authenticates with a bare api-key header, not a bearer
		// token; sending Authorization instead is a 401.
		return "api-key", key, nil

	default:
		key := wire.ConnectionString(conn, "apiKey", "api_key", "key")
		if key == "" {
			key = env.lookup(EnvOpenAIAPIKey)
		}
		if key == "" {
			return "", "", missingCredentialError("openai", EnvOpenAIAPIKey)
		}
		return "Authorization", "Bearer " + key, nil
	}
}

func foundryToken(conn map[string]interface{}, env Env) string {
	token := wire.ConnectionString(conn, "token", "accessToken", "access_token", "apiKey", "api_key")
	if token == "" {
		token = env.lookup(EnvInferenceCredential)
	}
	return token
}

func missingCredentialError(dialect, envVar string) error {
	return &wire.ProviderError{
		Provider: "openai",
		Op:       "resolve credential",
		Body: "no credential found for " + dialect +
			" connection; set model.connection.apiKey or " + envVar,
	}
}
