package openai

import (
	prompty "prompty"
)

// Registration is explicit: importing this package installs nothing, so a host
// that never talks to OpenAI never links a network client it did not ask for.
// Call one of the Register functions during start-up, before the first Run.
//
//	import (
//	    "prompty"
//	    "prompty/openai"
//	)
//
//	func main() {
//	    openai.Register()          // provider: openai
//	    openai.RegisterFoundry()   // provider: foundry, azure
//	    // ...
//	}
//
// Each Register call installs an Executor and a Processor under a provider key,
// which is matched against the agent's model.provider (spec §11.3). Register
// twice under the same key and the second call wins, which is how a host swaps
// in its own transport.

// ProviderOpenAI is the registry key for OpenAI and OpenAI-compatible gateways.
const ProviderOpenAI = "openai"

// ProviderFoundry and ProviderAzure are the registry keys for Azure AI Foundry
// and Azure OpenAI. Both resolve their exact request shape from the agent's
// connection, so an agent may name either.
const (
	ProviderFoundry = "foundry"
	ProviderAzure   = "azure"
)

// Register installs the OpenAI executor and processor under "openai".
func Register() {
	RegisterAs(ProviderOpenAI, &Executor{}, NewProcessor())
}

// RegisterFoundry installs the Azure/Foundry executor and processor under both
// "foundry" and "azure".
//
// Each key carries a fallback dialect for agents whose connection says nothing,
// but the connection always wins: an agent registered under "azure" whose
// connection declares kind "foundry" still gets Foundry endpoint and auth
// shaping. See ResolveDialect.
func RegisterFoundry() {
	RegisterAs(ProviderFoundry, &Executor{Dialect: DialectFoundry}, NewProcessor())
	RegisterAs(ProviderAzure, &Executor{Dialect: DialectAzure}, NewProcessor())
}

// RegisterAs installs a specific executor and processor under a provider key.
// Hosts use it to register a configured executor — a custom HTTP client, an
// injected credential source — instead of the defaults.
func RegisterAs(provider string, executor *Executor, processor *Processor) {
	prompty.RegisterExecutor(provider, executor)
	prompty.RegisterProcessor(provider, processor)
}
