package anthropic

import (
	prompty "prompty"
)

// Registration is explicit: importing this package installs nothing, so a host
// that never talks to Anthropic never links a network client it did not ask
// for. Call Register during start-up, before the first Run.
//
//	import (
//	    "prompty"
//	    "prompty/anthropic"
//	)
//
//	func main() {
//	    anthropic.Register()
//	    // ...
//	}

// Provider is the registry key matched against an agent's model.provider.
const Provider = "anthropic"

// Register installs the Anthropic executor and processor under "anthropic".
func Register() {
	RegisterAs(Provider, NewExecutor(), NewProcessor())
}

// RegisterAs installs a specific executor and processor under a provider key.
// Hosts use it to register a configured executor — a custom HTTP client, an
// injected credential source — instead of the defaults.
func RegisterAs(provider string, executor *Executor, processor *Processor) {
	prompty.RegisterExecutor(provider, executor)
	prompty.RegisterProcessor(provider, processor)
}
