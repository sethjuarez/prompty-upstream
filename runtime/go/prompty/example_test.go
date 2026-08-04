package prompty_test

import (
	"context"
	"fmt"

	prompty "prompty"
	anthropic "prompty/anthropic"
	model "prompty/model"
	openai "prompty/openai"
)

// Example_providerRegistration shows the explicit opt-in that installs a
// provider. Importing a provider package registers nothing on its own.
func Example_providerRegistration() {
	defer prompty.ClearCache()

	openai.Register()        // provider: "openai"
	openai.RegisterFoundry() // providers: "foundry", "azure"
	anthropic.Register()     // provider: "anthropic"

	fmt.Println(prompty.ExecutorKeys())

	// Output:
	// [anthropic azure foundry openai]
}

// Example_turnLoopWithPolicies runs a tool round with a permission callback and
// the optional turn policies, using a scripted transport so the example does no
// network I/O.
func Example_turnLoopWithPolicies() {
	agent, err := prompty.LoadFrontmatter(map[string]interface{}{
		"name":         "example",
		"instructions": "",
		"model":        map[string]interface{}{"id": "gpt-4o-mini", "provider": "openai", "apiType": "chat"},
	}, ".", prompty.LoadOptions{})
	if err != nil {
		fmt.Println("load agent:", err)
		return
	}

	tools := prompty.NewToolRegistry()
	tools.RegisterText("get_weather", func(_ context.Context, args map[string]interface{}) (string, error) {
		return fmt.Sprintf("22C and sunny in %v", args["city"]), nil
	})

	steering := prompty.NewSteering()
	steering.Send("answer in one sentence")

	result, err := prompty.RunMessages(
		context.Background(),
		agent,
		[]model.Message{model.NewUserMessage("What is the weather in Paris?")},
		prompty.RunOptions{
			Tools: tools,
			Permit: func(_ context.Context, call model.ToolCall, _ map[string]interface{}) prompty.PermissionDecision {
				if call.Name == "delete_everything" {
					return prompty.Deny("that tool is not available in this session")
				}
				return prompty.Allow()
			},
			// Trimming, guardrails and steering are all optional; leaving any
			// of them unset changes nothing about how the turn runs.
			ContextBudget: 50000,
			Steering:      steering,
			Guardrails: &prompty.Guardrails{
				Output: func(context.Context, interface{}, model.Prompty) prompty.GuardrailResult {
					return prompty.AllowGuardrail()
				},
			},
			Executor:  exampleExecutor(),
			Processor: openai.NewProcessor(),
		},
	)
	if err != nil {
		fmt.Println("run:", err)
		return
	}

	fmt.Println("answer:", result.Text())
	fmt.Println("iterations:", result.Iterations)
	fmt.Println("tool calls:", len(result.Dispatches))
	fmt.Println("steering injected:", result.SteeringInjected)

	// Output:
	// answer: It is 22C and sunny in Paris.
	// iterations: 2
	// tool calls: 1
	// steering injected: 1
}

// Example_modelDiscovery lists a provider's models through a registered lister
// and shows the shared capability dataset filling what the provider omitted.
func Example_modelDiscovery() {
	defer prompty.UnregisterModelLister(openai.ProviderOpenAI)

	// Listing is behind an interface, so a test or an example registers a
	// recorded response instead of reaching the network.
	openai.RegisterModelLister(openai.ModelLister{
		Fetch: func(interface{}) ([]interface{}, error) {
			return []interface{}{
				map[string]interface{}{"id": "gpt-4o-mini-2024-07-18", "owned_by": "system"},
				map[string]interface{}{"id": "ft:custom-model:acme::xyz"},
			}, nil
		},
	})

	models, err := prompty.ListModels(openai.ProviderOpenAI, nil)
	if err != nil {
		fmt.Println("list models:", err)
		return
	}
	for _, info := range models {
		window := "unknown"
		if info.ContextWindow != nil {
			window = fmt.Sprint(*info.ContextWindow)
		}
		fmt.Printf("%s context=%s input=%v\n", info.Id, window, info.InputModalities)
	}

	// Output:
	// gpt-4o-mini-2024-07-18 context=128000 input=[text image]
	// ft:custom-model:acme::xyz context=unknown input=[]
}

// exampleExecutor scripts a tool round followed by a final answer, in the
// OpenAI chat wire shape so the real processor decodes it.
func exampleExecutor() prompty.Executor {
	responses := []interface{}{
		map[string]interface{}{
			"object": "chat.completion",
			"choices": []interface{}{map[string]interface{}{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "",
					"tool_calls": []interface{}{map[string]interface{}{
						"id": "call-1", "type": "function",
						"function": map[string]interface{}{
							"name":      "get_weather",
							"arguments": `{"city":"Paris"}`,
						},
					}},
				},
			}},
		},
		map[string]interface{}{
			"object": "chat.completion",
			"choices": []interface{}{map[string]interface{}{
				"index":   0,
				"message": map[string]interface{}{"role": "assistant", "content": "It is 22C and sunny in Paris."},
			}},
		},
	}
	index := 0
	return executorFunc(func(model.Prompty, []model.Message) (interface{}, error) {
		if index >= len(responses) {
			return nil, fmt.Errorf("the example scripted %d responses but the loop asked for %d", len(responses), index+1)
		}
		response := responses[index]
		index++
		return response, nil
	})
}
