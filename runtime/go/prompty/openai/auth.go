package openai

import (
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
)

// FoundryTokenScope is the Microsoft Entra ID scope used by Azure AI Foundry
// inference and project deployment APIs.
const FoundryTokenScope = "https://ai.azure.com/.default"

// TokenProvider acquires a bearer token for Azure AI Foundry.
//
// Hosts may implement this interface to supply tokens from their own credential
// broker. Executor uses DefaultAzureCredential when TokenProvider is nil and no
// explicit Foundry credential is present on the connection or in the environment.
type TokenProvider interface {
	Token(context.Context) (string, error)
}

// AzureCredentialTokenProvider adapts an Azure SDK TokenCredential to the
// provider-neutral TokenProvider contract.
type AzureCredentialTokenProvider struct {
	credential azcore.TokenCredential
	scopes     []string
}

// NewAzureCredentialTokenProvider wraps credential and requests the given
// scopes. An empty scopes list uses FoundryTokenScope.
func NewAzureCredentialTokenProvider(
	credential azcore.TokenCredential,
	scopes ...string,
) (*AzureCredentialTokenProvider, error) {
	if credential == nil {
		return nil, errors.New("openai: Azure token credential is nil")
	}
	if len(scopes) == 0 {
		scopes = []string{FoundryTokenScope}
	}
	return &AzureCredentialTokenProvider{
		credential: credential,
		scopes:     append([]string(nil), scopes...),
	}, nil
}

// NewDefaultAzureCredentialTokenProvider creates the ambient Azure credential
// chain used by the Azure SDK (environment, workload identity, managed identity,
// Azure CLI, and the other supported developer credentials).
func NewDefaultAzureCredentialTokenProvider() (*AzureCredentialTokenProvider, error) {
	credential, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		return nil, err
	}
	return NewAzureCredentialTokenProvider(credential)
}

// Token acquires an access token and rejects an empty success response.
func (p *AzureCredentialTokenProvider) Token(ctx context.Context) (string, error) {
	if p == nil || p.credential == nil {
		return "", errors.New("openai: Azure token provider is not configured")
	}
	token, err := p.credential.GetToken(ctx, policy.TokenRequestOptions{Scopes: p.scopes})
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(token.Token) == "" {
		return "", errors.New("openai: Azure credential returned an empty token")
	}
	return token.Token, nil
}

var (
	defaultFoundryTokenProviderMu sync.Mutex
	defaultFoundryTokenProvider   TokenProvider
)

func ambientFoundryTokenProvider() (TokenProvider, error) {
	defaultFoundryTokenProviderMu.Lock()
	defer defaultFoundryTokenProviderMu.Unlock()
	if defaultFoundryTokenProvider != nil {
		return defaultFoundryTokenProvider, nil
	}
	provider, err := NewDefaultAzureCredentialTokenProvider()
	if err != nil {
		// Construction failures may be transient (for example, a workload
		// identity file mounted after process startup), so only cache success.
		return nil, err
	}
	defaultFoundryTokenProvider = provider
	return provider, nil
}
