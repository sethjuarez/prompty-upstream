// Package wire holds the provider-neutral primitives that every Prompty
// provider needs: safe inspection of emitted agent/property values, JSON Schema
// projection, model-option dialect mapping, content-part helpers, Server-Sent
// Event decoding and diagnostic redaction.
//
// It imports only prompty/model. The root prompty package and each provider
// package (prompty/openai, prompty/anthropic) may import it, which keeps the
// dependency graph acyclic:
//
//	model  <- wire  <- prompty (root)  <- openai, anthropic
//
// Nothing here performs I/O or reads the environment. Everything is a pure
// function of its arguments so request builders stay deterministic and unit
// testable against the shared spec vectors without a network.
package wire
