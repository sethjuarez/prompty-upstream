package openai

import (
	"context"

	prompty "prompty"
	model "prompty/model"
	wire "prompty/wire"
)

// ProcessContext is the cancellation-aware form of Process.
func (p *Processor) ProcessContext(ctx context.Context, agent model.Prompty, response interface{}) (interface{}, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return p.Process(agent, response)
}

// ProcessStream implements model.Processor. It exists for the emitted contract;
// prefer ProcessStreamContext, whose stream can be cancelled.
func (p *Processor) ProcessStream(stream interface{}) (interface{}, error) {
	return p.ProcessStreamContext(context.Background(), stream)
}

// ProcessStreamContext converts the executor's RawStream into processed chunks.
//
// An already-processed *prompty.Stream is passed through so a host can compose
// a decorated stream without the processor unwrapping it.
func (p *Processor) ProcessStreamContext(ctx context.Context, stream interface{}) (interface{}, error) {
	switch s := stream.(type) {
	case RawStream:
		return DecodeStream(ctx, s), nil
	case *RawStream:
		if s == nil {
			return nil, wire.NewProviderError("openai", "process stream", 0, "", "received a nil stream", nil)
		}
		return DecodeStream(ctx, *s), nil
	case *prompty.Stream:
		return s, nil
	default:
		return nil, wire.NewProviderError("openai", "process stream", 0, "",
			"expected an openai.RawStream from the executor", errUnexpectedResponse(stream))
	}
}
