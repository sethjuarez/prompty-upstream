package prompty

import (
	"sync"

	model "prompty/model"
)

// Steering (spec §13.5) lets a host inject messages into a running turn.
//
// The queue exists because an agent turn can run for many iterations, and the
// user may want to redirect it — "actually, in Celsius" — without cancelling
// and starting over. Queued text is drained between iterations and appended as
// user messages, so the next model call sees it as ordinary conversation.
//
// The drain happens before every model call *after the first*: by the time the
// loop runs, the opening prompt has already been built, so anything the caller
// wants the model to see first belongs in the prompt rather than in the queue.
// Text queued before the turn starts is not discarded — it is injected at the
// next iteration boundary.
//
// Send is safe to call from any goroutine while the loop is running: the whole
// point is that steering arrives from outside the turn.

// Steering is a concurrency-safe FIFO queue of steering messages.
//
// The zero value is ready to use, and a Steering value may be copied by
// reference (it holds a pointer to shared state), so a host can hand the same
// queue to a UI goroutine and to RunOptions.
type Steering struct {
	mu    sync.Mutex
	queue []string
}

// NewSteering returns an empty steering queue.
func NewSteering() *Steering { return &Steering{} }

// Send enqueues a steering message. It is safe to call concurrently with a
// running turn.
func (s *Steering) Send(message string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.queue = append(s.queue, message)
}

// SendAll enqueues several messages, preserving their order relative to each
// other and to anything already queued.
func (s *Steering) SendAll(messages ...string) {
	if s == nil || len(messages) == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.queue = append(s.queue, messages...)
}

// Drain atomically removes every queued message and returns them as user
// messages in FIFO order.
//
// The drain is atomic so a burst of steering sent while the loop was mid-call
// arrives together in the next iteration, rather than being split across two
// model calls where the model would see half an instruction.
func (s *Steering) Drain() []model.Message {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	pending := s.queue
	s.queue = nil
	s.mu.Unlock()

	if len(pending) == 0 {
		return nil
	}
	messages := make([]model.Message, 0, len(pending))
	for _, text := range pending {
		messages = append(messages, model.NewUserMessage(text))
	}
	return messages
}

// Len reports how many messages are queued.
func (s *Steering) Len() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.queue)
}

// HasPending reports whether any message is queued.
func (s *Steering) HasPending() bool { return s.Len() > 0 }

// IsEmpty reports whether the queue is empty.
func (s *Steering) IsEmpty() bool { return s.Len() == 0 }
