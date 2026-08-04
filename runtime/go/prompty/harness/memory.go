package harness

import (
	"sort"
	"strings"
	"sync"
	"time"

	model "prompty/model"
)

// Durability adapters beyond the turn journal: an agent memory store and the
// helpers a host needs to resume a session from a checkpoint.

// InMemoryMemoryStore holds agent memory entries in process.
//
// It is built on the emitted model.MemoryStore and model.MemoryEntry so a
// snapshot round-trips through the canonical wire shape: a host that later
// swaps this for a database keeps the same on-disk format.
//
// The zero value is ready to use and is safe for concurrent access.
type InMemoryMemoryStore struct {
	mu      sync.RWMutex
	entries []model.MemoryEntry
	// Now stamps entries that arrive without a CreatedAt. Nil means time.Now
	// in UTC, RFC3339.
	Now func() time.Time
}

// NewInMemoryMemoryStore returns an empty memory store.
func NewInMemoryMemoryStore() *InMemoryMemoryStore { return &InMemoryMemoryStore{} }

// Add appends an entry, stamping CreatedAt when the caller left it unset so
// every entry is orderable.
func (s *InMemoryMemoryStore) Add(entry model.MemoryEntry) model.MemoryEntry {
	if entry.CreatedAt == nil {
		entry.CreatedAt = String(s.now().UTC().Format(time.RFC3339Nano))
	}
	entry.Tags = copyStringSlice(entry.Tags)

	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = append(s.entries, entry)
	return entry
}

// AddText appends a plain-text entry in a category.
func (s *InMemoryMemoryStore) AddText(category model.MemoryCategory, content string, tags ...string) model.MemoryEntry {
	return s.Add(model.MemoryEntry{Category: category, Content: content, Tags: tags})
}

// Entries returns every entry in insertion order.
func (s *InMemoryMemoryStore) Entries() []model.MemoryEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneEntries(s.entries)
}

// ByCategory returns the entries in one category, in insertion order.
func (s *InMemoryMemoryStore) ByCategory(category model.MemoryCategory) []model.MemoryEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	matched := make([]model.MemoryEntry, 0, len(s.entries))
	for _, entry := range s.entries {
		if entry.Category == category {
			matched = append(matched, entry)
		}
	}
	return cloneEntries(matched)
}

// Search returns the entries whose content or tags contain the query,
// case-insensitively, in insertion order.
//
// The match is a substring scan rather than anything cleverer on purpose: a
// ranked or embedding search belongs to whatever store a host actually deploys,
// and pretending to offer one here would invite hosts to rely on relevance this
// implementation cannot deliver.
func (s *InMemoryMemoryStore) Search(query string) []model.MemoryEntry {
	needle := strings.ToLower(strings.TrimSpace(query))
	if needle == "" {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	matched := make([]model.MemoryEntry, 0, len(s.entries))
	for _, entry := range s.entries {
		if strings.Contains(strings.ToLower(entry.Content), needle) {
			matched = append(matched, entry)
			continue
		}
		for _, tag := range entry.Tags {
			if strings.Contains(strings.ToLower(tag), needle) {
				matched = append(matched, entry)
				break
			}
		}
	}
	return cloneEntries(matched)
}

// Len reports how many entries are stored.
func (s *InMemoryMemoryStore) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entries)
}

// Clear discards every entry.
func (s *InMemoryMemoryStore) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = nil
}

// Snapshot projects the store onto the emitted model.MemoryStore, which is what
// a host persists.
func (s *InMemoryMemoryStore) Snapshot() model.MemoryStore {
	return model.MemoryStore{Entries: s.Entries()}
}

// Restore replaces the store's contents from a persisted snapshot.
func (s *InMemoryMemoryStore) Restore(snapshot model.MemoryStore) {
	entries := cloneEntries(snapshot.Entries)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = entries
}

func (s *InMemoryMemoryStore) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

func cloneEntries(entries []model.MemoryEntry) []model.MemoryEntry {
	if entries == nil {
		return nil
	}
	out := make([]model.MemoryEntry, len(entries))
	for index, entry := range entries {
		entry.CreatedAt = copyString(entry.CreatedAt)
		entry.Tags = copyStringSlice(entry.Tags)
		out[index] = entry
	}
	return out
}

func copyStringSlice(values []string) []string {
	if values == nil {
		return nil
	}
	out := make([]string, len(values))
	copy(out, values)
	return out
}

// ---------------------------------------------------------------------------
// Checkpoint resume helpers
// ---------------------------------------------------------------------------

// ResumePoint describes where a session can be picked up.
type ResumePoint struct {
	// Checkpoint is the newest checkpoint for the session.
	Checkpoint model.Checkpoint
	// Iteration is the model loop iteration the checkpoint captured, or -1
	// when the checkpoint did not record one.
	Iteration int
	// CompletedIterations is how many checkpoints the session has, which is how
	// many model rounds already completed.
	CompletedIterations int
}

// LatestResumePoint finds the newest checkpoint for a session.
//
// It returns ok=false rather than an error when the session has none: a session
// that never checkpointed is a cold start, not a failure.
func LatestResumePoint(store model.CheckpointStore, sessionId string) (ResumePoint, bool, error) {
	checkpoints, err := store.ListCheckpoints(sessionId)
	if err != nil {
		return ResumePoint{}, false, err
	}
	if len(checkpoints) == 0 {
		return ResumePoint{}, false, nil
	}

	// ListCheckpoints ordering is a store's own promise, so the newest is
	// selected here by checkpoint number rather than trusting the position.
	sorted := append([]model.Checkpoint(nil), checkpoints...)
	sort.SliceStable(sorted, func(i, j int) bool {
		return checkpointNumber(sorted[i]) < checkpointNumber(sorted[j])
	})
	latest := sorted[len(sorted)-1]

	return ResumePoint{
		Checkpoint:          latest,
		Iteration:           checkpointIteration(latest),
		CompletedIterations: len(sorted),
	}, true, nil
}

// checkpointIteration reads the iteration a checkpoint recorded, tolerating the
// numeric types JSON round-tripping produces. It returns -1 when absent, so a
// caller can tell "iteration zero" from "no iteration recorded".
func checkpointIteration(checkpoint model.Checkpoint) int {
	raw, ok := checkpoint.State["iteration"]
	if !ok {
		return -1
	}
	switch value := raw.(type) {
	case int:
		return value
	case int32:
		return int(value)
	case int64:
		return int(value)
	case float64:
		return int(value)
	default:
		return -1
	}
}

// CheckpointOutput reads the model output a checkpoint captured, and whether
// one was recorded at all. A checkpoint taken on a tool-calling round has no
// output, which is different from an output of nil.
func CheckpointOutput(checkpoint model.Checkpoint) (interface{}, bool) {
	value, ok := checkpoint.State["output"]
	if !ok || value == nil {
		return nil, false
	}
	return value, true
}

// ReplayJournalUpTo returns the prefix of a normalized journal that ends at the
// given checkpoint number, which is the journal a resumed run must reproduce
// before it may diverge.
//
// The prefix ends after the checkpoint_created session event whose payload
// names that checkpoint number; a number the journal never reached returns the
// whole journal, since nothing in it can be replayed past its own end.
func ReplayJournalUpTo(records []JournalRecord, checkpointNumber int32) []JournalRecord {
	for index, record := range records {
		if record.Kind != JournalKindSession || record.Session == nil {
			continue
		}
		if record.Session.Type != model.SessionEventTypeCheckpointCreated {
			continue
		}
		if journalCheckpointNumber(record.Session.Payload) == checkpointNumber {
			return append([]JournalRecord(nil), records[:index+1]...)
		}
	}
	return append([]JournalRecord(nil), records...)
}

func journalCheckpointNumber(payload map[string]interface{}) int32 {
	switch value := payload["checkpointNumber"].(type) {
	case int:
		return int32(value)
	case int32:
		return value
	case int64:
		return int32(value)
	case float64:
		return int32(value)
	default:
		return -1
	}
}
