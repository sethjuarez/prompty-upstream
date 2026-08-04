package harness

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	model "prompty/model"
)

// The replay journal is newline-delimited JSON: exactly one self-describing
// record per line.
//
//	{"kind":"session","event":{...}}
//	{"kind":"turn","event":{...}}
//	{"kind":"summary","summary":{...}}
//
// The format is chosen for crash behaviour, not elegance. A process that dies
// mid-write leaves at most one torn trailing line, and every complete line
// before it is still a valid, independently parseable record — so a journal is
// readable up to the last durable event without a repair pass.

// Journal record kinds, as written in the `kind` field.
const (
	JournalKindTurn    = "turn"
	JournalKindSession = "session"
	JournalKindSummary = "summary"
)

// ErrJournalClosed is returned when a record is appended after Close. A closed
// journal is a finalised one; silently accepting more records would produce a
// file whose summary does not describe its contents.
var ErrJournalClosed = errors.New("harness: journal is closed")

// JSONLJournalWriter appends replayable records to a newline-delimited JSON
// file. It satisfies model.EventJournalWriter and is safe for concurrent use.
//
// The file handle is held open for the writer's lifetime and each record is
// emitted with a single write of one fully assembled line. That is what keeps
// interleaved records from different goroutines whole: the mutex orders them,
// and the single append-mode write means a record is never split by another.
type JSONLJournalWriter struct {
	mu     sync.Mutex
	path   string
	file   *os.File
	writer *bufio.Writer
	closed bool

	// Sync flushes the record to stable storage before Append returns. It
	// defaults to true because the whole point of a durable journal is to
	// survive the crash that made you want one; a host that is replaying into
	// a scratch file can turn it off for speed.
	Sync bool
}

// NewJSONLJournalWriter opens (creating if needed) a journal at path, creating
// any missing parent directories.
//
// Records are appended, so reopening an existing journal continues it rather
// than truncating: a resumed session keeps the history of the run it resumes.
//
// The path is the host's to choose and is used as given, including an absolute
// one — a host decides where its own sessions are stored. It is NOT confined to
// any root, and filepath.Clean below only normalizes; it does not strip leading
// "..". A caller that derives a journal path from untrusted input (a model, a
// request parameter, a filename in a payload) must validate and confine it
// before calling this.
func NewJSONLJournalWriter(path string) (*JSONLJournalWriter, error) {
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("harness: journal path is required")
	}
	path = filepath.Clean(path)
	if directory := filepath.Dir(path); directory != "" && directory != "." {
		if err := os.MkdirAll(directory, 0o755); err != nil {
			return nil, fmt.Errorf("harness: create journal directory: %w", err)
		}
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("harness: open journal: %w", err)
	}
	return &JSONLJournalWriter{
		path:   path,
		file:   file,
		writer: bufio.NewWriter(file),
		Sync:   true,
	}, nil
}

// Path reports the journal file this writer appends to.
func (w *JSONLJournalWriter) Path() string { return w.path }

// AppendTurn writes a turn event record.
func (w *JSONLJournalWriter) AppendTurn(turnEvent model.TurnEvent) (bool, error) {
	return w.write(map[string]interface{}{
		"kind":  JournalKindTurn,
		"event": turnEvent.Save(model.NewSaveContext()),
	})
}

// AppendSession writes a session event record.
func (w *JSONLJournalWriter) AppendSession(sessionEvent model.SessionEvent) (bool, error) {
	return w.write(map[string]interface{}{
		"kind":  JournalKindSession,
		"event": sessionEvent.Save(model.NewSaveContext()),
	})
}

// Close finalises the journal, optionally writing a summary record first, and
// releases the file.
//
// Close is idempotent: a second call is a no-op rather than an error, so a
// deferred Close after an explicit one is safe. Once closed, appends fail with
// ErrJournalClosed.
func (w *JSONLJournalWriter) Close(summary *model.SessionSummary) (bool, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return true, nil
	}
	if summary != nil {
		if err := w.writeLocked(map[string]interface{}{
			"kind":    JournalKindSummary,
			"summary": summary.Save(model.NewSaveContext()),
		}); err != nil {
			// Mark closed anyway: the file is in an unknown state and further
			// appends would append after a failed summary.
			w.closed = true
			_ = w.file.Close()
			return false, err
		}
	}
	w.closed = true

	if err := w.writer.Flush(); err != nil {
		_ = w.file.Close()
		return false, fmt.Errorf("harness: flush journal: %w", err)
	}
	if w.Sync {
		if err := w.file.Sync(); err != nil {
			_ = w.file.Close()
			return false, fmt.Errorf("harness: sync journal: %w", err)
		}
	}
	if err := w.file.Close(); err != nil {
		return false, fmt.Errorf("harness: close journal: %w", err)
	}
	return true, nil
}

func (w *JSONLJournalWriter) write(record map[string]interface{}) (bool, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return false, ErrJournalClosed
	}
	if err := w.writeLocked(record); err != nil {
		return false, err
	}
	return true, nil
}

// writeLocked serialises one record and emits it as a single line.
//
// The line is fully built in memory before any of it reaches the file: a
// marshalling failure must not leave half a record on disk, because a torn
// record in the middle of a journal is unrecoverable in a way a torn trailing
// one is not.
func (w *JSONLJournalWriter) writeLocked(record map[string]interface{}) error {
	encoded, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("harness: encode journal record: %w", err)
	}
	line := make([]byte, 0, len(encoded)+1)
	line = append(line, encoded...)
	line = append(line, '\n')

	if _, err := w.writer.Write(line); err != nil {
		return fmt.Errorf("harness: write journal record: %w", err)
	}
	if err := w.writer.Flush(); err != nil {
		return fmt.Errorf("harness: flush journal record: %w", err)
	}
	if w.Sync {
		if err := w.file.Sync(); err != nil {
			return fmt.Errorf("harness: sync journal record: %w", err)
		}
	}
	return nil
}

// MemoryJournalWriter buffers journal records in memory. It satisfies
// model.EventJournalWriter and is the writer for a test, or for a host that
// wants to inspect a turn's journal without touching a filesystem.
type MemoryJournalWriter struct {
	mu      sync.Mutex
	records []JournalRecord
	closed  bool
}

// AppendTurn buffers a turn event record.
func (w *MemoryJournalWriter) AppendTurn(turnEvent model.TurnEvent) (bool, error) {
	return w.append(JournalRecord{Kind: JournalKindTurn, Turn: &turnEvent})
}

// AppendSession buffers a session event record.
func (w *MemoryJournalWriter) AppendSession(sessionEvent model.SessionEvent) (bool, error) {
	return w.append(JournalRecord{Kind: JournalKindSession, Session: &sessionEvent})
}

// Close finalises the buffer, optionally appending a summary record.
func (w *MemoryJournalWriter) Close(summary *model.SessionSummary) (bool, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return true, nil
	}
	if summary != nil {
		copied := *summary
		w.records = append(w.records, JournalRecord{Kind: JournalKindSummary, Summary: &copied})
	}
	w.closed = true
	return true, nil
}

// Records returns a copy of the buffered records in write order.
func (w *MemoryJournalWriter) Records() []JournalRecord {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]JournalRecord(nil), w.records...)
}

func (w *MemoryJournalWriter) append(record JournalRecord) (bool, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return false, ErrJournalClosed
	}
	w.records = append(w.records, record)
	return true, nil
}

// JournalRecord is one decoded journal line. Exactly one of Turn, Session and
// Summary is set, selected by Kind.
type JournalRecord struct {
	Kind    string
	Turn    *model.TurnEvent
	Session *model.SessionEvent
	Summary *model.SessionSummary
	// Raw is the record exactly as it appeared on disk, so a reader can reach
	// payload fields the emitted types do not model.
	Raw map[string]interface{}
}

// JournalDefect describes a line that could not be decoded.
//
// Defects are reported rather than thrown because a journal is forensic
// evidence: the records before a corrupt line are still valid and still worth
// reading, and a reader that refuses the whole file on one bad line destroys
// the thing it was meant to recover.
type JournalDefect struct {
	// Line is the 1-based line number within the journal.
	Line int
	// Partial marks a final line with no terminating newline — the signature
	// of a process that died mid-append.
	Partial bool
	// Content is the raw text of the offending line, truncated for logging.
	Content string
	// Err explains why the line could not be decoded, nil for a partial line.
	Err error
}

func (d JournalDefect) String() string {
	if d.Partial {
		return fmt.Sprintf("line %d: partial record (no terminating newline)", d.Line)
	}
	return fmt.Sprintf("line %d: %v", d.Line, d.Err)
}

// maxJournalLine bounds a single record. It is generous enough for a turn event
// carrying a large payload and small enough that a corrupted or hostile journal
// cannot make the reader allocate without limit.
const maxJournalLine = 16 << 20

// errOversizedRecord marks a line past maxJournalLine. It is a per-record
// defect rather than a read failure, so the rest of the journal still loads.
var errOversizedRecord = errors.New("journal record exceeds the maximum record size")

// maxDiscardBytes bounds how far the reader will scan for the next record
// boundary after an oversized one. Beyond it the stream is treated as having no
// further boundary, which fails the read instead of scanning forever.
const maxDiscardBytes = 4 * maxJournalLine

// errUnterminatedStream marks a stream with no record boundary within
// maxDiscardBytes. Unlike an oversized record this is fatal: the reader has no
// way to find where the next record starts.
var errUnterminatedStream = errors.New("journal stream has no record terminator")

// journalDefectContentLimit bounds how much of a bad line is retained for
// diagnosis, so a defect report cannot itself become the memory problem.
const journalDefectContentLimit = 512

// ReadJournal reads every record from a journal file.
//
// It returns the records that decoded, the defects it skipped, and an error
// only for a failure to read the file at all. A torn trailing line — the normal
// result of a crash — is reported as a partial defect and excluded from the
// records, so the caller sees exactly the prefix that was durably written.
func ReadJournal(path string) ([]JournalRecord, []JournalDefect, error) {
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, nil, fmt.Errorf("harness: open journal: %w", err)
	}
	defer file.Close()
	return ReadJournalFrom(file)
}

// ReadJournalFrom reads journal records from an arbitrary reader.
func ReadJournalFrom(reader io.Reader) ([]JournalRecord, []JournalDefect, error) {
	buffered := bufio.NewReaderSize(reader, 64<<10)

	var (
		records []JournalRecord
		defects []JournalDefect
		line    int
	)
	for {
		line++
		text, terminated, err := readJournalLine(buffered)
		if err != nil {
			if !errors.Is(err, errOversizedRecord) {
				// A genuine read failure cannot be skipped past: retrying the
				// same reader would spin forever.
				return records, defects, fmt.Errorf("harness: read journal: %w", err)
			}
			// An oversized record is a defect, not a read failure: the reader
			// consumed it, and the rest of the journal is still readable.
			defects = append(defects, JournalDefect{
				Line:    line,
				Content: truncateForDefect(text),
				Err:     err,
			})
			continue
		}
		if !terminated {
			// The last line had no newline: the writer was interrupted between
			// the payload and its terminator, so the record is not durable.
			if strings.TrimSpace(text) != "" {
				defects = append(defects, JournalDefect{
					Line:    line,
					Partial: true,
					Content: truncateForDefect(text),
				})
			}
			return records, defects, nil
		}

		if strings.TrimSpace(text) == "" {
			continue
		}
		record, decodeErr := decodeJournalLine(text)
		if decodeErr != nil {
			defects = append(defects, JournalDefect{
				Line:    line,
				Content: truncateForDefect(text),
				Err:     decodeErr,
			})
			continue
		}
		records = append(records, record)
	}
}

// readJournalLine reads one line and reports whether it was newline-terminated.
//
// bufio.Reader.ReadSlice cannot make that distinction on its own — it strips
// nothing and reports ErrBufferFull for a long line — so the record is
// accumulated explicitly and the terminator inspected. An unterminated final
// line is the signature of a process that died mid-append, and telling it apart
// from a complete record is the whole point of reading a crashed journal.
//
// The record is capped while it is being consumed, not after: a corrupted
// journal must not be able to make the reader allocate without limit.
func readJournalLine(reader *bufio.Reader) (text string, terminated bool, err error) {
	var builder strings.Builder
	for {
		// ReadSlice hands back the reader's own buffer without growing it, so
		// an oversized record is detected while it is being consumed rather
		// than after the whole thing has already been allocated. A journal is
		// untrusted input once a disk has corrupted it.
		chunk, readErr := reader.ReadSlice('\n')

		// The cap is on record content, not on the terminator: a record whose
		// payload is exactly maxJournalLine bytes is at the limit, not over it.
		content := len(chunk)
		if readErr == nil && content > 0 && chunk[content-1] == '\n' {
			content--
		}

		if builder.Len()+content > maxJournalLine {
			builder.Write(chunk[:maxJournalLine-builder.Len()])
			if readErr == bufio.ErrBufferFull {
				// Skip the rest of the record so the reader lands on the next
				// line. The skip is bounded: a stream with no further newline
				// at all is not a long record, it is a broken stream, and
				// scanning it forever would hang rather than fail.
				if !discardToNewline(reader) {
					return builder.String(), false, fmt.Errorf(
						"%w: no record terminator within %d bytes", errUnterminatedStream, maxDiscardBytes)
				}
			}
			return builder.String(), false, fmt.Errorf("%w: over %d bytes", errOversizedRecord, maxJournalLine)
		}
		builder.Write(chunk)

		switch {
		case readErr == bufio.ErrBufferFull:
			// The record is longer than the read buffer but still within the
			// cap; keep accumulating.
			continue
		case readErr == nil:
			raw := builder.String()
			return strings.TrimRight(raw[:len(raw)-1], "\r"), true, nil
		case errors.Is(readErr, io.EOF):
			// No terminator was found: whatever was read is a fragment.
			return builder.String(), false, nil
		default:
			return builder.String(), false, readErr
		}
	}
}

// discardToNewline drops the remainder of an oversized record, up to
// maxDiscardBytes. It reports whether a terminator was found; false means the
// stream has no further record boundary and cannot be resynchronised.
func discardToNewline(reader *bufio.Reader) bool {
	discarded := 0
	for discarded < maxDiscardBytes {
		chunk, err := reader.ReadSlice('\n')
		discarded += len(chunk)
		if err == bufio.ErrBufferFull {
			continue
		}
		// A terminator was found, or the stream ended — either way there is
		// nothing further to skip.
		return true
	}
	return false
}

func truncateForDefect(text string) string {
	if len(text) <= journalDefectContentLimit {
		return text
	}
	return text[:journalDefectContentLimit] + "..."
}

func decodeJournalLine(text string) (JournalRecord, error) {
	var raw map[string]interface{}
	if err := json.Unmarshal([]byte(text), &raw); err != nil {
		return JournalRecord{}, fmt.Errorf("not a JSON object: %w", err)
	}
	kind, _ := raw["kind"].(string)
	record := JournalRecord{Kind: kind, Raw: raw}

	switch kind {
	case JournalKindTurn:
		payload, ok := raw["event"].(map[string]interface{})
		if !ok {
			return JournalRecord{}, fmt.Errorf("turn record has no event object")
		}
		event, err := loadJournalValue("turn event", func() (model.TurnEvent, error) {
			return model.LoadTurnEvent(payload, model.NewLoadContext())
		})
		if err != nil {
			return JournalRecord{}, fmt.Errorf("decode turn event: %w", err)
		}
		record.Turn = &event
	case JournalKindSession:
		payload, ok := raw["event"].(map[string]interface{})
		if !ok {
			return JournalRecord{}, fmt.Errorf("session record has no event object")
		}
		event, err := loadJournalValue("session event", func() (model.SessionEvent, error) {
			return model.LoadSessionEvent(payload, model.NewLoadContext())
		})
		if err != nil {
			return JournalRecord{}, fmt.Errorf("decode session event: %w", err)
		}
		record.Session = &event
	case JournalKindSummary:
		payload, ok := raw["summary"].(map[string]interface{})
		if !ok {
			return JournalRecord{}, fmt.Errorf("summary record has no summary object")
		}
		summary, err := loadJournalValue("session summary", func() (model.SessionSummary, error) {
			return model.LoadSessionSummary(payload, model.NewLoadContext())
		})
		if err != nil {
			return JournalRecord{}, fmt.Errorf("decode session summary: %w", err)
		}
		record.Summary = &summary
	default:
		return JournalRecord{}, fmt.Errorf("unknown journal record kind %q", kind)
	}
	return record, nil
}

func loadJournalValue[T any](label string, load func() (T, error)) (value T, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("invalid %s shape: %v", label, recovered)
		}
	}()
	return load()
}
