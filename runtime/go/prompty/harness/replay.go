package harness

import (
	"encoding/json"
	"fmt"
	"strings"

	model "prompty/model"
)

// Replay normalization projects a runtime journal onto the stable, comparable
// shape every runtime shares: model.ReplayJournalRecord.
//
// A runtime journal carries far more than the orchestration semantics —
// timestamps, event identifiers, durations, provider payloads, telemetry. None
// of that is comparable across runtimes or across runs, and comparing it would
// make replay verification a test of the clock. Normalization keeps only the
// fields that describe *what the harness decided*, in order.

// NormalizeRecord projects one decoded journal record onto its replay shape.
//
// An error is returned rather than a best-effort record when a field the shape
// requires is absent or unrecognised: a replay comparison built on quietly
// dropped fields would pass for the wrong reason.
func NormalizeRecord(record JournalRecord) (model.ReplayJournalRecord, error) {
	switch record.Kind {
	case JournalKindSummary:
		return normalizeSummary(record)
	case JournalKindSession:
		return normalizeSession(record)
	case JournalKindTurn:
		return normalizeTurn(record)
	default:
		return model.ReplayJournalRecord{}, fmt.Errorf("harness: unknown journal record kind %q", record.Kind)
	}
}

// NormalizeRecords projects a whole journal, preserving order.
func NormalizeRecords(records []JournalRecord) ([]model.ReplayJournalRecord, error) {
	out := make([]model.ReplayJournalRecord, 0, len(records))
	for index, record := range records {
		normalized, err := NormalizeRecord(record)
		if err != nil {
			return nil, fmt.Errorf("harness: journal record %d: %w", index, err)
		}
		out = append(out, normalized)
	}
	return out, nil
}

func normalizeSummary(record JournalRecord) (model.ReplayJournalRecord, error) {
	if record.Summary == nil {
		return model.ReplayJournalRecord{}, fmt.Errorf("summary record carries no summary")
	}
	summary := record.Summary
	normalized := model.ReplayJournalRecord{
		Kind:        model.ReplayRecordKindSummary,
		SessionId:   String(summary.SessionId),
		Turns:       copyInt32(summary.Turns),
		Checkpoints: copyInt32(summary.Checkpoints),
	}
	if summary.Status != nil {
		status, err := replayStatus(string(*summary.Status))
		if err != nil {
			return model.ReplayJournalRecord{}, err
		}
		normalized.Status = &status
	}
	return normalized, nil
}

func normalizeSession(record JournalRecord) (model.ReplayJournalRecord, error) {
	if record.Session == nil {
		return model.ReplayJournalRecord{}, fmt.Errorf("session record carries no event")
	}
	event := record.Session
	normalized := model.ReplayJournalRecord{
		Kind:      model.ReplayRecordKindSession,
		Type:      String(string(event.Type)),
		SessionId: copyString(event.SessionId),
		TurnId:    copyString(event.TurnId),
	}
	// Only the terminal session event carries a semantic status; every other
	// session event is a marker, and inventing a status for it would make two
	// runs with different outcomes compare equal at that index.
	if event.Type == model.SessionEventTypeSessionEnd {
		raw, ok := payloadString(event.Payload, "status")
		if !ok {
			return model.ReplayJournalRecord{}, fmt.Errorf("session_end payload has no status")
		}
		status, err := replayStatus(raw)
		if err != nil {
			return model.ReplayJournalRecord{}, err
		}
		normalized.Status = &status
	}
	return normalized, nil
}

func normalizeTurn(record JournalRecord) (model.ReplayJournalRecord, error) {
	if record.Turn == nil {
		return model.ReplayJournalRecord{}, fmt.Errorf("turn record carries no event")
	}
	event := record.Turn
	normalized := model.ReplayJournalRecord{
		Kind:      model.ReplayRecordKindTurn,
		Type:      String(string(event.Type)),
		TurnId:    copyString(event.TurnId),
		Iteration: copyInt32(event.Iteration),
	}
	payload := event.Payload

	switch event.Type {
	case model.TurnEventTypePermissionRequested:
		if value, ok := payloadString(payload, "requestId"); ok {
			normalized.RequestId = String(value)
		}

	case model.TurnEventTypePermissionCompleted:
		// The decision's boolean lands in `success`: it is the record's single
		// boolean slot, and a permission that was refused and a tool that
		// failed are both "this step did not succeed".
		approved, ok := payloadBool(payload, "approved")
		if !ok {
			return model.ReplayJournalRecord{}, fmt.Errorf("permission_completed payload has no approved flag")
		}
		normalized.Success = &approved

	case model.TurnEventTypeToolExecutionStart:
		if value, ok := payloadString(payload, "toolName"); ok {
			normalized.ToolName = String(value)
		}

	case model.TurnEventTypeToolExecutionComplete, model.TurnEventTypeToolResult:
		if value, ok := payloadString(payload, "toolName"); ok {
			normalized.ToolName = String(value)
		}
		if value, ok := payloadBool(payload, "success"); ok {
			normalized.Success = &value
		}
		if value, ok := payloadString(payload, "errorKind"); ok {
			normalized.ErrorKind = String(value)
		}

	case model.TurnEventTypeError:
		if value, ok := payloadString(payload, "errorKind"); ok {
			normalized.ErrorKind = String(value)
		}

	case model.TurnEventTypeTurnEnd:
		raw, ok := payloadString(payload, "status")
		if !ok {
			return model.ReplayJournalRecord{}, fmt.Errorf("turn_end payload has no status")
		}
		status, err := replayStatus(raw)
		if err != nil {
			return model.ReplayJournalRecord{}, err
		}
		normalized.Status = &status
	}
	return normalized, nil
}

func replayStatus(raw string) (model.ReplayRecordStatus, error) {
	switch model.ReplayRecordStatus(raw) {
	case model.ReplayRecordStatusSuccess:
		return model.ReplayRecordStatusSuccess, nil
	case model.ReplayRecordStatusError:
		return model.ReplayRecordStatusError, nil
	case model.ReplayRecordStatusCancelled:
		return model.ReplayRecordStatusCancelled, nil
	default:
		return "", fmt.Errorf("status %q is outside the replay record status set", raw)
	}
}

func payloadString(payload map[string]interface{}, key string) (string, bool) {
	value, ok := payload[key].(string)
	return value, ok && value != ""
}

func payloadBool(payload map[string]interface{}, key string) (bool, bool) {
	value, ok := payload[key].(bool)
	return value, ok
}

func copyInt32(value *int32) *int32 {
	if value == nil {
		return nil
	}
	copied := *value
	return &copied
}

// ReplayKey renders a normalized record as the compact, human-readable string
// the shared harness vectors are written in.
//
//	session:session_start:session-1:turn-1
//	turn:tool_result:0:add:false:permission_denied
//	summary:session-1:success:turns=1:checkpoints=2
//
// The vectors use this form rather than embedded JSON so a diff points at the
// step that changed instead of at a wall of serialized events.
func ReplayKey(record model.ReplayJournalRecord) string {
	switch record.Kind {
	case model.ReplayRecordKindSummary:
		return fmt.Sprintf("summary:%s:%s:turns=%s:checkpoints=%s",
			stringOr(record.SessionId, ""),
			statusOr(record.Status),
			int32Or(record.Turns),
			int32Or(record.Checkpoints))

	case model.ReplayRecordKindSession:
		eventType := stringOr(record.Type, "")
		base := fmt.Sprintf("session:%s:%s:%s",
			eventType, stringOr(record.SessionId, ""), stringOr(record.TurnId, ""))
		if record.Status != nil {
			return base + ":" + statusOr(record.Status)
		}
		return base

	default:
		var builder strings.Builder
		fmt.Fprintf(&builder, "turn:%s:%s", stringOr(record.Type, ""), int32Or(record.Iteration))
		switch {
		case record.RequestId != nil:
			builder.WriteString(":" + *record.RequestId)
		case record.ToolName != nil:
			builder.WriteString(":" + *record.ToolName)
			if record.Success != nil {
				fmt.Fprintf(&builder, ":%t", *record.Success)
			}
			if record.ErrorKind != nil {
				builder.WriteString(":" + *record.ErrorKind)
			}
		case record.Success != nil:
			fmt.Fprintf(&builder, ":%t", *record.Success)
		case record.ErrorKind != nil:
			builder.WriteString(":" + *record.ErrorKind)
		case record.Status != nil:
			builder.WriteString(":" + statusOr(record.Status))
		}
		return builder.String()
	}
}

// ReplayKeys renders a whole normalized journal.
func ReplayKeys(records []model.ReplayJournalRecord) []string {
	out := make([]string, 0, len(records))
	for _, record := range records {
		out = append(out, ReplayKey(record))
	}
	return out
}

func stringOr(value *string, fallback string) string {
	if value == nil {
		return fallback
	}
	return *value
}

func statusOr(status *model.ReplayRecordStatus) string {
	if status == nil {
		return ""
	}
	return string(*status)
}

func int32Or(value *int32) string {
	if value == nil {
		return ""
	}
	return fmt.Sprint(*value)
}

// ReplayVerifier compares an expected normalized journal against an actual one.
// It mirrors the emitted ReferenceReplayVerifier's contract and produces the
// emitted model.ReplayVerificationResult.
type ReplayVerifier struct{}

// Verify aligns the two journals index by index and reports every difference.
//
// Comparison is positional rather than set-based because a replay journal is an
// ordered decision log: the same events in a different order describe a
// different turn, and a set comparison would call that identical.
func (ReplayVerifier) Verify(request model.ReplayVerificationRequest) model.ReplayVerificationResult {
	expected, actual := request.Expected, request.Actual

	longest := len(expected)
	if len(actual) > longest {
		longest = len(actual)
	}

	mismatches := []model.ReplayMismatch{}
	for index := 0; index < longest; index++ {
		var expectedRecord, actualRecord *model.ReplayJournalRecord
		if index < len(expected) {
			expectedRecord = &expected[index]
		}
		if index < len(actual) {
			actualRecord = &actual[index]
		}
		if comparableRecord(expectedRecord) == comparableRecord(actualRecord) {
			continue
		}

		message := "Replay record mismatch"
		switch {
		case expectedRecord == nil:
			message = "Unexpected extra replay record"
		case actualRecord == nil:
			message = "Missing replay record"
		}
		mismatches = append(mismatches, model.ReplayMismatch{
			Index:    int32(index),
			Expected: expectedRecord,
			Actual:   actualRecord,
			Message:  message,
		})
	}

	status := model.ReplayVerificationStatusPassed
	if len(mismatches) > 0 {
		status = model.ReplayVerificationStatusFailed
	}
	return model.ReplayVerificationResult{
		Status:        status,
		Mismatches:    mismatches,
		ExpectedCount: int32(len(expected)),
		ActualCount:   int32(len(actual)),
	}
}

// comparableRecord renders a record through the emitted serializer so
// comparison is on the canonical wire projection rather than on Go pointer
// identity, which two structurally equal records would fail.
func comparableRecord(record *model.ReplayJournalRecord) string {
	if record == nil {
		return "<nil>"
	}
	encoded, err := json.Marshal(record.Save(model.NewSaveContext()))
	if err != nil {
		// Unreachable for the emitted shape (only strings, numbers, booleans),
		// but returning a distinguishable marker keeps a hypothetical failure
		// visible as a mismatch instead of silently comparing equal.
		return fmt.Sprintf("<unencodable:%v>", err)
	}
	return string(encoded)
}

// VerifyJournalFile normalizes a journal on disk and verifies it against an
// expected sequence of replay keys.
//
// This is the shape a regression test wants: point it at the journal a turn
// produced, hand it the keys the vector recorded, and get back the emitted
// verification result.
func VerifyJournalFile(path string, expectedKeys []string) (model.ReplayVerificationResult, []JournalDefect, error) {
	records, defects, err := ReadJournal(path)
	if err != nil {
		return model.ReplayVerificationResult{}, defects, err
	}
	normalized, err := NormalizeRecords(records)
	if err != nil {
		return model.ReplayVerificationResult{}, defects, err
	}
	actualKeys := ReplayKeys(normalized)

	result := model.ReplayVerificationResult{
		Status:        model.ReplayVerificationStatusPassed,
		Mismatches:    []model.ReplayMismatch{},
		ExpectedCount: int32(len(expectedKeys)),
		ActualCount:   int32(len(actualKeys)),
	}
	longest := len(expectedKeys)
	if len(actualKeys) > longest {
		longest = len(actualKeys)
	}
	for index := 0; index < longest; index++ {
		var want, got string
		haveWant := index < len(expectedKeys)
		haveGot := index < len(actualKeys)
		if haveWant {
			want = expectedKeys[index]
		}
		if haveGot {
			got = actualKeys[index]
		}
		if haveWant && haveGot && want == got {
			continue
		}
		message := fmt.Sprintf("Replay record mismatch: expected %q, got %q", want, got)
		switch {
		case !haveWant:
			message = fmt.Sprintf("Unexpected extra replay record %q", got)
		case !haveGot:
			message = fmt.Sprintf("Missing replay record %q", want)
		}
		var actualRecord *model.ReplayJournalRecord
		if haveGot {
			actualRecord = &normalized[index]
		}
		result.Mismatches = append(result.Mismatches, model.ReplayMismatch{
			Index:   int32(index),
			Actual:  actualRecord,
			Message: message,
		})
	}
	if len(result.Mismatches) > 0 {
		result.Status = model.ReplayVerificationStatusFailed
	}
	return result, defects, nil
}
