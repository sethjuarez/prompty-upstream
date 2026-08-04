package prompty

import (
	"errors"
	"fmt"

	model "prompty/model"
)

// Sentinel errors. Every error produced by this package wraps exactly one of
// these, so callers can branch with errors.Is without depending on concrete
// types.
//
// The concrete types below mirror the emitted payload shapes in prompty/model
// (model.FileNotFoundError, model.InvokerError, model.ValidationError) so a host
// can serialize them over the wire without a second vocabulary.
var (
	// ErrFileNotFound reports a missing .prompty file or a missing target of a
	// ${file:...} reference (spec §4.5).
	ErrFileNotFound = errors.New("prompty: file not found")

	// ErrValue reports the ValueError family: malformed frontmatter, an unset
	// ${env:VAR} with no default, a missing required input, template syntax
	// errors (spec §12.4).
	ErrValue = errors.New("prompty: invalid value")

	// ErrInvoker reports a registry miss for a renderer/parser/executor/processor
	// key (spec §11.3).
	ErrInvoker = errors.New("prompty: invoker not registered")

	// ErrFileAccessDenied reports a ${file:...} reference whose canonical target
	// escapes every allowed root (spec §2.11). It also wraps ErrValue so generic
	// ValueError handling still catches it.
	ErrFileAccessDenied = errors.New("prompty: file reference outside allowed roots")
)

// FileNotFoundError is raised when a .prompty file or a ${file:...} target
// cannot be read. It corresponds to model.FileNotFoundError.
type FileNotFoundError struct {
	Message string
	Path    string
	Err     error
}

func (e *FileNotFoundError) Error() string { return e.Message }

func (e *FileNotFoundError) Unwrap() []error {
	if e.Err == nil {
		return []error{ErrFileNotFound}
	}
	return []error{ErrFileNotFound, e.Err}
}

// ToModel converts the error into its emitted wire shape.
func (e *FileNotFoundError) ToModel() model.FileNotFoundError {
	return model.FileNotFoundError{Message: e.Message, Path: e.Path}
}

func newFileNotFound(path string, err error, format string, args ...any) *FileNotFoundError {
	return &FileNotFoundError{Message: fmt.Sprintf(format, args...), Path: path, Err: err}
}

// ValueError is the Go equivalent of the spec's ValueError: a semantically
// invalid document, reference or input. Field carries the offending property
// name when one is known (for example the input name for a missing required
// input), and is empty otherwise.
type ValueError struct {
	Message    string
	Field      string
	Constraint string
	Err        error
}

func (e *ValueError) Error() string { return e.Message }

func (e *ValueError) Unwrap() []error {
	if e.Err == nil {
		return []error{ErrValue}
	}
	return []error{ErrValue, e.Err}
}

// ToModel converts the error into its emitted wire shape.
func (e *ValueError) ToModel() model.ValidationError {
	return model.ValidationError{Message: e.Message, Property: e.Field, Constraint: e.Constraint}
}

func newValueError(format string, args ...any) *ValueError {
	return &ValueError{Message: fmt.Sprintf(format, args...)}
}

// InvokerError is raised when no component is registered for a lookup key
// (spec §11.3). It corresponds to model.InvokerError.
type InvokerError struct {
	Component string
	Key       string
	Message   string
}

func (e *InvokerError) Error() string {
	if e.Message != "" {
		return e.Message
	}
	return fmt.Sprintf("No %s registered for key: %s", e.Component, e.Key)
}

func (e *InvokerError) Unwrap() error { return ErrInvoker }

// ToModel converts the error into its emitted wire shape.
func (e *InvokerError) ToModel() model.InvokerError {
	return model.InvokerError{Message: e.Error(), Component: e.Component, Key: e.Key}
}

func newInvokerError(component, key string) *InvokerError {
	return &InvokerError{Component: component, Key: key}
}
