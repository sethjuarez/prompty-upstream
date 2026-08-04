package harness_test

import (
	"os"
)

// exampleDirectory gives an Example function a scratch directory. Examples get
// no *testing.T, so t.TempDir is unavailable.
func exampleDirectory() (string, func()) {
	directory, err := os.MkdirTemp("", "prompty-harness-example")
	if err != nil {
		panic(err)
	}
	return directory, func() { _ = os.RemoveAll(directory) }
}
