//go:build !XLA && !ALL

package backends

import (
	"errors"

	"github.com/knights-analytics/hugot/options"
)

// GoMLXModel is a stub for when XLA/GoMLX is not enabled
type GoMLXModel struct {
	Backend         any   // Stub for backends.Backend
	Ctx             any   // Stub for *context.Context
	Call            any   // Stub for call function
	Destroy         func()
	MaxCache        int   // MaxCache sets the maximum number of unique input shapes to cache.
	BatchBuckets    []int // BatchBuckets defines bucket sizes for batch dimension padding.
	SequenceBuckets []int // SequenceBuckets defines bucket sizes for sequence length padding.
}

func (m *GoMLXModel) Save(_ any) error {
	return errors.New("GoMLX backend not enabled - build with XLA tag")
}

func createGoMLXModelBackend(_ *Model, _ *options.Options) error {
	return errors.New("GoMLX backend not enabled - build with XLA tag")
}

func runGoMLXSessionOnBatch(_ *PipelineBatch, _ *BasePipeline) error {
	return errors.New("GoMLX backend not enabled - build with XLA tag")
}

func createInputTensorsGoMLX(_ *PipelineBatch, _ *Model, _, _ bool) error {
	return errors.New("GoMLX backend not enabled - build with XLA tag")
}

func createImageTensorsGoXLA(_ *PipelineBatch, _ *Model, _ [][][][]float32) error {
	return errors.New("GoMLX backend not enabled - build with XLA tag")
}
