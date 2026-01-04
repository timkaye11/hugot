//go:build !ORT && !XLA && !ALL

package backends

import "errors"

// Seq2Seq functions are disabled when neither ORT nor XLA backends are available.
// Build with -tags="ORT" for ONNX Runtime or -tags="XLA" for GoMLX.

var errSeq2SeqDisabled = errors.New("seq2seq backend not enabled: build with ORT or XLA tag")

// RunSeq2SeqEncoder is disabled without a backend.
func RunSeq2SeqEncoder(_ Seq2SeqBatchInterface, _ *Model, _ string) error {
	return errSeq2SeqDisabled
}

// RunSeq2SeqGenerationGreedy is disabled without a backend.
func RunSeq2SeqGenerationGreedy(_ Seq2SeqBatchInterface, _ Seq2SeqPipelineInterface) error {
	return errSeq2SeqDisabled
}

// RunSeq2SeqGenerationSampling is disabled without a backend.
func RunSeq2SeqGenerationSampling(_ Seq2SeqBatchInterface, _ Seq2SeqPipelineInterface) error {
	return errSeq2SeqDisabled
}
