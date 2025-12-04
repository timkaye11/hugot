package pipelines

import (
	"errors"
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/tensors"

	"github.com/knights-analytics/hugot/pipelineBackends"
)

// createGLiNERTensorsGoMLX creates ALL tensors needed for GLiNER inference using GoMLX
func createGLiNERTensorsGoMLX(batch *GLiNERBatch, model *pipelineBackends.Model) error {
	batchSize := batch.Size
	maxSeqLen := batch.MaxSequenceLength
	numSpans := batch.NumSpans

	// Prepare all input tensors in the correct order
	inputTensors := make([]*tensors.Tensor, len(model.InputsMeta))

	for i, meta := range model.InputsMeta {
		switch meta.Name {
		case "input_ids":
			// Create input_ids tensor from tokenized input
			backing := make([]int64, batchSize*maxSeqLen)
			for b := 0; b < batchSize; b++ {
				for s := 0; s < maxSeqLen; s++ {
					idx := b*maxSeqLen + s
					if b < len(batch.Input) && s < len(batch.Input[b].TokenIDs) {
						backing[idx] = int64(batch.Input[b].TokenIDs[s])
					}
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, maxSeqLen)

		case "attention_mask":
			// Create attention_mask tensor from tokenized input
			backing := make([]int64, batchSize*maxSeqLen)
			for b := 0; b < batchSize; b++ {
				for s := 0; s < maxSeqLen; s++ {
					idx := b*maxSeqLen + s
					if b < len(batch.Input) && s < len(batch.Input[b].AttentionMask) {
						backing[idx] = int64(batch.Input[b].AttentionMask[s])
					}
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, maxSeqLen)

		case "token_type_ids":
			// Create zero tensor for token_type_ids
			backing := make([]int64, batchSize*maxSeqLen)
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, maxSeqLen)

		case "words_mask":
			// Flatten words_mask: [batch_size, seq_len]
			backing := make([]int64, batchSize*maxSeqLen)
			for b := 0; b < batchSize; b++ {
				for s := 0; s < maxSeqLen; s++ {
					idx := b*maxSeqLen + s
					if b < len(batch.WordsMask) && s < len(batch.WordsMask[b]) {
						backing[idx] = batch.WordsMask[b][s]
					}
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, maxSeqLen)

		case "text_lengths":
			// Flatten text_lengths: [batch_size, 1]
			backing := make([]int64, batchSize)
			for b := 0; b < batchSize; b++ {
				if b < len(batch.TextLengths) && len(batch.TextLengths[b]) > 0 {
					backing[b] = batch.TextLengths[b][0]
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, 1)

		case "span_idx":
			// Flatten span_idx: [batch_size, num_spans, 2]
			backing := make([]int64, batchSize*numSpans*2)
			for b := 0; b < batchSize; b++ {
				for s := 0; s < numSpans; s++ {
					baseIdx := (b*numSpans + s) * 2
					if b < len(batch.SpanIdx) && s < len(batch.SpanIdx[b]) && len(batch.SpanIdx[b][s]) >= 2 {
						backing[baseIdx] = batch.SpanIdx[b][s][0]
						backing[baseIdx+1] = batch.SpanIdx[b][s][1]
					}
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, numSpans, 2)

		case "span_mask":
			// Flatten span_mask: [batch_size, num_spans] - GLiNER expects bool type
			backing := make([]bool, batchSize*numSpans)
			for b := 0; b < batchSize; b++ {
				for s := 0; s < numSpans; s++ {
					idx := b*numSpans + s
					if b < len(batch.SpanMask) && s < len(batch.SpanMask[b]) {
						backing[idx] = batch.SpanMask[b][s] != 0
					}
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, numSpans)

		default:
			return fmt.Errorf("unknown input meta name %s", meta.Name)
		}
	}

	// Update batch with new tensors
	batch.InputValues = inputTensors

	// Set up destroy function
	batch.DestroyInputs = func() error {
		for _, t := range inputTensors {
			t.FinalizeAll()
		}
		return nil
	}

	return nil
}

// runGLiNERSessionOnBatchGoMLX runs the GLiNER model on a batch using GoMLX (pure Go)
func runGLiNERSessionOnBatchGoMLX(batch *GLiNERBatch, p *pipelineBackends.BasePipeline) error {
	if p.Model.GoMLXModel == nil || p.Model.GoMLXModel.Exec == nil {
		return errors.New("GoMLX model not initialized")
	}
	inputTensors, ok := batch.InputValues.([]*tensors.Tensor)
	if !ok {
		return errors.New("expected []*tensors.Tensor for input tensors")
	}

	// Run inference
	outputTensors, err := p.Model.GoMLXModel.Exec.Exec(inputTensors)
	if err != nil {
		return fmt.Errorf("running GLiNER session: %w", err)
	}
	defer func() {
		for _, t := range outputTensors {
			t.FinalizeAll()
		}
	}()

	// Convert output to Go slices
	batch.OutputValues = make([]any, len(outputTensors))
	for i, t := range outputTensors {
		var rawOutput []float32
		tensors.ConstFlatData(t, func(flat []float32) {
			rawOutput = make([]float32, len(flat))
			copy(rawOutput, flat)
		})
		dims := p.Model.OutputsMeta[i].Dimensions

		// Reshape based on actual dimensions
		batch.OutputValues[i] = reshapeGLiNEROutputGoMLX(rawOutput, dims, batch)
	}

	return nil
}

// reshapeGLiNEROutputGoMLX reshapes flat output data to 4D array
func reshapeGLiNEROutputGoMLX(data []float32, dims pipelineBackends.Shape, batch *GLiNERBatch) [][][][]float32 {
	batchSize := batch.Size

	// Determine actual dimensions
	var numWords, numSpans, numClasses int

	// Find max words across batch
	for _, tl := range batch.TextLengths {
		if len(tl) > 0 && int(tl[0]) > numWords {
			numWords = int(tl[0])
		}
	}
	if numWords == 0 {
		numWords = 1
	}

	numSpans = batch.NumSpans
	if numSpans == 0 {
		numSpans = 1
	}

	// Last dimension is num_classes (should be fixed in model)
	if len(dims) >= 4 && dims[3] > 0 {
		numClasses = int(dims[3])
	} else {
		// Infer from data length
		expectedSize := batchSize * numWords * numSpans
		if expectedSize > 0 {
			numClasses = len(data) / expectedSize
		}
		if numClasses == 0 {
			numClasses = 1
		}
	}

	// Reshape to 4D: [batch_size, num_words, num_spans, num_classes]
	result := make([][][][]float32, batchSize)
	idx := 0

	for b := 0; b < batchSize; b++ {
		result[b] = make([][][]float32, numWords)
		for w := 0; w < numWords; w++ {
			result[b][w] = make([][]float32, numSpans)
			for s := 0; s < numSpans; s++ {
				result[b][w][s] = make([]float32, numClasses)
				for c := 0; c < numClasses; c++ {
					if idx < len(data) {
						result[b][w][s][c] = data[idx]
						idx++
					}
				}
			}
		}
	}

	return result
}
