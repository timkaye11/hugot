//go:build ORT || ALL

package pipelines

import (
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/knights-analytics/hugot/pipelineBackends"
)

// createGLiNERTensorsORT creates ALL tensors needed for GLiNER inference
func createGLiNERTensorsORT(batch *GLiNERBatch, model *pipelineBackends.Model) error {
	batchSize := int64(batch.Size)
	maxSeqLen := int64(batch.MaxSequenceLength)
	numSpans := int64(batch.NumSpans)

	// Prepare all input tensors in the correct order
	inputTensors := make([]ort.Value, len(model.InputsMeta))
	var destroyFuncs []func() error

	for i, meta := range model.InputsMeta {
		var tensor ort.Value
		var err error

		switch meta.Name {
		case "input_ids":
			// Create input_ids tensor from tokenized input
			backing := make([]int64, batchSize*maxSeqLen)
			for b := 0; b < int(batchSize); b++ {
				for s := 0; s < int(maxSeqLen); s++ {
					idx := b*int(maxSeqLen) + s
					if b < len(batch.Input) && s < len(batch.Input[b].TokenIDs) {
						backing[idx] = int64(batch.Input[b].TokenIDs[s])
					}
				}
			}
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, maxSeqLen), backing)
			if err != nil {
				return fmt.Errorf("creating input_ids tensor: %w", err)
			}

		case "attention_mask":
			// Create attention_mask tensor from tokenized input
			backing := make([]int64, batchSize*maxSeqLen)
			for b := 0; b < int(batchSize); b++ {
				for s := 0; s < int(maxSeqLen); s++ {
					idx := b*int(maxSeqLen) + s
					if b < len(batch.Input) && s < len(batch.Input[b].AttentionMask) {
						backing[idx] = int64(batch.Input[b].AttentionMask[s])
					}
				}
			}
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, maxSeqLen), backing)
			if err != nil {
				return fmt.Errorf("creating attention_mask tensor: %w", err)
			}

		case "token_type_ids":
			// Create zero tensor for token_type_ids (not used by GLiNER but may be required)
			backing := make([]int64, batchSize*maxSeqLen)
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, maxSeqLen), backing)
			if err != nil {
				return fmt.Errorf("creating token_type_ids tensor: %w", err)
			}

		case "words_mask":
			// Flatten words_mask: [batch_size, seq_len]
			backing := make([]int64, batchSize*maxSeqLen)
			for b := 0; b < int(batchSize); b++ {
				for s := 0; s < int(maxSeqLen); s++ {
					idx := b*int(maxSeqLen) + s
					if b < len(batch.WordsMask) && s < len(batch.WordsMask[b]) {
						backing[idx] = batch.WordsMask[b][s]
					}
				}
			}
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, maxSeqLen), backing)
			if err != nil {
				return fmt.Errorf("creating words_mask tensor: %w", err)
			}

		case "text_lengths":
			// Flatten text_lengths: [batch_size, 1]
			backing := make([]int64, batchSize)
			for b := 0; b < int(batchSize); b++ {
				if b < len(batch.TextLengths) && len(batch.TextLengths[b]) > 0 {
					backing[b] = batch.TextLengths[b][0]
				}
			}
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, 1), backing)
			if err != nil {
				return fmt.Errorf("creating text_lengths tensor: %w", err)
			}

		case "span_idx":
			// Flatten span_idx: [batch_size, num_spans, 2]
			backing := make([]int64, batchSize*numSpans*2)
			for b := 0; b < int(batchSize); b++ {
				for s := 0; s < int(numSpans); s++ {
					baseIdx := (b*int(numSpans) + s) * 2
					if b < len(batch.SpanIdx) && s < len(batch.SpanIdx[b]) && len(batch.SpanIdx[b][s]) >= 2 {
						backing[baseIdx] = batch.SpanIdx[b][s][0]
						backing[baseIdx+1] = batch.SpanIdx[b][s][1]
					}
				}
			}
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, numSpans, 2), backing)
			if err != nil {
				return fmt.Errorf("creating span_idx tensor: %w", err)
			}

		case "span_mask":
			// Flatten span_mask: [batch_size, num_spans] - GLiNER expects bool type
			backing := make([]bool, batchSize*numSpans)
			for b := 0; b < int(batchSize); b++ {
				for s := 0; s < int(numSpans); s++ {
					idx := b*int(numSpans) + s
					if b < len(batch.SpanMask) && s < len(batch.SpanMask[b]) {
						backing[idx] = batch.SpanMask[b][s] != 0
					}
				}
			}
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, numSpans), backing)
			if err != nil {
				return fmt.Errorf("creating span_mask tensor: %w", err)
			}

		default:
			return fmt.Errorf("unknown input meta name %s", meta.Name)
		}

		inputTensors[i] = tensor
		destroyFuncs = append(destroyFuncs, tensor.Destroy)
	}

	// Update batch with new tensors
	batch.InputValues = inputTensors

	// Set up destroy function
	batch.DestroyInputs = func() error {
		var errs []error
		for _, f := range destroyFuncs {
			if err := f(); err != nil {
				errs = append(errs, err)
			}
		}
		return errors.Join(errs...)
	}

	return nil
}

// runGLiNERSessionOnBatchORT runs the GLiNER model on a batch using ORT
func runGLiNERSessionOnBatchORT(batch *GLiNERBatch, p *pipelineBackends.BasePipeline) error {
	inputTensors, ok := batch.InputValues.([]ort.Value)
	if !ok {
		return errors.New("expected []ort.Value for input tensors")
	}

	// Get max words for calculating output shape
	maxWords := int64(0)
	for _, tl := range batch.TextLengths {
		if len(tl) > 0 && tl[0] > maxWords {
			maxWords = tl[0]
		}
	}
	if maxWords == 0 {
		maxWords = 1
	}

	// GLiNER output shape: [batch_size, num_words, max_width, num_classes]
	// We need to determine num_classes from the number of entity labels
	// The model extracts this from the input, but we need to know it for the output tensor
	// For GLiNER, max_width is fixed (typically 12), and num_classes is derived from entity tokens in input
	maxWidth := int64(12) // Standard GLiNER max_width

	// Count entity tokens (<<ENT>>) in the input to determine num_classes
	// Each <<ENT>> token (id 128002) marks an entity label
	numClasses := int64(0)
	if len(batch.Input) > 0 {
		for _, tokenID := range batch.Input[0].TokenIDs {
			if tokenID == 128002 { // <<ENT>> token ID
				numClasses++
			}
		}
	}
	if numClasses == 0 {
		numClasses = 1
	}

	// Create output tensor with correct shape
	outputShape := ort.NewShape(int64(batch.Size), maxWords, maxWidth, numClasses)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return fmt.Errorf("creating output tensor: %w", err)
	}
	outputTensors := []ort.Value{outputTensor}

	// Run inference
	if err := p.Model.ORTModel.Session.Run(inputTensors, outputTensors); err != nil {
		outputTensor.Destroy()
		return fmt.Errorf("running GLiNER session: %w", err)
	}

	// Convert output to Go slices
	data := outputTensor.GetData()
	batch.OutputValues = []any{reshapeGLiNEROutputFixed(data, int(batch.Size), int(maxWords), int(maxWidth), int(numClasses))}

	// Clean up output tensor
	outputTensor.Destroy()

	return nil
}

// reshapeGLiNEROutputFixed reshapes flat output data to 4D array with known dimensions
func reshapeGLiNEROutputFixed(data []float32, batchSize, numWords, maxWidth, numClasses int) [][][][]float32 {
	result := make([][][][]float32, batchSize)
	idx := 0

	for b := 0; b < batchSize; b++ {
		result[b] = make([][][]float32, numWords)
		for w := 0; w < numWords; w++ {
			result[b][w] = make([][]float32, maxWidth)
			for s := 0; s < maxWidth; s++ {
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

// reshapeGLiNEROutput reshapes flat output data to 4D array
func reshapeGLiNEROutput(data []float32, dims pipelineBackends.Shape, batch *GLiNERBatch) [][][][]float32 {
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
