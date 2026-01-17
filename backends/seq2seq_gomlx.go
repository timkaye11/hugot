//go:build !ORT && !ALL

package backends

import (
	"errors"
	"fmt"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// RunSeq2SeqEncoder runs the encoder model on the input batch using GoMLX backend.
func RunSeq2SeqEncoder(batch Seq2SeqBatchInterface, model *Model, runtime string) error {
	if model == nil || model.GoMLXModel == nil {
		return errors.New("encoder model not loaded for GoMLX backend")
	}

	batchSize := batch.GetSize()
	inputTokenIDs := batch.GetInputTokenIDs()
	inputAttentionMask := batch.GetInputAttentionMask()
	maxLen := batch.GetMaxInputLength()

	// Create input tensors using shared utility
	inputIDsFlat, attentionMaskFlat := Flatten2DInt64Pair(inputTokenIDs, inputAttentionMask, batchSize, maxLen)

	inputIDsTensor := tensors.FromFlatDataAndDimensions(inputIDsFlat, batchSize, maxLen)
	attentionMaskTensor := tensors.FromFlatDataAndDimensions(attentionMaskFlat, batchSize, maxLen)

	// Execute encoder
	outputs, err := model.GoMLXModel.Exec.Exec(inputIDsTensor, attentionMaskTensor)
	if err != nil {
		inputIDsTensor.FinalizeAll()
		attentionMaskTensor.FinalizeAll()
		return fmt.Errorf("encoder execution failed: %w", err)
	}

	// inputIDsTensor is no longer needed after Exec() - finalize immediately to free memory
	inputIDsTensor.FinalizeAll()

	if len(outputs) == 0 {
		attentionMaskTensor.FinalizeAll()
		return errors.New("encoder returned no outputs")
	}

	// Move encoder outputs to local storage so they can be used by decoder
	// (which has a different backend instance)
	if err := outputs[0].ToLocal(); err != nil {
		attentionMaskTensor.FinalizeAll()
		for _, out := range outputs {
			out.FinalizeAll()
		}
		return fmt.Errorf("moving encoder hidden states to local: %w", err)
	}
	if err := attentionMaskTensor.ToLocal(); err != nil {
		attentionMaskTensor.FinalizeAll()
		for _, out := range outputs {
			out.FinalizeAll()
		}
		return fmt.Errorf("moving attention mask to local: %w", err)
	}

	// Store encoder hidden states for decoder use
	batch.SetEncoderHiddenStates(outputs[0])
	batch.SetEncoderAttentionMask(attentionMaskTensor)

	// Set cleanup function for encoder outputs
	// NOTE: attentionMaskTensor is stored in batch and used by decoder,
	// so it must be cleaned up here along with other encoder resources
	// inputIDsTensor already finalized above
	batch.SetDestroyEncoder(func() error {
		var errs []error
		errs = append(errs, attentionMaskTensor.FinalizeAll())
		for _, out := range outputs {
			errs = append(errs, out.FinalizeAll())
		}
		return errors.Join(errs...)
	})

	return nil
}

// RunSeq2SeqGenerationGreedy performs greedy decoding using GoMLX backend.
func RunSeq2SeqGenerationGreedy(batch Seq2SeqBatchInterface, pipeline Seq2SeqPipelineInterface) error {
	return runSeq2SeqGenerationGoMLX(batch, pipeline, false)
}

// RunSeq2SeqGenerationSampling performs sampling-based decoding using GoMLX backend.
func RunSeq2SeqGenerationSampling(batch Seq2SeqBatchInterface, pipeline Seq2SeqPipelineInterface) error {
	return runSeq2SeqGenerationGoMLX(batch, pipeline, true)
}

// runSeq2SeqGenerationGoMLX is the main generation loop for GoMLX backend.
func runSeq2SeqGenerationGoMLX(batch Seq2SeqBatchInterface, pipeline Seq2SeqPipelineInterface, doSample bool) error {
	decoderInitModel := pipeline.GetDecoderInitModel()
	decoderModel := pipeline.GetDecoderModel()

	if decoderInitModel == nil || decoderInitModel.GoMLXModel == nil {
		return errors.New("decoder-init model not loaded for GoMLX backend")
	}
	if decoderModel == nil || decoderModel.GoMLXModel == nil {
		return errors.New("decoder model not loaded for GoMLX backend")
	}

	batchSize := batch.GetSize()
	maxNewTokens := pipeline.GetMaxNewTokens()
	decoderStartTokenID := pipeline.GetDecoderStartTokenID()
	eosTokenIDs := pipeline.GetEosTokenIDs()
	topP := pipeline.GetTopP()
	temperature := pipeline.GetTemperature()

	// Initialize generation state
	generatedTokens := make([][]int64, batchSize)
	for i := range generatedTokens {
		generatedTokens[i] = make([]int64, 0, maxNewTokens)
	}
	finished := make([]bool, batchSize)
	finishedCount := 0

	// Get encoder outputs with nil checks
	encoderHiddenStatesAny := batch.GetEncoderHiddenStates()
	if encoderHiddenStatesAny == nil {
		return errors.New("encoder hidden states not set - was Encode() called?")
	}
	encoderHiddenStates, ok := encoderHiddenStatesAny.(*tensors.Tensor)
	if !ok {
		return fmt.Errorf("encoder hidden states has unexpected type %T", encoderHiddenStatesAny)
	}

	encoderAttentionMaskAny := batch.GetEncoderAttentionMask()
	if encoderAttentionMaskAny == nil {
		return errors.New("encoder attention mask not set - was Encode() called?")
	}
	encoderAttentionMask, ok := encoderAttentionMaskAny.(*tensors.Tensor)
	if !ok {
		return fmt.Errorf("encoder attention mask has unexpected type %T", encoderAttentionMaskAny)
	}

	// Initialize decoder input with start token
	currentIDs := make([]int64, batchSize)
	for i := range currentIDs {
		currentIDs[i] = decoderStartTokenID
	}

	// Track KV cache tensors for cleanup
	// encoderPKV stays constant throughout generation (cross-attention KV)
	// decoderPKV gets updated each step (self-attention KV)
	var encoderPKV []*tensors.Tensor
	var decoderPKV []*tensors.Tensor

	// Set cleanup function BEFORE generation loop to ensure cleanup on any error path
	// The cleanup function safely handles nil slices and nil tensors
	batch.SetDestroyDecoder(func() error {
		var errs []error
		// Clean up encoder PKV (cross-attention, constant throughout generation)
		for _, kv := range encoderPKV {
			if kv != nil {
				errs = append(errs, kv.FinalizeAll())
			}
		}
		// Clean up decoder PKV (self-attention, updated each step)
		for _, kv := range decoderPKV {
			if kv != nil {
				errs = append(errs, kv.FinalizeAll())
			}
		}
		return errors.Join(errs...)
	})

	// Generation loop
	var actualSteps int
	for step := 0; step < maxNewTokens && finishedCount < batchSize; step++ {
		actualSteps = step + 1 // Track how many steps we've completed

		// Create input tensor for this step
		inputTensor := tensors.FromFlatDataAndDimensions(currentIDs, batchSize, 1)

		var outputs []*tensors.Tensor
		var err error

		if step == 0 {
			// First step: use decoder-init (no past_key_values)
			// Input order matches ONNX model: encoder_attention_mask, input_ids, encoder_hidden_states
			outputs, err = decoderInitModel.GoMLXModel.Exec.Exec(
				encoderAttentionMask,
				inputTensor,
				encoderHiddenStates,
			)
			// Move attention mask back to local storage after decoder-init uses it,
			// so it can be reused by the decoder model (different backend instance)
			if err == nil {
				if localErr := encoderAttentionMask.ToLocal(); localErr != nil {
					inputTensor.FinalizeAll()
					return fmt.Errorf("moving attention mask to local after step 0: %w", localErr)
				}
			}
		} else {
			// Subsequent steps: use decoder with past_key_values
			// Input order: encoder_attention_mask, input_ids, past_key_values...
			// PKV order: decoder.key, decoder.value, encoder.key, encoder.value (per layer)
			combinedPKV := combineEncoderDecoderPKVGoMLX(encoderPKV, decoderPKV, decoderModel)
			inputs := []any{encoderAttentionMask, inputTensor}
			for _, kv := range combinedPKV {
				inputs = append(inputs, kv)
			}
			outputs, err = decoderModel.GoMLXModel.Exec.Exec(inputs...)
		}

		if err != nil {
			inputTensor.FinalizeAll()
			return fmt.Errorf("decoder step %d failed: %w", step, err)
		}

		if len(outputs) < 1 {
			inputTensor.FinalizeAll()
			return fmt.Errorf("decoder step %d returned no outputs", step)
		}

		// First output is logits - move to local for data extraction
		logits := outputs[0]
		if err := logits.ToLocal(); err != nil {
			inputTensor.FinalizeAll()
			return fmt.Errorf("moving logits to local at step %d: %w", step, err)
		}

		// Update KV cache from remaining outputs
		if len(outputs) > 1 {
			// Move all PKV outputs to local storage
			for i := 1; i < len(outputs); i++ {
				if err := outputs[i].ToLocal(); err != nil {
					inputTensor.FinalizeAll()
					logits.FinalizeAll()
					// Clean up PKV tensors that were already converted successfully
					for j := 1; j < i; j++ {
						outputs[j].FinalizeAll()
					}
					return fmt.Errorf("moving KV cache tensor %d to local at step %d: %w", i-1, step, err)
				}
			}

			if step == 0 {
				// decoder-init outputs both encoder and decoder PKV
				// Split them: outputs are ordered as [decoder.key, decoder.value, encoder.key, encoder.value] per layer
				encoderPKV, decoderPKV = splitEncoderDecoderPKVGoMLX(outputs[1:], decoderInitModel)
			} else {
				// decoder only outputs decoder PKV (self-attention)
				// Clean up old decoder PKV tensors
				for _, kv := range decoderPKV {
					kv.FinalizeAll()
				}
				// All outputs after logits are decoder PKV
				decoderPKV = make([]*tensors.Tensor, len(outputs)-1)
				for i := 1; i < len(outputs); i++ {
					decoderPKV[i-1] = outputs[i]
				}
				// encoderPKV stays the same (from step 0)
			}
		}

		// Get next tokens from logits
		var nextTokens []int64
		if doSample {
			nextTokens, err = sampleFromLogitsGoMLX(logits, topP, temperature)
		} else {
			nextTokens, err = argmaxFromLogitsGoMLX(logits)
		}
		if err != nil {
			inputTensor.FinalizeAll()
			logits.FinalizeAll()
			return fmt.Errorf("failed to get next tokens at step %d: %w", step, err)
		}

		// Update generated sequences
		for i := 0; i < batchSize; i++ {
			if finished[i] {
				continue
			}

			token := nextTokens[i]
			generatedTokens[i] = append(generatedTokens[i], token)
			currentIDs[i] = token

			// Check for EOS
			if eosTokenIDs[token] {
				finished[i] = true
				finishedCount++
			}
		}

		// Cleanup input tensor (logits kept in KV cache cleanup)
		inputTensor.FinalizeAll()
		logits.FinalizeAll()
	}

	batch.SetActualSteps(actualSteps)
	batch.SetGeneratedTokens(generatedTokens)
	batch.SetFinished(finished)
	batch.SetFinishedCount(finishedCount)

	// Cleanup function was already set before the loop to handle error paths
	return nil
}

// argmaxFromLogitsGoMLX extracts the argmax token ID from logits for each batch item.
// Uses shared ArgmaxBatch after extracting and reshaping tensor data.
func argmaxFromLogitsGoMLX(logits *tensors.Tensor) ([]int64, error) {
	shape := logits.Shape()
	if shape.Rank() < 2 || shape.Rank() > 3 {
		return nil, fmt.Errorf("expected logits rank 2 or 3, got %d", shape.Rank())
	}

	batchSize := shape.Dimensions[0]
	vocabSize := shape.Dimensions[shape.Rank()-1]

	// Extract logits as float32 slice
	logitsData, err := extractLogitsFloat32(logits)
	if err != nil {
		return nil, err
	}

	// For 3D tensors, extract only the last position
	if shape.Rank() == 3 {
		seqLen := shape.Dimensions[1]
		logitsData = extractLastPosition(logitsData, batchSize, seqLen, vocabSize)
	}

	// Use shared argmax utility
	return ArgmaxBatch(logitsData, batchSize, vocabSize)
}

// sampleFromLogitsGoMLX samples token IDs from logits with temperature and top-p.
// Uses shared SampleTopPBatch after extracting and reshaping tensor data.
func sampleFromLogitsGoMLX(logits *tensors.Tensor, topP, temperature float32) ([]int64, error) {
	shape := logits.Shape()
	if shape.Rank() < 2 || shape.Rank() > 3 {
		return nil, fmt.Errorf("expected logits rank 2 or 3, got %d", shape.Rank())
	}

	batchSize := shape.Dimensions[0]
	vocabSize := shape.Dimensions[shape.Rank()-1]

	// Extract logits as float32 slice
	logitsData, err := extractLogitsFloat32(logits)
	if err != nil {
		return nil, err
	}

	// For 3D tensors, extract only the last position
	if shape.Rank() == 3 {
		seqLen := shape.Dimensions[1]
		logitsData = extractLastPosition(logitsData, batchSize, seqLen, vocabSize)
	}

	// Use shared sampling utility with buffer pooling
	rng := NewSamplingRNG()
	return SampleTopPBatch(logitsData, batchSize, vocabSize, topP, temperature, rng)
}

// extractLogitsFloat32 extracts logits from a tensor as float32 slice.
func extractLogitsFloat32(logits *tensors.Tensor) ([]float32, error) {
	shape := logits.Shape()
	switch shape.DType {
	case dtypes.Float32:
		return tensors.MustCopyFlatData[float32](logits), nil
	case dtypes.Float64:
		float64Data := tensors.MustCopyFlatData[float64](logits)
		result := make([]float32, len(float64Data))
		for i, v := range float64Data {
			result[i] = float32(v)
		}
		return result, nil
	default:
		return nil, fmt.Errorf("unsupported dtype for logits: %s", shape.DType)
	}
}

// extractLastPosition extracts the last sequence position from 3D logits.
// Input: [batch, seq, vocab] -> Output: [batch, vocab] (last position only)
func extractLastPosition(logitsData []float32, batchSize, seqLen, vocabSize int) []float32 {
	result := make([]float32, batchSize*vocabSize)
	for b := 0; b < batchSize; b++ {
		srcOffset := b*seqLen*vocabSize + (seqLen-1)*vocabSize
		dstOffset := b * vocabSize
		copy(result[dstOffset:dstOffset+vocabSize], logitsData[srcOffset:srcOffset+vocabSize])
	}
	return result
}

// splitEncoderDecoderPKVGoMLX splits the PKV outputs from decoder-init into encoder and decoder PKV.
// decoder-init outputs are ordered as: present.0.decoder.key, present.0.decoder.value,
// present.0.encoder.key, present.0.encoder.value, present.1.decoder.key, ...
// Returns (encoderPKV, decoderPKV) where:
// - encoderPKV contains cross-attention KV (constant throughout generation)
// - decoderPKV contains self-attention KV (updated each step)
func splitEncoderDecoderPKVGoMLX(allPKV []*tensors.Tensor, decoderInitModel *Model) (encoderPKV, decoderPKV []*tensors.Tensor) {
	// Handle nil model gracefully
	if decoderInitModel == nil || len(decoderInitModel.OutputsMeta) == 0 {
		return nil, nil
	}

	// Analyze output names to determine which are encoder vs decoder PKV
	// Output names from decoder-init are like: present.0.decoder.key, present.0.decoder.value,
	// present.0.encoder.key, present.0.encoder.value, ...
	for i, pkv := range allPKV {
		// Output index in model is i+1 (since we skipped logits at index 0)
		outputIdx := i + 1
		if outputIdx >= len(decoderInitModel.OutputsMeta) {
			continue
		}
		outputName := decoderInitModel.OutputsMeta[outputIdx].Name
		if strings.Contains(outputName, ".encoder.") {
			encoderPKV = append(encoderPKV, pkv)
		} else {
			decoderPKV = append(decoderPKV, pkv)
		}
	}
	return encoderPKV, decoderPKV
}

// combineEncoderDecoderPKVGoMLX combines encoder and decoder PKV in the order expected by decoder input.
// Decoder expects: past_key_values.0.decoder.key, past_key_values.0.decoder.value,
// past_key_values.0.encoder.key, past_key_values.0.encoder.value, ...
func combineEncoderDecoderPKVGoMLX(encoderPKV, decoderPKV []*tensors.Tensor, decoderModel *Model) []*tensors.Tensor {
	// Handle nil model gracefully
	if decoderModel == nil || len(decoderModel.InputsMeta) < 2 {
		return nil
	}

	// Number of PKV inputs = total inputs - 2 (encoder_attention_mask, input_ids)
	numPKVInputs := len(decoderModel.InputsMeta) - 2
	result := make([]*tensors.Tensor, numPKVInputs)

	encIdx := 0
	decIdx := 0

	// Iterate through decoder input metadata to determine the correct order
	for i := 2; i < len(decoderModel.InputsMeta); i++ {
		inputName := decoderModel.InputsMeta[i].Name
		resultIdx := i - 2
		if strings.Contains(inputName, ".encoder.") {
			if encIdx < len(encoderPKV) {
				result[resultIdx] = encoderPKV[encIdx]
				encIdx++
			}
		} else {
			if decIdx < len(decoderPKV) {
				result[resultIdx] = decoderPKV[decIdx]
				decIdx++
			}
		}
	}

	return result
}
