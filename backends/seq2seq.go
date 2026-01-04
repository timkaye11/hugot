package backends

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// Seq2SeqPipelineInterface defines the interface for seq2seq pipeline access.
// This avoids import cycles between backends and pipelines packages.
type Seq2SeqPipelineInterface interface {
	GetEncoderModel() *Model
	GetDecoderInitModel() *Model
	GetDecoderModel() *Model
	GetTokenizer() *Tokenizer
	GetRuntime() string
	GetMaxNewTokens() int
	GetNumReturnSeqs() int
	GetDoSample() bool
	GetTopP() float32
	GetTemperature() float32
	GetRepetitionPenalty() float32
	GetDecoderStartTokenID() int64
	GetEosTokenIDs() map[int64]bool
	GetPadTokenID() int64
	GetNumDecoderLayers() int
	GetVocabSize() int
	GetNoCacheMode() bool
}

// Seq2SeqBatchInterface defines the interface for seq2seq batch access.
type Seq2SeqBatchInterface interface {
	GetSize() int
	GetInputTokenIDs() [][]int64
	GetInputAttentionMask() [][]int64
	GetMaxInputLength() int
	SetEncoderHiddenStates(states any)
	GetEncoderHiddenStates() any
	SetEncoderAttentionMask(mask any)
	GetEncoderAttentionMask() any
	SetPastKeyValues(pkv []any)
	GetPastKeyValues() []any
	SetLogits(logits any)
	GetLogits() any
	GetGeneratedTokens() [][]int64
	SetGeneratedTokens(tokens [][]int64)
	GetFinished() []bool
	SetFinished(finished []bool)
	GetFinishedCount() int
	SetFinishedCount(count int)
	SetDestroyEncoder(fn func() error)
	SetDestroyDecoder(fn func() error)
}

// Seq2SeqConfig holds configuration for seq2seq models (T5, BART, etc.).
type Seq2SeqConfig struct {
	DecoderStartTokenID int64
	EosTokenIDs         map[int64]bool
	PadTokenID          int64
	NumDecoderLayers    int
	NumHeads            int
	HeadDim             int
	DModel              int // Hidden size (d_model)
	VocabSize           int
}

// Seq2SeqTokenized holds tokenized inputs for seq2seq models.
type Seq2SeqTokenized struct {
	TokenIDs      [][]int64
	AttentionMask [][]int64
	MaxLength     int
}

// LoadSeq2SeqEncoder loads the encoder model for seq2seq inference.
// Looks for encoder.onnx or *-encoder*.onnx in the model path.
func LoadSeq2SeqEncoder(modelPath string, opts *options.Options) (*Model, error) {
	onnxFile, err := findOnnxFile(modelPath, "encoder")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model, opts); err != nil {
		return nil, err
	}

	if err := CreateModelBackend(model, opts); err != nil {
		return nil, err
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadSeq2SeqDecoderInit loads the initial decoder model (no past_key_values).
// Looks for decoder-init.onnx or *-init-decoder*.onnx in the model path.
func LoadSeq2SeqDecoderInit(modelPath string, opts *options.Options) (*Model, error) {
	onnxFile, err := findOnnxFile(modelPath, "init-decoder", "decoder-init")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model, opts); err != nil {
		return nil, err
	}

	if err := CreateModelBackend(model, opts); err != nil {
		return nil, err
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadSeq2SeqDecoder loads the decoder model with past_key_values support.
// Looks for decoder.onnx (but not decoder-init.onnx) in the model path.
func LoadSeq2SeqDecoder(modelPath string, opts *options.Options) (*Model, error) {
	onnxFile, err := findDecoderOnnxFile(modelPath)
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model, opts); err != nil {
		return nil, err
	}

	if err := CreateModelBackend(model, opts); err != nil {
		return nil, err
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadSeq2SeqTokenizer loads the tokenizer for seq2seq models.
func LoadSeq2SeqTokenizer(modelPath string, opts *options.Options) (*Tokenizer, error) {
	// Create a temporary model struct to use existing tokenizer loading
	tempModel := &Model{
		Path: modelPath,
	}

	if err := LoadTokenizer(tempModel, opts); err != nil {
		return nil, err
	}

	return tempModel.Tokenizer, nil
}

// LoadSeq2SeqConfig loads the model configuration from config.json.
func LoadSeq2SeqConfig(modelPath string) (*Seq2SeqConfig, error) {
	configPath := fileutil.PathJoinSafe(modelPath, "config.json")

	exists, err := fileutil.FileExists(configPath)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, fmt.Errorf("config.json not found at %s", modelPath)
	}

	configBytes, err := fileutil.ReadFileBytes(configPath)
	if err != nil {
		return nil, err
	}

	var configMap map[string]any
	if err := json.Unmarshal(configBytes, &configMap); err != nil {
		return nil, err
	}

	config := &Seq2SeqConfig{
		EosTokenIDs: make(map[int64]bool),
	}

	// Decoder start token (typically 0 for T5, but T5Gemma2 uses bos_token_id=2)
	if v, ok := configMap["decoder_start_token_id"].(float64); ok {
		config.DecoderStartTokenID = int64(v)
	} else if v, ok := configMap["bos_token_id"].(float64); ok {
		// Fallback to bos_token_id when decoder_start_token_id is null/None
		// This is needed for models like T5Gemma2 that don't set decoder_start_token_id
		config.DecoderStartTokenID = int64(v)
	}

	// EOS token(s)
	if eosRaw, exists := configMap["eos_token_id"]; exists {
		switch v := eosRaw.(type) {
		case []any:
			for _, item := range v {
				if num, ok := item.(float64); ok {
					config.EosTokenIDs[int64(num)] = true
				}
			}
		case float64:
			config.EosTokenIDs[int64(v)] = true
		}
	}

	// Pad token
	if v, ok := configMap["pad_token_id"].(float64); ok {
		config.PadTokenID = int64(v)
	}

	// Number of decoder layers
	if v, ok := configMap["num_decoder_layers"].(float64); ok {
		config.NumDecoderLayers = int(v)
	} else if v, ok := configMap["num_layers"].(float64); ok {
		config.NumDecoderLayers = int(v)
	}

	// Number of attention heads
	if v, ok := configMap["num_heads"].(float64); ok {
		config.NumHeads = int(v)
	} else if v, ok := configMap["num_attention_heads"].(float64); ok {
		config.NumHeads = int(v)
	}

	// Head dimension
	if v, ok := configMap["d_kv"].(float64); ok {
		config.HeadDim = int(v)
	} else if v, ok := configMap["head_dim"].(float64); ok {
		config.HeadDim = int(v)
	}

	// Hidden size (d_model)
	if v, ok := configMap["d_model"].(float64); ok {
		config.DModel = int(v)
	} else if v, ok := configMap["hidden_size"].(float64); ok {
		config.DModel = int(v)
	}

	// Vocab size
	if v, ok := configMap["vocab_size"].(float64); ok {
		config.VocabSize = int(v)
	}

	return config, nil
}

// TokenizeSeq2SeqInputs tokenizes inputs for seq2seq models.
// It uses the existing TokenizeInputs infrastructure and converts to the seq2seq format.
func TokenizeSeq2SeqInputs(inputs []string, tokenizer *Tokenizer, padTokenID int64) (*Seq2SeqTokenized, error) {
	if tokenizer == nil {
		return nil, errors.New("tokenizer is nil")
	}

	// Use the standard batch tokenization
	batch := NewBatch(len(inputs))
	TokenizeInputs(batch, tokenizer, inputs)

	batchSize := len(inputs)
	tokenIDs := make([][]int64, batchSize)
	attentionMask := make([][]int64, batchSize)

	// Calculate max length from actual token IDs, not from MaxSequenceLength
	// (some tokenizers like T5 don't produce attention masks, so MaxSequenceLength may be wrong)
	maxLen := 0
	for i := 0; i < batchSize; i++ {
		if len(batch.Input[i].TokenIDs) > maxLen {
			maxLen = len(batch.Input[i].TokenIDs)
		}
	}

	if maxLen == 0 {
		return nil, errors.New("no tokens produced by tokenizer")
	}

	// Convert from TokenizedInput to int64 format and pad
	for i := 0; i < batchSize; i++ {
		curLen := len(batch.Input[i].TokenIDs)
		padded := make([]int64, maxLen)
		mask := make([]int64, maxLen)

		for j := 0; j < curLen; j++ {
			padded[j] = int64(batch.Input[i].TokenIDs[j])
			mask[j] = 1
		}
		for j := curLen; j < maxLen; j++ {
			padded[j] = padTokenID
			mask[j] = 0
		}

		tokenIDs[i] = padded
		attentionMask[i] = mask
	}

	return &Seq2SeqTokenized{
		TokenIDs:      tokenIDs,
		AttentionMask: attentionMask,
		MaxLength:     maxLen,
	}, nil
}

// findOnnxFile finds an ONNX file containing any of the given patterns.
func findOnnxFile(modelPath string, patterns ...string) (string, error) {
	onnxFiles, err := getOnnxFiles(modelPath)
	if err != nil {
		return "", err
	}

	for _, pattern := range patterns {
		for _, file := range onnxFiles {
			filename := file[1]
			if strings.Contains(strings.ToLower(filename), pattern) {
				return filename, nil
			}
		}
	}

	return "", fmt.Errorf("no ONNX file found matching patterns %v in %s", patterns, modelPath)
}

// PKVSplitResult contains the indices for splitting encoder and decoder past key values.
// This is used by both ORT and GoMLX backends to split PKV outputs from decoder-init.
type PKVSplitResult struct {
	EncoderIndices []int // Indices of encoder (cross-attention) PKV in the allPKV slice
	DecoderIndices []int // Indices of decoder (self-attention) PKV in the allPKV slice
}

// SplitEncoderDecoderPKVIndices analyzes the decoder-init model's output metadata
// to determine which PKV outputs are encoder (cross-attention) vs decoder (self-attention).
// The allPKV slice excludes the first output (logits), so indices are relative to that.
//
// decoder-init outputs are typically ordered as:
//
//	present.0.decoder.key, present.0.decoder.value,
//	present.0.encoder.key, present.0.encoder.value, present.1.decoder.key, ...
//
// Returns a PKVSplitResult with indices for separating encoder and decoder PKV.
// Returns empty result if decoderInitModel is nil or has no output metadata.
func SplitEncoderDecoderPKVIndices(numPKV int, decoderInitModel *Model) PKVSplitResult {
	result := PKVSplitResult{
		EncoderIndices: make([]int, 0, numPKV/2),
		DecoderIndices: make([]int, 0, numPKV/2),
	}

	// Defensive nil check
	if decoderInitModel == nil || decoderInitModel.OutputsMeta == nil {
		return result
	}

	for i := 0; i < numPKV; i++ {
		// Output index in model is i+1 (since we skip logits at index 0)
		outputIdx := i + 1
		if outputIdx >= len(decoderInitModel.OutputsMeta) {
			continue
		}

		outputName := decoderInitModel.OutputsMeta[outputIdx].Name
		if strings.Contains(outputName, ".encoder.") {
			result.EncoderIndices = append(result.EncoderIndices, i)
		} else {
			result.DecoderIndices = append(result.DecoderIndices, i)
		}
	}

	return result
}

// PKVCombineOrder contains the order for combining encoder and decoder PKV for decoder input.
type PKVCombineOrder struct {
	Order []PKVSource // Each element indicates the source for that position
}

// PKVSource indicates whether a PKV tensor comes from encoder or decoder, and its index.
type PKVSource struct {
	IsEncoder bool
	Index     int
}

// GetCombinedPKVOrder analyzes the decoder model's input metadata to determine
// the order for combining encoder and decoder PKV tensors.
// The decoder expects PKV inputs in a specific order matching its input names.
//
// Returns a PKVCombineOrder that maps each output position to its source.
// Returns empty order if decoderModel is nil or has insufficient input metadata.
func GetCombinedPKVOrder(decoderModel *Model) PKVCombineOrder {
	// Defensive nil check
	if decoderModel == nil || decoderModel.InputsMeta == nil || len(decoderModel.InputsMeta) < 2 {
		return PKVCombineOrder{Order: []PKVSource{}}
	}

	// Decoder inputs (after first 2: encoder_attention_mask, input_ids) are PKV tensors
	numPKVInputs := len(decoderModel.InputsMeta) - 2
	result := PKVCombineOrder{
		Order: make([]PKVSource, numPKVInputs),
	}

	encIdx := 0
	decIdx := 0

	// Skip the first 2 inputs (encoder_attention_mask, input_ids)
	for i := 2; i < len(decoderModel.InputsMeta); i++ {
		inputName := decoderModel.InputsMeta[i].Name
		resultIdx := i - 2

		if strings.Contains(inputName, ".encoder.") {
			result.Order[resultIdx] = PKVSource{IsEncoder: true, Index: encIdx}
			encIdx++
		} else {
			result.Order[resultIdx] = PKVSource{IsEncoder: false, Index: decIdx}
			decIdx++
		}
	}

	return result
}

// ApplyRepetitionPenalty modifies logits in-place to penalize previously generated tokens.
// For each token that appears in generatedTokens, its logit is divided by the penalty
// (if the logit is positive) or multiplied by the penalty (if negative).
// This encourages the model to generate diverse tokens rather than repeating.
// A penalty of 1.0 has no effect; values > 1.0 penalize repetition.
func ApplyRepetitionPenalty(logits []float32, generatedTokens []int64, penalty float32, vocabSize int) {
	if penalty == 1.0 {
		return
	}

	// Create a set of tokens to penalize
	tokenSet := make(map[int64]bool, len(generatedTokens))
	for _, tok := range generatedTokens {
		tokenSet[tok] = true
	}

	// Apply penalty to each token that has been generated
	for tok := range tokenSet {
		if tok >= 0 && int(tok) < vocabSize && int(tok) < len(logits) {
			if logits[tok] > 0 {
				logits[tok] /= penalty
			} else {
				logits[tok] *= penalty
			}
		}
	}
}

// ApplyRepetitionPenaltyBatch applies repetition penalty to batched logits.
// logits is a flat array of shape [batchSize, vocabSize].
// generatedTokens[i] contains the tokens generated so far for batch item i.
func ApplyRepetitionPenaltyBatch(logits []float32, generatedTokens [][]int64, penalty float32, batchSize, vocabSize int) {
	if penalty == 1.0 {
		return
	}

	for b := 0; b < batchSize; b++ {
		offset := b * vocabSize
		batchLogits := logits[offset : offset+vocabSize]
		ApplyRepetitionPenalty(batchLogits, generatedTokens[b], penalty, vocabSize)
	}
}

// findDecoderOnnxFile finds the decoder ONNX file (not the init decoder).
func findDecoderOnnxFile(modelPath string) (string, error) {
	onnxFiles, err := getOnnxFiles(modelPath)
	if err != nil {
		return "", err
	}

	for _, file := range onnxFiles {
		filename := strings.ToLower(file[1])
		// Match "decoder" but not "init-decoder" or "decoder-init"
		if strings.Contains(filename, "decoder") &&
			!strings.Contains(filename, "init") {
			return file[1], nil
		}
	}

	return "", fmt.Errorf("no decoder ONNX file found in %s", modelPath)
}
