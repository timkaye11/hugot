package pipelines

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/util"
)

// GLiNERPipeline implements zero-shot Named Entity Recognition using GLiNER models.
// GLiNER (Generalist and Lightweight model for Named Entity Recognition) can extract
// any entity types without retraining - just specify the entity labels at inference time.
type GLiNERPipeline struct {
	*pipelineBackends.BasePipeline
	MaxWidth   int      // Maximum span width in words
	Labels     []string // Entity labels to recognize
	Threshold  float32  // Score threshold for entity detection
	FlatNER    bool     // If true, don't allow nested entities
	MultiLabel bool     // If true, allow multiple labels per span
}

// GLiNEREntity represents a recognized named entity
type GLiNEREntity struct {
	Text  string  // The text span of the entity
	Label string  // The entity type/label
	Start int     // Character offset where entity begins
	End   int     // Character offset where entity ends
	Score float32 // Model confidence score (0.0-1.0)
}

// GLiNEROutput holds the output of GLiNER inference
type GLiNEROutput struct {
	Entities [][]GLiNEREntity
}

func (o *GLiNEROutput) GetOutput() []any {
	out := make([]any, len(o.Entities))
	for i, entities := range o.Entities {
		out[i] = any(entities)
	}
	return out
}

// GLiNERBatch extends PipelineBatch with GLiNER-specific data
type GLiNERBatch struct {
	*pipelineBackends.PipelineBatch
	WordsMask    [][]int64   // Word boundary mask for each input
	TextLengths  [][]int64   // Number of words in each input
	SpanIdx      [][][]int64 // Span indices [batch][num_spans][2]
	SpanMask     [][]int64   // Valid spans mask
	NumSpans     int         // Number of spans per input
	WordsToChars [][][2]int  // Mapping from word index to character offsets [batch][word][start,end]
	OriginalText []string    // Original input texts
}

// Pipeline options

// WithGLiNERLabels sets the entity labels to recognize
func WithGLiNERLabels(labels []string) pipelineBackends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		p.Labels = labels
		return nil
	}
}

// WithGLiNERMaxWidth sets the maximum span width in words
func WithGLiNERMaxWidth(maxWidth int) pipelineBackends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		if maxWidth <= 0 {
			return errors.New("maxWidth must be positive")
		}
		p.MaxWidth = maxWidth
		return nil
	}
}

// WithGLiNERThreshold sets the score threshold for entity detection
func WithGLiNERThreshold(threshold float32) pipelineBackends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		if threshold < 0 || threshold > 1 {
			return errors.New("threshold must be between 0 and 1")
		}
		p.Threshold = threshold
		return nil
	}
}

// WithGLiNERFlatNER enables flat NER mode (no nested entities)
func WithGLiNERFlatNER() pipelineBackends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		p.FlatNER = true
		return nil
	}
}

// WithGLiNERMultiLabel enables multi-label mode
func WithGLiNERMultiLabel() pipelineBackends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		p.MultiLabel = true
		return nil
	}
}

// NewGLiNERPipeline creates a new GLiNER pipeline
func NewGLiNERPipeline(config pipelineBackends.PipelineConfig[*GLiNERPipeline], s *options.Options, model *pipelineBackends.Model) (*GLiNERPipeline, error) {
	basePipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &GLiNERPipeline{
		BasePipeline: basePipeline,
		MaxWidth:     12, // default max span width
		Labels:       []string{"person", "organization", "location"},
		Threshold:    0.5,
		FlatNER:      true,
		MultiLabel:   false,
	}

	// Apply options
	for _, o := range config.Options {
		if err := o(pipeline); err != nil {
			return nil, err
		}
	}

	// GLiNER needs offsets and special token masks to detect word boundaries
	pipelineBackends.AllInputTokens(pipeline.BasePipeline)

	// Validate the pipeline
	if err := pipeline.Validate(); err != nil {
		return nil, err
	}

	return pipeline, nil
}

// GetModel returns the underlying model
func (p *GLiNERPipeline) GetModel() *pipelineBackends.Model {
	return p.Model
}

// GetMetadata returns pipeline metadata
func (p *GLiNERPipeline) GetMetadata() pipelineBackends.PipelineMetadata {
	return pipelineBackends.PipelineMetadata{
		OutputsInfo: []pipelineBackends.OutputInfo{
			{
				Name:       p.Model.OutputsMeta[0].Name,
				Dimensions: p.Model.OutputsMeta[0].Dimensions,
			},
		},
	}
}

// GetStats returns runtime statistics
func (p *GLiNERPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for GLiNER pipeline: %s", p.PipelineName),
		fmt.Sprintf("Tokenizer: Total time=%s, Execution count=%d, Average time=%s",
			time.Duration(p.Model.Tokenizer.TokenizerTimings.TotalNS),
			p.Model.Tokenizer.TokenizerTimings.NumCalls,
			time.Duration(float64(p.Model.Tokenizer.TokenizerTimings.TotalNS)/math.Max(1, float64(p.Model.Tokenizer.TokenizerTimings.NumCalls)))),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

// Validate checks that the pipeline configuration is valid
func (p *GLiNERPipeline) Validate() error {
	var validationErrors []error

	if p.Model.Tokenizer == nil {
		validationErrors = append(validationErrors, errors.New("GLiNER pipeline requires a tokenizer"))
	}

	if len(p.Labels) == 0 {
		validationErrors = append(validationErrors, errors.New("GLiNER pipeline requires at least one label"))
	}

	// Verify model has expected inputs
	expectedInputs := map[string]bool{
		"input_ids":      false,
		"attention_mask": false,
		"words_mask":     false,
		"text_lengths":   false,
		"span_idx":       false,
		"span_mask":      false,
	}
	for _, meta := range p.Model.InputsMeta {
		if _, ok := expectedInputs[meta.Name]; ok {
			expectedInputs[meta.Name] = true
		}
	}
	for name, found := range expectedInputs {
		if !found {
			validationErrors = append(validationErrors, fmt.Errorf("GLiNER model missing required input: %s", name))
		}
	}

	return errors.Join(validationErrors...)
}

// Run executes the pipeline on input texts
func (p *GLiNERPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline executes the pipeline and returns the concrete output type
func (p *GLiNERPipeline) RunPipeline(inputs []string) (*GLiNEROutput, error) {
	return p.RunPipelineWithLabels(inputs, p.Labels)
}

// RunPipelineWithLabels executes the pipeline with custom labels (zero-shot NER)
func (p *GLiNERPipeline) RunPipelineWithLabels(inputs []string, labels []string) (*GLiNEROutput, error) {
	if len(inputs) == 0 {
		return &GLiNEROutput{Entities: [][]GLiNEREntity{}}, nil
	}

	var runErrors []error
	batch := p.prepareGLiNERBatch(len(inputs))
	defer func() {
		if batch.PipelineBatch != nil {
			runErrors = append(runErrors, batch.Destroy())
		}
	}()

	// Preprocess
	if err := p.Preprocess(batch, inputs, labels); err != nil {
		return nil, err
	}

	// Forward pass
	if err := p.Forward(batch); err != nil {
		return nil, err
	}

	// Postprocess
	result, err := p.Postprocess(batch, labels)
	if err != nil {
		return nil, err
	}

	return result, errors.Join(runErrors...)
}

func (p *GLiNERPipeline) prepareGLiNERBatch(size int) *GLiNERBatch {
	return &GLiNERBatch{
		PipelineBatch: pipelineBackends.NewBatch(size),
		OriginalText:  make([]string, size),
	}
}

// Preprocess prepares the batch for inference
func (p *GLiNERPipeline) Preprocess(batch *GLiNERBatch, inputs []string, labels []string) error {
	start := time.Now()

	// GLiNER requires labels to be prepended to the input text
	// Format: "<<ENT>> label1 <<ENT>> label2 <<SEP>> text"
	labelPrefix := buildGLiNERLabelPrefix(labels)
	prefixedTexts := make([]string, len(inputs))
	for i, text := range inputs {
		prefixedTexts[i] = labelPrefix + " " + text
		batch.OriginalText[i] = text
	}

	// Tokenize the prefixed texts
	pipelineBackends.TokenizeInputs(batch.PipelineBatch, p.Model.Tokenizer, prefixedTexts)

	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))

	// Build GLiNER-specific inputs
	if err := p.buildGLiNERInputs(batch, labels); err != nil {
		return err
	}

	return nil
}

// buildGLiNERInputs constructs the GLiNER-specific input tensors
func (p *GLiNERPipeline) buildGLiNERInputs(batch *GLiNERBatch, labels []string) error {
	batchSize := batch.Size
	maxSeqLen := batch.MaxSequenceLength

	// Initialize arrays
	batch.WordsMask = make([][]int64, batchSize)
	batch.TextLengths = make([][]int64, batchSize)
	batch.WordsToChars = make([][][2]int, batchSize)

	// Calculate the character length of the label prefix to skip
	labelPrefixCharLen := uint(calculateLabelPrefixLength(labels))

	for i, input := range batch.Input {
		wordsMask := make([]int64, maxSeqLen)
		wordsToChars := [][2]int{}

		// Track word boundaries
		wordCount := int64(0)
		inTextRegion := false

		for j, offset := range input.Offsets {
			// Skip special tokens (CLS, SEP, PAD)
			if input.SpecialTokensMask[j] > 0 {
				continue
			}

			tokenStart := offset[0]
			tokenEnd := offset[1]

			// Skip tokens that are part of the label prefix
			if tokenEnd <= labelPrefixCharLen {
				continue
			}

			// We've entered the text region (past the label prefix)
			inTextRegion = true

			// Adjust offsets to be relative to the original text (subtract prefix length)
			adjustedStart := tokenStart - labelPrefixCharLen
			adjustedEnd := tokenEnd - labelPrefixCharLen

			// Detect word boundaries using SentencePiece convention:
			// Tokens starting with ▁ (U+2581) indicate a new word
			token := ""
			if j < len(input.Tokens) {
				token = input.Tokens[j]
			}

			isNewWord := false
			if inTextRegion && wordCount == 0 {
				// First token in text region is always a new word
				isNewWord = true
			} else if strings.HasPrefix(token, "▁") || strings.HasPrefix(token, " ") {
				// Token starts with word boundary marker
				isNewWord = true
			}

			if isNewWord {
				wordCount++
				wordsMask[j] = wordCount
				// Calculate start offset, accounting for SentencePiece space markers
				// For the first word, the ▁ represents the space in the prefix, not in the original text
				// For subsequent words, the ▁ represents an actual space in the original text
				startOffset := int(adjustedStart)
				if wordCount > 1 && (strings.HasPrefix(token, "▁") || strings.HasPrefix(token, " ")) {
					startOffset++ // Skip the space for words after the first
				}
				wordsToChars = append(wordsToChars, [2]int{startOffset, int(adjustedEnd)})
			} else {
				// Continuation of previous word (subword token)
				wordsMask[j] = wordCount
				if len(wordsToChars) > 0 {
					wordsToChars[len(wordsToChars)-1][1] = int(adjustedEnd)
				}
			}
		}

		batch.WordsMask[i] = wordsMask
		batch.TextLengths[i] = []int64{wordCount}
		batch.WordsToChars[i] = wordsToChars
	}

	// Generate spans for each input
	if err := p.generateSpans(batch); err != nil {
		return err
	}

	// Create ORT tensors
	return p.createGLiNERTensors(batch)
}

// generateSpans creates all valid spans up to MaxWidth
// GLiNER expects spans organized as: for each word position, max_width spans (one per width)
// Total spans = num_words × max_width, organized as [pos0_w1, pos0_w2, ..., pos0_wN, pos1_w1, ...]
func (p *GLiNERPipeline) generateSpans(batch *GLiNERBatch) error {
	batchSize := batch.Size

	// Find max words across batch
	maxWords := 0
	for i := 0; i < batchSize; i++ {
		numWords := int(batch.TextLengths[i][0])
		if numWords > maxWords {
			maxWords = numWords
		}
	}

	// GLiNER expects exactly num_words × max_width spans
	// The model internally reshapes to [batch, num_words, max_width, hidden]
	numSpans := maxWords * p.MaxWidth
	if numSpans == 0 {
		numSpans = p.MaxWidth // At least max_width span slots
	}

	batch.NumSpans = numSpans
	batch.SpanIdx = make([][][]int64, batchSize)
	batch.SpanMask = make([][]int64, batchSize)

	for i := 0; i < batchSize; i++ {
		numWords := int(batch.TextLengths[i][0])
		spanIdx := make([][]int64, numSpans)
		spanMask := make([]int64, numSpans)

		// For each word position, generate max_width spans
		for pos := 0; pos < maxWords; pos++ {
			for width := 1; width <= p.MaxWidth; width++ {
				spanIndex := pos*p.MaxWidth + (width - 1)
				endPos := pos + width - 1 // inclusive end

				// Check if span is valid (within text bounds)
				if pos < numWords && endPos < numWords {
					spanIdx[spanIndex] = []int64{int64(pos), int64(endPos)}
					spanMask[spanIndex] = 1
				} else {
					// Invalid span (extends beyond text or position is padding)
					spanIdx[spanIndex] = []int64{0, 0}
					spanMask[spanIndex] = 0
				}
			}
		}

		batch.SpanIdx[i] = spanIdx
		batch.SpanMask[i] = spanMask
	}

	return nil
}

// createGLiNERTensors creates input tensors for GLiNER based on runtime
func (p *GLiNERPipeline) createGLiNERTensors(batch *GLiNERBatch) error {
	// GLiNER requires custom input tensors - don't use the standard CreateInputTensors
	// as it doesn't know how to handle GLiNER-specific inputs
	switch p.Runtime {
	case "ORT":
		return createGLiNERTensorsORT(batch, p.Model)
	case "GO", "XLA":
		return createGLiNERTensorsGoMLX(batch, p.Model)
	default:
		return fmt.Errorf("unsupported runtime for GLiNER: %s", p.Runtime)
	}
}

// Forward performs the forward inference pass
func (p *GLiNERPipeline) Forward(batch *GLiNERBatch) error {
	start := time.Now()
	var err error
	switch p.Runtime {
	case "ORT":
		err = runGLiNERSessionOnBatchORT(batch, p.BasePipeline)
	case "GO", "XLA":
		err = runGLiNERSessionOnBatchGoMLX(batch, p.BasePipeline)
	default:
		return fmt.Errorf("unsupported runtime for GLiNER: %s", p.Runtime)
	}
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

// Postprocess converts model output to entities
func (p *GLiNERPipeline) Postprocess(batch *GLiNERBatch, labels []string) (*GLiNEROutput, error) {
	if batch.Size == 0 {
		return &GLiNEROutput{}, nil
	}

	output := batch.OutputValues[0]
	batchSize := batch.Size

	// GLiNER output shape: [batch_size, num_words, num_spans, num_labels]
	// After sigmoid, values represent probability of each label for each span
	var logits [][][][]float32
	switch v := output.(type) {
	case [][][][]float32:
		logits = v
	default:
		return nil, fmt.Errorf("expected 4D output, got type %T", output)
	}

	result := &GLiNEROutput{
		Entities: make([][]GLiNEREntity, batchSize),
	}

	// Process each input in the batch
	for i := 0; i < batchSize; i++ {
		entities := p.extractEntities(
			batch.OriginalText[i],
			logits[i],
			batch.SpanIdx[i],
			batch.SpanMask[i],
			batch.WordsToChars[i],
			labels,
		)
		result.Entities[i] = entities
	}

	return result, nil
}

// extractEntities extracts entities from the logits for a single input
// logits shape: [num_words][max_width][num_labels]
// For each word position, there are max_width span options (width 1 to max_width)
func (p *GLiNERPipeline) extractEntities(
	text string,
	logits [][][]float32, // [num_words][max_width][num_labels]
	spanIdx [][]int64,
	spanMask []int64,
	wordsToChars [][2]int,
	labels []string,
) []GLiNEREntity {
	var candidates []GLiNEREntity

	numLabels := len(labels)
	numWords := len(wordsToChars)

	// Iterate over all spans organized as [word_position × max_width]
	for spanI := 0; spanI < len(spanIdx); spanI++ {
		if spanMask[spanI] == 0 {
			continue
		}

		startWord := int(spanIdx[spanI][0])
		endWord := int(spanIdx[spanI][1])

		// Get character offsets
		if startWord >= numWords || endWord >= numWords {
			continue
		}

		charStart := wordsToChars[startWord][0]
		charEnd := wordsToChars[endWord][1]

		if charStart >= len(text) || charEnd > len(text) || charStart >= charEnd {
			continue
		}

		entityText := text[charStart:charEnd]

		// Calculate indices into logits: [word_position][width_index][label]
		// width = endWord - startWord + 1, so width_index = width - 1 = endWord - startWord
		widthIndex := endWord - startWord

		if startWord >= len(logits) {
			continue
		}
		if widthIndex >= len(logits[startWord]) {
			continue
		}

		spanLogits := logits[startWord][widthIndex]
		if len(spanLogits) < numLabels {
			continue
		}

		// Apply sigmoid and check threshold
		for labelIdx := 0; labelIdx < numLabels; labelIdx++ {
			score := sigmoid(spanLogits[labelIdx])
			if score >= p.Threshold {
				candidates = append(candidates, GLiNEREntity{
					Text:  entityText,
					Label: labels[labelIdx],
					Start: charStart,
					End:   charEnd,
					Score: score,
				})

				// If not multi-label, only take the highest scoring label
				if !p.MultiLabel {
					break
				}
			}
		}
	}

	// If flat NER, remove nested entities (keep highest scoring)
	if p.FlatNER {
		candidates = removeNestedEntities(candidates)
	}

	// Sort by position
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Start != candidates[j].Start {
			return candidates[i].Start < candidates[j].Start
		}
		return candidates[i].End < candidates[j].End
	})

	return candidates
}

// Helper functions

// GLiNER special tokens
const (
	glinerEntityToken = "<<ENT>>"
	glinerSepToken    = "<<SEP>>"
)

func buildGLiNERLabelPrefix(labels []string) string {
	var sb strings.Builder
	for _, label := range labels {
		sb.WriteString(glinerEntityToken)
		sb.WriteString(" ")
		sb.WriteString(label)
		sb.WriteString(" ")
	}
	sb.WriteString(glinerSepToken)
	return sb.String()
}

func calculateLabelPrefixLength(labels []string) int {
	// Calculate the character length of the label prefix "<<ENT>> label1 <<ENT>> label2 <<SEP>>"
	// This is used to find where the actual text starts in character offsets
	length := 0
	for _, label := range labels {
		// "<<ENT>> " + label + " "
		length += len(glinerEntityToken) + 1 + len(label) + 1
	}
	// "<<SEP>>" + space before text
	length += len(glinerSepToken) + 1
	return length
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(-float64(x))))
}

func removeNestedEntities(entities []GLiNEREntity) []GLiNEREntity {
	if len(entities) <= 1 {
		return entities
	}

	// Sort by score descending
	sort.Slice(entities, func(i, j int) bool {
		return entities[i].Score > entities[j].Score
	})

	var result []GLiNEREntity
	for _, entity := range entities {
		overlaps := false
		for _, kept := range result {
			// Check if entity overlaps with any kept entity
			if entity.Start < kept.End && entity.End > kept.Start {
				overlaps = true
				break
			}
		}
		if !overlaps {
			result = append(result, entity)
		}
	}

	return result
}

// softMax applies softmax normalization - using util package
var _ = util.SoftMax // ensure import is used
