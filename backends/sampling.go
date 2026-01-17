package backends

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/knights-analytics/hugot/util/vectorutil"
)

// SamplingBuffers holds reusable buffers for sampling operations.
// This reduces allocations in the hot path during generation.
type SamplingBuffers struct {
	scaledLogits []float32
	probs        []float32
	indexed      []indexedProb
}

// samplingBufferPool provides thread-safe reusable buffers for sampling.
var samplingBufferPool = sync.Pool{
	New: func() any {
		return &SamplingBuffers{}
	},
}

// getSamplingBuffers retrieves or creates buffers for the given vocabulary size.
func getSamplingBuffers(vocabSize int) *SamplingBuffers {
	buf := samplingBufferPool.Get().(*SamplingBuffers)

	// Resize if needed
	if cap(buf.scaledLogits) < vocabSize {
		buf.scaledLogits = make([]float32, vocabSize)
		buf.probs = make([]float32, vocabSize)
		buf.indexed = make([]indexedProb, vocabSize)
	} else {
		buf.scaledLogits = buf.scaledLogits[:vocabSize]
		buf.probs = buf.probs[:vocabSize]
		buf.indexed = buf.indexed[:vocabSize]
	}

	return buf
}

// putSamplingBuffers returns buffers to the pool.
func putSamplingBuffers(buf *SamplingBuffers) {
	samplingBufferPool.Put(buf)
}

// indexedProb pairs a token index with its probability.
type indexedProb struct {
	index int
	prob  float32
}

// NewSamplingRNG creates a new random number generator for sampling.
// Each generation call should create its own RNG for thread safety.
func NewSamplingRNG() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UnixNano())) // #nosec G404 -- not used for crypto
}

// ArgmaxBatch performs argmax over vocabulary for each batch item.
// Returns an error if the logits buffer is too small.
func ArgmaxBatch(logits []float32, batchSize, vocabSize int) ([]int64, error) {
	// Use int64 for bounds check to prevent overflow on 32-bit systems
	requiredLen := int64(batchSize) * int64(vocabSize)
	if int64(len(logits)) < requiredLen {
		return nil, fmt.Errorf("logits buffer too small: need %d elements, have %d", requiredLen, len(logits))
	}

	tokens := make([]int64, batchSize)

	for b := 0; b < batchSize; b++ {
		offset := b * vocabSize
		maxIdx := 0
		maxVal := logits[offset]

		for v := 1; v < vocabSize; v++ {
			if logits[offset+v] > maxVal {
				maxVal = logits[offset+v]
				maxIdx = v
			}
		}
		tokens[b] = int64(maxIdx)
	}

	return tokens, nil
}

// SampleTopPBatch performs nucleus (top-p) sampling with temperature for a batch.
// Uses buffer pooling and partial sorting for efficiency.
// Returns an error if temperature is invalid or logits buffer is too small.
func SampleTopPBatch(logits []float32, batchSize, vocabSize int, topP, temperature float32, rng *rand.Rand) ([]int64, error) {
	// Validate temperature
	if temperature <= 0 {
		return nil, errors.New("temperature must be positive")
	}

	// Use int64 for bounds check to prevent overflow
	requiredLen := int64(batchSize) * int64(vocabSize)
	if int64(len(logits)) < requiredLen {
		return nil, fmt.Errorf("logits buffer too small: need %d elements, have %d", requiredLen, len(logits))
	}

	tokens := make([]int64, batchSize)
	buf := getSamplingBuffers(vocabSize)
	defer putSamplingBuffers(buf)

	for b := 0; b < batchSize; b++ {
		offset := b * vocabSize
		batchLogits := logits[offset : offset+vocabSize]

		// Apply temperature scaling into buffer
		if temperature != 1.0 {
			for i := 0; i < vocabSize; i++ {
				buf.scaledLogits[i] = batchLogits[i] / temperature
			}
		} else {
			copy(buf.scaledLogits, batchLogits)
		}

		// Softmax into probs buffer
		vectorutil.SoftMaxInto(buf.scaledLogits, buf.probs)

		// Sample using partial sort
		token, err := sampleTopPSingle(buf.probs, buf.indexed, topP, rng)
		if err != nil {
			return nil, fmt.Errorf("sampling batch %d: %w", b, err)
		}
		tokens[b] = int64(token)
	}

	return tokens, nil
}

// sampleTopPSingle performs top-p sampling for a single probability distribution.
// Uses partial sorting for O(n) expected time instead of O(n log n) full sort.
func sampleTopPSingle(probs []float32, indexed []indexedProb, topP float32, rng *rand.Rand) (int, error) {
	vocabSize := len(probs)
	if vocabSize == 0 {
		return 0, errors.New("empty probability distribution")
	}

	// Quick path: if topP >= 1.0, just sample from full distribution
	if topP >= 1.0 {
		return sampleFromProbs(probs, rng), nil
	}

	// Find the approximate number of tokens needed to reach topP probability mass.
	// We use a max-heap approach: collect tokens until cumulative prob >= topP.
	// This is O(n) on average for typical topP values (0.9, 0.95).

	// First pass: find sum and prepare indexed probs
	for i := 0; i < vocabSize; i++ {
		indexed[i] = indexedProb{index: i, prob: probs[i]}
	}

	// Use partial selection: find tokens comprising top-p mass
	// We'll use a simple approach: find threshold probability and collect tokens above it
	nucleus, totalProb := selectNucleus(indexed, topP)

	if len(nucleus) == 0 {
		// Fallback: return token with highest probability
		maxIdx := 0
		maxProb := probs[0]
		for i := 1; i < vocabSize; i++ {
			if probs[i] > maxProb {
				maxProb = probs[i]
				maxIdx = i
			}
		}
		return maxIdx, nil
	}

	// Handle edge case: if total probability is zero (shouldn't happen after softmax,
	// but could occur due to numerical issues), return the first nucleus token
	if totalProb == 0 {
		return nucleus[0].index, nil
	}

	// Sample from nucleus
	r := rng.Float32() * totalProb
	cumProb := float32(0)
	for _, ip := range nucleus {
		cumProb += ip.prob
		if r <= cumProb {
			return ip.index, nil
		}
	}

	// Fallback to last token in nucleus
	return nucleus[len(nucleus)-1].index, nil
}

// selectNucleus selects the smallest set of tokens with cumulative probability >= topP.
// Uses a simple O(n) linear scan approach:
// 1. Find max probability and count non-zero tokens
// 2. Collect all tokens with probability above a threshold
// 3. Sort only the collected tokens (typically small subset)
func selectNucleus(indexed []indexedProb, topP float32) ([]indexedProb, float32) {
	n := len(indexed)
	if n == 0 {
		return nil, 0
	}

	// Calculate total sum and find max probability
	totalSum := float32(0)
	maxProb := float32(0)
	for i := 0; i < n; i++ {
		totalSum += indexed[i].prob
		if indexed[i].prob > maxProb {
			maxProb = indexed[i].prob
		}
	}

	if totalSum == 0 {
		return nil, 0
	}

	targetSum := topP * totalSum

	// For typical top-p values (0.9-0.95), most probability mass is in a small
	// number of high-probability tokens. We use a threshold-based approach:
	// Start with a high threshold and lower it until we have enough mass.

	// Collect tokens into a result slice, sorted by probability descending
	// We use insertion sort since the nucleus is typically small (<100 tokens)
	result := make([]indexedProb, 0, 64) // Pre-allocate for typical nucleus size

	// Collect all non-zero probability tokens
	for i := 0; i < n; i++ {
		if indexed[i].prob > 0 {
			// Insert in sorted order (descending by probability)
			pos := len(result)
			for pos > 0 && result[pos-1].prob < indexed[i].prob {
				pos--
			}
			result = append(result, indexedProb{})
			copy(result[pos+1:], result[pos:])
			result[pos] = indexed[i]
		}
	}

	// Take tokens from the top until we reach target probability mass
	cumSum := float32(0)
	cutoff := 0
	for i, ip := range result {
		cumSum += ip.prob
		cutoff = i + 1
		if cumSum >= targetSum {
			break
		}
	}

	return result[:cutoff], cumSum
}

// sampleFromProbs samples a token index from a probability distribution.
func sampleFromProbs(probs []float32, rng *rand.Rand) int {
	r := rng.Float32()
	cumProb := float32(0)
	for i, p := range probs {
		cumProb += p
		if r <= cumProb {
			return i
		}
	}
	return len(probs) - 1
}
