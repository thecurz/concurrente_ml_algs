package ann

import (
	"math/rand"
	"sync"
)

type ConcurrentANN struct {
	inputSize    int
	hiddenSize   int
	hiddenLayer  [][]float64
	outputWeight []float64
	outputBias   float64
	learningRate float64
	epochs       int
}

func NewConcurrentANN(
	inputSize, hiddenSize int,
	learningRate float64,
	epochs int,
) *ConcurrentANN {
	ann := &ConcurrentANN{
		inputSize:    inputSize,
		hiddenSize:   hiddenSize,
		hiddenLayer:  make([][]float64, hiddenSize),
		outputWeight: make([]float64, hiddenSize),
		learningRate: learningRate,
		epochs:       epochs,
	}

	// Initialize weights with small random values
	for i := range ann.hiddenLayer {
		ann.hiddenLayer[i] = make([]float64, inputSize+1) // +1 for bias
		for j := range ann.hiddenLayer[i] {
			ann.hiddenLayer[i][j] = rand.Float64()*0.2 - 0.1
		}
	}
	for i := range ann.outputWeight {
		ann.outputWeight[i] = rand.Float64()*0.2 - 0.1
	}
	ann.outputBias = rand.Float64()*0.2 - 0.1

	return ann
}

func (ann *ConcurrentANN) Train(data [][]float64) {
	var wg sync.WaitGroup
	numGoroutines := 4 // Adjust based on your system's capabilities

	for epoch := 0; epoch < ann.epochs; epoch++ {
		wg.Add(numGoroutines)
		chunkSize := len(data) / numGoroutines

		for i := 0; i < numGoroutines; i++ {
			start := i * chunkSize
			end := start + chunkSize
			if i == numGoroutines-1 {
				end = len(data)
			}

			go func(start, end int) {
				defer wg.Done()
				ann.trainChunk(data[start:end])
			}(start, end)
		}

		wg.Wait()
	}
}

func (ann *ConcurrentANN) trainChunk(chunk [][]float64) {
	localHiddenLayer := make([][]float64, len(ann.hiddenLayer))
	for i := range localHiddenLayer {
		localHiddenLayer[i] = make([]float64, len(ann.hiddenLayer[i]))
		copy(localHiddenLayer[i], ann.hiddenLayer[i])
	}
	localOutputWeight := make([]float64, len(ann.outputWeight))
	copy(localOutputWeight, ann.outputWeight)
	localOutputBias := ann.outputBias

	for _, sample := range chunk {
		features := sample[:len(sample)-1]
		label := sample[len(sample)-1]

		// Forward pass
		hiddenOutputs := make([]float64, ann.hiddenSize)
		for i := 0; i < ann.hiddenSize; i++ {
			sum := localHiddenLayer[i][0] // bias
			for j, feature := range features {
				sum += localHiddenLayer[i][j+1] * feature
			}
			hiddenOutputs[i] = sigmoid(sum)
		}

		finalOutput := localOutputBias
		for i, hiddenOutput := range hiddenOutputs {
			finalOutput += localOutputWeight[i] * hiddenOutput
		}
		finalOutput = sigmoid(finalOutput)

		// Backpropagation
		outputDelta := (label - finalOutput) * sigmoidDerivative(finalOutput)

		for i := range localOutputWeight {
			localOutputWeight[i] += ann.learningRate * outputDelta * hiddenOutputs[i]
		}
		localOutputBias += ann.learningRate * outputDelta

		for i := range localHiddenLayer {
			hiddenDelta := outputDelta * localOutputWeight[i] * sigmoidDerivative(hiddenOutputs[i])
			for j := range localHiddenLayer[i] {
				if j == 0 {
					localHiddenLayer[i][j] += ann.learningRate * hiddenDelta
				} else {
					localHiddenLayer[i][j] += ann.learningRate * hiddenDelta * features[j-1]
				}
			}
		}
	}

	// Update the shared resources once, after processing the entire chunk
	var mutex sync.Mutex
	mutex.Lock()
	defer mutex.Unlock()

	for i := range ann.hiddenLayer {
		for j := range ann.hiddenLayer[i] {
			ann.hiddenLayer[i][j] += localHiddenLayer[i][j]
		}
	}
	for i := range ann.outputWeight {
		ann.outputWeight[i] += localOutputWeight[i]
	}
	ann.outputBias += localOutputBias
}

func (ann *ConcurrentANN) Predict(sample []float64) float64 {
	hiddenOutputs := make([]float64, ann.hiddenSize)
	for i := 0; i < ann.hiddenSize; i++ {
		sum := ann.hiddenLayer[i][0] // bias
		for j, feature := range sample {
			sum += ann.hiddenLayer[i][j+1] * feature
		}
		hiddenOutputs[i] = sigmoid(sum)
	}

	finalOutput := ann.outputBias
	for i, hiddenOutput := range hiddenOutputs {
		finalOutput += ann.outputWeight[i] * hiddenOutput
	}
	finalOutput = sigmoid(finalOutput)

	if finalOutput >= 0.5 {
		return 1
	}
	return 0
}
