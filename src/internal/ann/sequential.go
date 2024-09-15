package ann

import (
	"math"
	"math/rand"
)

type SequentialANN struct {
	inputSize    int
	hiddenSize   int
	hiddenLayer  [][]float64
	outputWeight []float64
	outputBias   float64
	learningRate float64
	epochs       int
}

func NewSequentialANN(
	inputSize, hiddenSize int,
	learningRate float64,
	epochs int,
) *SequentialANN {
	ann := &SequentialANN{
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

func (ann *SequentialANN) Train(data [][]float64) {
	for epoch := 0; epoch < ann.epochs; epoch++ {
		for _, sample := range data {
			features := sample[:len(sample)-1]
			label := sample[len(sample)-1]

			// Forward pass
			hiddenOutputs := make([]float64, ann.hiddenSize)
			for i := 0; i < ann.hiddenSize; i++ {
				sum := ann.hiddenLayer[i][0] // bias
				for j, feature := range features {
					sum += ann.hiddenLayer[i][j+1] * feature
				}
				hiddenOutputs[i] = sigmoid(sum)
			}

			finalOutput := ann.outputBias
			for i, hiddenOutput := range hiddenOutputs {
				finalOutput += ann.outputWeight[i] * hiddenOutput
			}
			finalOutput = sigmoid(finalOutput)

			// Backpropagation
			outputDelta := (label - finalOutput) * sigmoidDerivative(finalOutput)

			for i := range ann.outputWeight {
				ann.outputWeight[i] += ann.learningRate * outputDelta * hiddenOutputs[i]
			}
			ann.outputBias += ann.learningRate * outputDelta

			for i := range ann.hiddenLayer {
				hiddenDelta := outputDelta * ann.outputWeight[i] * sigmoidDerivative(
					hiddenOutputs[i],
				)
				for j := range ann.hiddenLayer[i] {
					if j == 0 {
						ann.hiddenLayer[i][j] += ann.learningRate * hiddenDelta
					} else {
						ann.hiddenLayer[i][j] += ann.learningRate * hiddenDelta * features[j-1]
					}
				}
			}
		}
	}
}

func (ann *SequentialANN) Predict(sample []float64) float64 {
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

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

