package svm

import (
	"sync"
)

type ConcurrentSVM struct {
	weights      []float64
	bias         float64
	learningRate float64
	lambda       float64
	epochs       int
}

func NewConcurrentSVM(features int, learningRate, lambda float64, epochs int) *ConcurrentSVM {
	return &ConcurrentSVM{
		weights:      make([]float64, features),
		bias:         0,
		learningRate: learningRate,
		lambda:       lambda,
		epochs:       epochs,
	}
}

func (svm *ConcurrentSVM) Train(data [][]float64) {
	var wg sync.WaitGroup
	numGoroutines := 4 // Adjust based on your system's capabilities

	for epoch := 0; epoch < svm.epochs; epoch++ {
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
				svm.trainChunk(data[start:end])
			}(start, end)
		}

		wg.Wait()
	}
}

func (svm *ConcurrentSVM) trainChunk(chunk [][]float64) {
	localWeights := make([]float64, len(svm.weights))
	localBias := 0.0

	for _, sample := range chunk {
		features := sample[:len(sample)-1]
		label := sample[len(sample)-1]
		prediction := svm.predict(features)

		// Hinge loss gradient
		if label*prediction < 1 {
			svm.updateLocalWeights(localWeights, &localBias, features, label)
		}

		// L2 regularization
		for i := range localWeights {
			localWeights[i] -= svm.learningRate * svm.lambda * svm.weights[i]
		}
	}

	svm.updateGlobalWeights(localWeights, localBias)
}

func (svm *ConcurrentSVM) predict(features []float64) float64 {
	sum := svm.bias
	for i, feature := range features {
		sum += svm.weights[i] * feature
	}
	return sum
}

func (svm *ConcurrentSVM) updateLocalWeights(
	localWeights []float64,
	localBias *float64,
	features []float64,
	label float64,
) {
	for i, feature := range features {
		localWeights[i] += svm.learningRate * (label * feature)
	}
	*localBias += svm.learningRate * label
}

func (svm *ConcurrentSVM) updateGlobalWeights(localWeights []float64, localBias float64) {
	var mutex sync.Mutex
	mutex.Lock()
	defer mutex.Unlock()

	for i := range svm.weights {
		svm.weights[i] += localWeights[i]
	}
	svm.bias += localBias
}

func (svm *ConcurrentSVM) Predict(sample []float64) float64 {
	prediction := svm.predict(sample)
	if prediction >= 0 {
		return 1
	}
	return 0
}
