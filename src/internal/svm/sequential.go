package svm

type SequentialSVM struct {
	weights      []float64
	bias         float64
	learningRate float64
	lambda       float64
	epochs       int
}

func NewSequentialSVM(features int, learningRate, lambda float64, epochs int) *SequentialSVM {
	return &SequentialSVM{
		weights:      make([]float64, features),
		bias:         0,
		learningRate: learningRate,
		lambda:       lambda,
		epochs:       epochs,
	}
}

func (svm *SequentialSVM) Train(data [][]float64) {
	for epoch := 0; epoch < svm.epochs; epoch++ {
		for _, sample := range data {
			features := sample[:len(sample)-1]
			label := sample[len(sample)-1]
			prediction := svm.predict(features)

			// Hinge loss gradient
			if label*prediction < 1 {
				svm.updateWeights(features, label)
			}

			// L2 regularization
			for i := range svm.weights {
				svm.weights[i] -= svm.learningRate * svm.lambda * svm.weights[i]
			}
		}
	}
}

func (svm *SequentialSVM) predict(features []float64) float64 {
	sum := svm.bias
	for i, feature := range features {
		sum += svm.weights[i] * feature
	}
	return sum
}

func (svm *SequentialSVM) updateWeights(features []float64, label float64) {
	for i, feature := range features {
		svm.weights[i] += svm.learningRate * (label * feature)
	}
	svm.bias += svm.learningRate * label
}

func (svm *SequentialSVM) Predict(sample []float64) float64 {
	prediction := svm.predict(sample)
	if prediction >= 0 {
		return 1
	}
	return 0
}
