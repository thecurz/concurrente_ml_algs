package randomforest

import (
	"math/rand"

	"concurrente/internal/decisiontree"
)

type SequentialRandomForest struct {
	trees       []*decisiontree.SequentialDecisionTree
	numTrees    int
	subsetRatio float64
}

func NewSequentialRandomForest(numTrees int, subsetRatio float64) *SequentialRandomForest {
	return &SequentialRandomForest{
		trees:       make([]*decisiontree.SequentialDecisionTree, numTrees),
		numTrees:    numTrees,
		subsetRatio: subsetRatio,
	}
}

func (rf *SequentialRandomForest) Train(data [][]float64) {
	for i := 0; i < rf.numTrees; i++ {
		bootstrapSample := rf.createBootstrapSample(data)
		tree := decisiontree.NewSequentialDecisionTree()
		tree.Train(bootstrapSample)
		rf.trees[i] = tree
	}
}

func (rf *SequentialRandomForest) Predict(sample []float64) float64 {
	predictions := make([]float64, rf.numTrees)
	for i, tree := range rf.trees {
		predictions[i] = tree.Predict(sample)
	}
	return rf.majorityVote(predictions)
}

func (rf *SequentialRandomForest) createBootstrapSample(data [][]float64) [][]float64 {
	sampleSize := int(float64(len(data)) * rf.subsetRatio)
	sample := make([][]float64, sampleSize)
	for i := 0; i < sampleSize; i++ {
		randomIndex := rand.Intn(len(data))
		sample[i] = data[randomIndex]
	}
	return sample
}

func (rf *SequentialRandomForest) majorityVote(predictions []float64) float64 {
	sum := 0.0
	for _, pred := range predictions {
		sum += pred
	}
	return sum / float64(len(predictions))
}
