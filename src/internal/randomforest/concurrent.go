package randomforest

import (
	"math/rand"
	"sync"

	"concurrente/internal/decisiontree"
)

type ConcurrentRandomForest struct {
	trees       []*decisiontree.ConcurrentDecisionTree
	numTrees    int
	subsetRatio float64
}

func NewConcurrentRandomForest(numTrees int, subsetRatio float64) *ConcurrentRandomForest {
	return &ConcurrentRandomForest{
		trees:       make([]*decisiontree.ConcurrentDecisionTree, numTrees),
		numTrees:    numTrees,
		subsetRatio: subsetRatio,
	}
}

func (rf *ConcurrentRandomForest) Train(data [][]float64) {
	var wg sync.WaitGroup
	wg.Add(rf.numTrees)

	for i := 0; i < rf.numTrees; i++ {
		go func(index int) {
			defer wg.Done()
			bootstrapSample := rf.createBootstrapSample(data)
			tree := decisiontree.NewConcurrentDecisionTree()
			tree.Train(bootstrapSample)
			rf.trees[index] = tree
		}(i)
	}

	wg.Wait()
}

func (rf *ConcurrentRandomForest) Predict(sample []float64) float64 {
	predictions := make([]float64, rf.numTrees)
	var wg sync.WaitGroup
	wg.Add(rf.numTrees)

	for i := range rf.trees {
		go func(index int) {
			defer wg.Done()
			predictions[index] = rf.trees[index].Predict(sample)
		}(i)
	}

	wg.Wait()
	return rf.majorityVote(predictions)
}

func (rf *ConcurrentRandomForest) createBootstrapSample(data [][]float64) [][]float64 {
	sampleSize := int(float64(len(data)) * rf.subsetRatio)
	sample := make([][]float64, sampleSize)
	for i := 0; i < sampleSize; i++ {
		randomIndex := rand.Intn(len(data))
		sample[i] = data[randomIndex]
	}
	return sample
}

func (rf *ConcurrentRandomForest) majorityVote(predictions []float64) float64 {
	sum := 0.0
	for _, pred := range predictions {
		sum += pred
	}
	return sum / float64(len(predictions))
}
