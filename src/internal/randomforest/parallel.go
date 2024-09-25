package randomforest

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
)

type ParallelRandomForest struct {
	trees       []*ParallelDecisionTree
	numTrees    int
	subsetRatio float64
	numWorkers  int
}

type ParallelDecisionTree struct {
	root *Node
}

type Node struct {
	Feature    int
	Threshold  float64
	Left       *Node
	Right      *Node
	Prediction float64
}

func NewParallelRandomForest(numTrees int, subsetRatio float64) *ParallelRandomForest {
	return &ParallelRandomForest{
		trees:       make([]*ParallelDecisionTree, numTrees),
		numTrees:    numTrees,
		subsetRatio: subsetRatio,
		numWorkers:  runtime.GOMAXPROCS(0),
	}
}

func (rf *ParallelRandomForest) Train(data [][]float64) {
	var wg sync.WaitGroup
	treeChan := make(chan int, rf.numTrees)

	// Fill the channel with tree indices
	for i := 0; i < rf.numTrees; i++ {
		treeChan <- i
	}
	close(treeChan)

	// Create worker pool
	for i := 0; i < rf.numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for treeIndex := range treeChan {
				bootstrapSample := rf.createBootstrapSample(data)
				tree := &ParallelDecisionTree{}
				tree.Train(bootstrapSample)
				rf.trees[treeIndex] = tree
			}
		}()
	}
	wg.Wait()
}

func (rf *ParallelRandomForest) Predict(sample []float64) float64 {
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

func (rf *ParallelRandomForest) createBootstrapSample(data [][]float64) [][]float64 {
	sampleSize := int(float64(len(data)) * rf.subsetRatio)
	sample := make([][]float64, sampleSize)
	var wg sync.WaitGroup
	wg.Add(sampleSize)

	for i := 0; i < sampleSize; i++ {
		go func(index int) {
			defer wg.Done()
			randomIndex := rand.Intn(len(data))
			sample[index] = make([]float64, len(data[randomIndex]))
			copy(sample[index], data[randomIndex])
		}(i)
	}
	wg.Wait()

	return sample
}

func (rf *ParallelRandomForest) majorityVote(predictions []float64) float64 {
	sum := 0.0
	for _, pred := range predictions {
		sum += pred
	}
	return sum / float64(len(predictions))
}

func (tree *ParallelDecisionTree) Train(data [][]float64) {
	tree.root = tree.buildTree(data, 0)
}

func (tree *ParallelDecisionTree) buildTree(data [][]float64, depth int) *Node {
	if len(data) == 0 {
		return &Node{Prediction: 0}
	}

	if depth >= 10 || len(data) < 2 {
		return &Node{Prediction: tree.calculatePrediction(data)}
	}

	bestFeature, bestThreshold := tree.findBestSplit(data)

	leftData, rightData := tree.splitData(data, bestFeature, bestThreshold)

	return &Node{
		Feature:   bestFeature,
		Threshold: bestThreshold,
		Left:      tree.buildTree(leftData, depth+1),
		Right:     tree.buildTree(rightData, depth+1),
	}
}

func (tree *ParallelDecisionTree) findBestSplit(data [][]float64) (int, float64) {
	bestFeature := 0
	bestThreshold := 0.0
	bestGini := float64(1)

	for feature := 0; feature < len(data[0])-1; feature++ {
		threshold := tree.findMedian(data, feature)
		gini := tree.calculateGiniIndex(data, feature, threshold)

		if gini < bestGini {
			bestGini = gini
			bestFeature = feature
			bestThreshold = threshold
		}
	}

	return bestFeature, bestThreshold
}

func (tree *ParallelDecisionTree) findMedian(data [][]float64, feature int) float64 {
	values := make([]float64, len(data))
	for i, row := range data {
		values[i] = row[feature]
	}
	return values[len(values)/2]
}

func (tree *ParallelDecisionTree) calculateGiniIndex(
	data [][]float64,
	feature int,
	threshold float64,
) float64 {
	leftCount, rightCount := 0, 0
	leftPositive, rightPositive := 0, 0

	for _, row := range data {
		if row[feature] < threshold {
			leftCount++
			if row[len(row)-1] == 1 {
				leftPositive++
			}
		} else {
			rightCount++
			if row[len(row)-1] == 1 {
				rightPositive++
			}
		}
	}

	leftGini := 1.0 - math.Pow(
		float64(leftPositive)/float64(leftCount),
		2,
	) - math.Pow(
		float64(leftCount-leftPositive)/float64(leftCount),
		2,
	)
	rightGini := 1.0 - math.Pow(
		float64(rightPositive)/float64(rightCount),
		2,
	) - math.Pow(
		float64(rightCount-rightPositive)/float64(rightCount),
		2,
	)

	totalGini := (float64(leftCount)*leftGini + float64(rightCount)*rightGini) / float64(len(data))
	return totalGini
}

func (tree *ParallelDecisionTree) splitData(
	data [][]float64,
	feature int,
	threshold float64,
) ([][]float64, [][]float64) {
	var left, right [][]float64

	for _, row := range data {
		if row[feature] < threshold {
			left = append(left, row)
		} else {
			right = append(right, row)
		}
	}

	return left, right
}

func (tree *ParallelDecisionTree) calculatePrediction(data [][]float64) float64 {
	sum := 0.0
	for _, row := range data {
		sum += row[len(row)-1]
	}
	return sum / float64(len(data))
}

func (tree *ParallelDecisionTree) Predict(sample []float64) float64 {
	node := tree.root
	for node.Left != nil && node.Right != nil {
		if sample[node.Feature] < node.Threshold {
			node = node.Left
		} else {
			node = node.Right
		}
	}
	return node.Prediction
}
