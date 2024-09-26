package randomforest

import (
	"math"
	"math/rand"
	"runtime"
	"strconv"
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

func (rf *ParallelRandomForest) Train(data [][]string) {
	var wg sync.WaitGroup
	treeChan := make(chan int, rf.numTrees)

	for i := 0; i < rf.numTrees; i++ {
		treeChan <- i
	}
	close(treeChan)

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

func (rf *ParallelRandomForest) Predict(sample []string) float64 {
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

func (rf *ParallelRandomForest) createBootstrapSample(data [][]string) [][]string {
	sampleSize := int(float64(len(data)) * rf.subsetRatio)
	sample := make([][]string, sampleSize)
	var wg sync.WaitGroup
	wg.Add(sampleSize)

	for i := 0; i < sampleSize; i++ {
		go func(index int) {
			defer wg.Done()
			randomIndex := rand.Intn(len(data))
			sample[index] = make([]string, len(data[randomIndex]))
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

func (tree *ParallelDecisionTree) Train(data [][]string) {
	tree.root = tree.buildTree(data, 0)
}

func (tree *ParallelDecisionTree) buildTree(data [][]string, depth int) *Node {
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

func (tree *ParallelDecisionTree) findBestSplit(data [][]string) (int, float64) {
	bestFeature := 0
	bestThreshold := 0.0
	bestGini := float64(1)

	for feature := 1; feature < len(data[0])-1; feature++ {
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

func (tree *ParallelDecisionTree) findMedian(data [][]string, feature int) float64 {
	values := make([]float64, 0, len(data))
	for _, row := range data {
		if val, err := strconv.ParseFloat(row[feature], 64); err == nil {
			values = append(values, val)
		}
	}
	if len(values) == 0 {
		return 0
	}
	return values[len(values)/2]
}

func (tree *ParallelDecisionTree) calculateGiniIndex(
	data [][]string,
	feature int,
	threshold float64,
) float64 {
	leftCount, rightCount := 0, 0
	leftPositive, rightPositive := 0, 0

	for _, row := range data {
		val, err := strconv.ParseFloat(row[feature], 64)
		if err != nil {
			continue
		}

		if val < threshold {
			leftCount++
			if row[len(row)-1] == "SI" {
				leftPositive++
			}
		} else {
			rightCount++
			if row[len(row)-1] == "SI" {
				rightPositive++
			}
		}
	}

	if leftCount == 0 || rightCount == 0 {
		return 1.0
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
	data [][]string,
	feature int,
	threshold float64,
) ([][]string, [][]string) {
	var left, right [][]string

	for _, row := range data {
		val, err := strconv.ParseFloat(row[feature], 64)
		if err != nil {
			continue
		}

		if val < threshold {
			left = append(left, row)
		} else {
			right = append(right, row)
		}
	}

	return left, right
}

func (tree *ParallelDecisionTree) calculatePrediction(data [][]string) float64 {
	sum := 0.0
	count := 0
	for _, row := range data {
		if row[len(row)-1] == "SI" {
			sum += 1
		}
		count++
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func (tree *ParallelDecisionTree) Predict(sample []string) float64 {
	node := tree.root
	for node.Left != nil && node.Right != nil {
		val, err := strconv.ParseFloat(sample[node.Feature], 64)
		if err != nil {
			break
		}
		if val < node.Threshold {
			node = node.Left
		} else {
			node = node.Right
		}
	}
	return node.Prediction
}

