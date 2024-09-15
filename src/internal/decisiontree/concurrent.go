package decisiontree

import (
	"math"
	"sync"
)

type ConcurrentDecisionTree struct {
	root *Node
}

func NewConcurrentDecisionTree() *ConcurrentDecisionTree {
	return &ConcurrentDecisionTree{}
}

func (dt *ConcurrentDecisionTree) Train(data [][]float64) {
	// fmt.Println("Starting concurrent decision tree training...")
	// startTime := time.Now()
	dt.root = dt.buildTree(data, 0, 5)
	// fmt.Printf("Concurrent training completed in %v\n", time.Since(startTime))
}

func (dt *ConcurrentDecisionTree) buildTree(data [][]float64, depth, maxDepth int) *Node {
	if len(data) == 0 || depth >= maxDepth {
		return &Node{Prediction: calculatePrediction(data)}
	}

	// splitStart := time.Now()
	bestFeature, bestThreshold := dt.findBestSplitConcurrent(data)
	// fmt.Printf("Depth %d: Found best split in %v\n", depth, time.Since(splitStart))

	if bestFeature == -1 {
		return &Node{Prediction: calculatePrediction(data)}
	}

	leftData, rightData := splitData(data, bestFeature, bestThreshold)
	/*
		fmt.Printf(
			"Depth %d: Split data into %d left and %d right\n",
			depth,
			len(leftData),
			len(rightData),
		)
	*/

	return &Node{
		Feature:   bestFeature,
		Threshold: bestThreshold,
		Left:      dt.buildTree(leftData, depth+1, maxDepth),
		Right:     dt.buildTree(rightData, depth+1, maxDepth),
	}
}

func (dt *ConcurrentDecisionTree) findBestSplitConcurrent(data [][]float64) (int, float64) {
	numFeatures := len(data[0]) - 1
	results := make(chan struct {
		feature   int
		threshold float64
		gini      float64
	}, numFeatures)

	var wg sync.WaitGroup
	for feature := 0; feature < numFeatures; feature++ {
		wg.Add(1)
		go func(f int) {
			defer wg.Done()
			bestThreshold, bestGini := findBestThresholdForFeature(data, f)
			results <- struct {
				feature   int
				threshold float64
				gini      float64
			}{f, bestThreshold, bestGini}
		}(feature)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	bestFeature := -1
	bestThreshold := 0.0
	bestGini := math.Inf(1)

	for result := range results {
		if result.gini < bestGini {
			bestFeature = result.feature
			bestThreshold = result.threshold
			bestGini = result.gini
		}
	}

	return bestFeature, bestThreshold
}

func findBestThresholdForFeature(data [][]float64, feature int) (float64, float64) {
	thresholds := getUniqueValues(data, feature)
	bestThreshold := 0.0
	bestGini := math.Inf(1)

	for _, threshold := range thresholds {
		gini := calculateGiniIndex(data, feature, threshold)
		if gini < bestGini {
			bestGini = gini
			bestThreshold = threshold
		}
	}

	return bestThreshold, bestGini
}

func (dt *ConcurrentDecisionTree) Predict(sample []float64) float64 {
	return predictNode(dt.root, sample)
}

// The following functions can be shared between sequential and concurrent versions
// You may want to move them to a common file if they're identical

func getUniqueValues(data [][]float64, feature int) []float64 {
	uniqueMap := make(map[float64]bool)
	for _, row := range data {
		uniqueMap[row[feature]] = true
	}
	unique := make([]float64, 0, len(uniqueMap))
	for value := range uniqueMap {
		unique = append(unique, value)
	}
	return unique
}

func calculateGiniIndex(data [][]float64, feature int, threshold float64) float64 {
	leftData, rightData := splitData(data, feature, threshold)
	leftGini := calculateGini(leftData)
	rightGini := calculateGini(rightData)
	totalSize := float64(len(data))
	weightedGini := (float64(len(leftData))/totalSize)*leftGini + (float64(len(rightData))/totalSize)*rightGini
	return weightedGini
}

func calculateGini(data [][]float64) float64 {
	if len(data) == 0 {
		return 0
	}
	positiveCount := 0
	for _, row := range data {
		if row[len(row)-1] == 1 {
			positiveCount++
		}
	}
	p := float64(positiveCount) / float64(len(data))
	return 2 * p * (1 - p)
}

func splitData(data [][]float64, feature int, threshold float64) ([][]float64, [][]float64) {
	var left, right [][]float64
	for _, row := range data {
		if row[feature] <= threshold {
			left = append(left, row)
		} else {
			right = append(right, row)
		}
	}
	return left, right
}

func calculatePrediction(data [][]float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, row := range data {
		sum += row[len(row)-1]
	}
	return sum / float64(len(data))
}

func predictNode(node *Node, sample []float64) float64 {
	if node.Left == nil && node.Right == nil {
		return node.Prediction
	}
	if sample[node.Feature] <= node.Threshold {
		return predictNode(node.Left, sample)
	}
	return predictNode(node.Right, sample)
}
