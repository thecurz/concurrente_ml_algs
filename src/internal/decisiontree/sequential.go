package decisiontree

import (
	"fmt"
	"math"
	"strings"
)

type SequentialDecisionTree struct {
	root *Node
}

type Node struct {
	Feature    int
	Threshold  float64
	Left       *Node
	Right      *Node
	Prediction float64
}

func (dt *SequentialDecisionTree) Predict(sample []float64) float64 {
	return dt.predictNode(dt.root, sample)
}

func (dt *SequentialDecisionTree) predictNode(node *Node, sample []float64) float64 {
	if node.Left == nil && node.Right == nil {
		return node.Prediction
	}

	if sample[node.Feature] <= node.Threshold {
		return dt.predictNode(node.Left, sample)
	}
	return dt.predictNode(node.Right, sample)
}

func NewSequentialDecisionTree() *SequentialDecisionTree {
	return &SequentialDecisionTree{}
}

func (dt *SequentialDecisionTree) Train(data [][]float64) {
	dt.root = dt.buildTree(data, 0, 5)
}

func (dt *SequentialDecisionTree) buildTree(data [][]float64, depth, maxDepth int) *Node {
	if len(data) == 0 || depth >= maxDepth {
		return &Node{Prediction: dt.calculatePrediction(data)}
	}

	bestFeature, bestThreshold := dt.findBestSplit(data)

	if bestFeature == -1 {
		return &Node{Prediction: dt.calculatePrediction(data)}
	}

	leftData, rightData := dt.splitData(data, bestFeature, bestThreshold)

	return &Node{
		Feature:   bestFeature,
		Threshold: bestThreshold,
		Left:      dt.buildTree(leftData, depth+1, maxDepth),
		Right:     dt.buildTree(rightData, depth+1, maxDepth),
	}
}

func (dt *SequentialDecisionTree) findBestSplit(data [][]float64) (int, float64) {
	bestFeature := -1
	bestThreshold := 0.0
	bestGini := math.Inf(1)

	for feature := 0; feature < len(data[0])-1; feature++ {
		thresholds := dt.getUniqueValues(data, feature)

		for _, threshold := range thresholds {
			gini := dt.calculateGiniIndex(data, feature, threshold)
			if gini < bestGini {
				bestGini = gini
				bestFeature = feature
				bestThreshold = threshold
			}
		}
	}

	return bestFeature, bestThreshold
}

func (dt *SequentialDecisionTree) getUniqueValues(data [][]float64, feature int) []float64 {
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

func (dt *SequentialDecisionTree) calculateGiniIndex(
	data [][]float64,
	feature int,
	threshold float64,
) float64 {
	leftData, rightData := dt.splitData(data, feature, threshold)

	leftGini := dt.calculateGini(leftData)
	rightGini := dt.calculateGini(rightData)

	totalSize := float64(len(data))
	weightedGini := (float64(len(leftData))/totalSize)*leftGini + (float64(len(rightData))/totalSize)*rightGini

	return weightedGini
}

func (dt *SequentialDecisionTree) calculateGini(data [][]float64) float64 {
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

func (dt *SequentialDecisionTree) splitData(
	data [][]float64,
	feature int,
	threshold float64,
) ([][]float64, [][]float64) {
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

func (dt *SequentialDecisionTree) calculatePrediction(data [][]float64) float64 {
	if len(data) == 0 {
		return 0
	}

	sum := 0.0
	for _, row := range data {
		sum += row[len(row)-1]
	}
	return sum / float64(len(data))
}

func (dt *SequentialDecisionTree) PrintTree() {
	dt.printNode(dt.root, 0)
}

func (dt *SequentialDecisionTree) printNode(node *Node, depth int) {
	if node == nil {
		return
	}

	indent := strings.Repeat("  ", depth)
	if node.Left == nil && node.Right == nil {
		fmt.Printf("%sLeaf: Prediction = %.2f\n", indent, node.Prediction)
	} else {
		fmt.Printf("%sNode: Feature %d, Threshold %.2f\n", indent, node.Feature, node.Threshold)
		dt.printNode(node.Left, depth+1)
		dt.printNode(node.Right, depth+1)
	}
}
