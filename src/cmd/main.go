package main

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"

	"concurrente/internal/ann"
	"concurrente/internal/collaborativefiltering"
	"concurrente/internal/decisiontree"
	"concurrente/internal/randomforest"
	"concurrente/internal/svm"
)

func main() {
	fmt.Println("Starting Machine Learning Algorithms Comparison")
	startTime := time.Now()

	allData := readAndPrepareData()
	trainData, testData := splitData(allData, 0.8)

	runAlgorithmComparison(
		"Decision Tree",
		func() (time.Duration, time.Duration, float64, float64) {
			return testDecisionTree(trainData, testData)
		},
	)

	runAlgorithmComparison(
		"Random Forest",
		func() (time.Duration, time.Duration, float64, float64) {
			return testRandomForest(trainData, testData)
		},
	)

	runAlgorithmComparison("SVM", func() (time.Duration, time.Duration, float64, float64) {
		return testSVM(trainData, testData)
	})

	runAlgorithmComparison("ANN", func() (time.Duration, time.Duration, float64, float64) {
		return testANN(trainData, testData)
	})

	fmt.Println("\n--- Testing Collaborative Filtering ---")
	testCollaborativeFiltering("datasets/Reviews.csv")

	fmt.Printf("\nTotal execution time: %v\n", time.Since(startTime))
}

func runAlgorithmComparison(
	name string,
	testFunc func() (time.Duration, time.Duration, float64, float64),
) {
	fmt.Printf("\n--- Testing %s ---\n", name)
	seqTime, concTime, seqAccuracy, concAccuracy := testFunc()
	speedup := float64(seqTime) / float64(concTime)
	fmt.Printf("Sequential Time: %v\n", seqTime)
	fmt.Printf("Concurrent Time: %v\n", concTime)
	fmt.Printf("Speedup: %.2fx\n", speedup)
	fmt.Printf("Sequential Accuracy: %.2f%%\n", seqAccuracy*100)
	fmt.Printf("Concurrent Accuracy: %.2f%%\n", concAccuracy*100)
}

func readAndPrepareData() [][]float64 {
	startTime := time.Now()
	allData, err := readCSV("datasets/adult.csv", 100000) // Read all data
	if err != nil {
		fmt.Println("Error reading CSV:", err)
		os.Exit(1)
	}
	fmt.Printf("Read %d records from the dataset\n", len(allData))
	fmt.Printf("Time taken to read data: %v\n", time.Since(startTime))
	return allData
}

func testDecisionTree(
	trainData, testData [][]float64,
) (time.Duration, time.Duration, float64, float64) {
	seqTree := decisiontree.NewSequentialDecisionTree()
	concTree := decisiontree.NewConcurrentDecisionTree()

	seqTime, seqAccuracy := trainAndTestAlgorithm(seqTree, trainData, testData)
	concTime, concAccuracy := trainAndTestAlgorithm(concTree, trainData, testData)

	return seqTime, concTime, seqAccuracy, concAccuracy
}

func testRandomForest(
	trainData, testData [][]float64,
) (time.Duration, time.Duration, float64, float64) {
	seqRF := randomforest.NewSequentialRandomForest(10, 0.8)
	concRF := randomforest.NewConcurrentRandomForest(10, 0.8)

	seqTime, seqAccuracy := trainAndTestAlgorithm(seqRF, trainData, testData)
	concTime, concAccuracy := trainAndTestAlgorithm(concRF, trainData, testData)

	return seqTime, concTime, seqAccuracy, concAccuracy
}

func testSVM(trainData, testData [][]float64) (time.Duration, time.Duration, float64, float64) {
	seqSVM := svm.NewSequentialSVM(len(trainData[0])-1, 0.01, 0.001, 10)
	concSVM := svm.NewConcurrentSVM(len(trainData[0])-1, 0.01, 0.001, 10)

	seqTime, seqAccuracy := trainAndTestAlgorithm(seqSVM, trainData, testData)
	concTime, concAccuracy := trainAndTestAlgorithm(concSVM, trainData, testData)

	return seqTime, concTime, seqAccuracy, concAccuracy
}

func testANN(trainData, testData [][]float64) (time.Duration, time.Duration, float64, float64) {
	inputSize := len(trainData[0]) - 1
	hiddenSize := 10
	learningRate := 0.01
	epochs := 10

	seqANN := ann.NewSequentialANN(inputSize, hiddenSize, learningRate, epochs)
	concANN := ann.NewConcurrentANN(inputSize, hiddenSize, learningRate, epochs)

	seqTime, seqAccuracy := trainAndTestAlgorithm(seqANN, trainData, testData)
	concTime, concAccuracy := trainAndTestAlgorithm(concANN, trainData, testData)

	return seqTime, concTime, seqAccuracy, concAccuracy
}

func trainAndTestAlgorithm(
	algorithm interface{},
	trainData, testData [][]float64,
) (time.Duration, float64) {
	trainingStart := time.Now()
	switch a := algorithm.(type) {
	case *decisiontree.SequentialDecisionTree:
		a.Train(trainData)
	case *decisiontree.ConcurrentDecisionTree:
		a.Train(trainData)
	case *randomforest.SequentialRandomForest:
		a.Train(trainData)
	case *randomforest.ConcurrentRandomForest:
		a.Train(trainData)
	case *svm.SequentialSVM:
		a.Train(trainData)
	case *svm.ConcurrentSVM:
		a.Train(trainData)
	case *ann.SequentialANN:
		a.Train(trainData)
	case *ann.ConcurrentANN:
		a.Train(trainData)
	default:
		panic("Unknown algorithm type")
	}
	trainingDuration := time.Since(trainingStart)

	accuracy := testAlgorithm(algorithm, testData)

	return trainingDuration, accuracy
}

func testAlgorithm(algorithm interface{}, testData [][]float64) float64 {
	correct := 0
	for _, sample := range testData {
		features := sample[:len(sample)-1]
		actualLabel := sample[len(sample)-1]
		var predictedLabel float64

		switch a := algorithm.(type) {
		case *decisiontree.SequentialDecisionTree:
			predictedLabel = a.Predict(features)
		case *decisiontree.ConcurrentDecisionTree:
			predictedLabel = a.Predict(features)
		case *randomforest.SequentialRandomForest:
			predictedLabel = a.Predict(features)
		case *randomforest.ConcurrentRandomForest:
			predictedLabel = a.Predict(features)
		case *svm.SequentialSVM:
			predictedLabel = a.Predict(features)
		case *svm.ConcurrentSVM:
			predictedLabel = a.Predict(features)
		case *ann.SequentialANN:
			predictedLabel = a.Predict(features)
		case *ann.ConcurrentANN:
			predictedLabel = a.Predict(features)
		default:
			panic("Unknown algorithm type")
		}

		if (predictedLabel > 0.5 && actualLabel == 1) ||
			(predictedLabel <= 0.5 && actualLabel == 0) {
			correct++
		}
	}

	return float64(correct) / float64(len(testData))
}

func testCollaborativeFiltering(filename string) {
	reviews, err := collaborativefiltering.ReadAmazonReviews(filename, 100000)
	if err != nil {
		fmt.Println("Error reading data:", err)
		return
	}

	ratingMatrix, userMap, productMap := collaborativefiltering.ConvertToMatrix(reviews)
	numUsers := len(userMap)
	numItems := len(productMap)

	fmt.Printf("Loaded %d users and %d items\n", numUsers, numItems)

	numFactors := 10
	learningRate := 0.005
	regularization := 0.02
	epochs := 20

	// Sequential version
	seqMF := collaborativefiltering.NewSequentialMatrixFactorization(
		numUsers,
		numItems,
		numFactors,
		learningRate,
		regularization,
		epochs,
	)
	seqStart := time.Now()
	seqMF.Train(ratingMatrix)
	seqDuration := time.Since(seqStart)
	seqRMSE := seqMF.CalculateRMSE(ratingMatrix)

	// Concurrent version
	concMF := collaborativefiltering.NewConcurrentMatrixFactorization(
		numUsers,
		numItems,
		numFactors,
		learningRate,
		regularization,
		epochs,
	)
	concStart := time.Now()
	concMF.Train(ratingMatrix)
	concDuration := time.Since(concStart)
	concRMSE := concMF.CalculateRMSE(ratingMatrix)

	// Compare performance
	speedup := float64(seqDuration) / float64(concDuration)
	fmt.Printf("Sequential Time: %v\n", seqDuration)
	fmt.Printf("Concurrent Time: %v\n", concDuration)
	fmt.Printf("Speedup: %.2fx\n", speedup)
	fmt.Printf("Sequential RMSE: %.4f\n", seqRMSE)
	fmt.Printf("Concurrent RMSE: %.4f\n", concRMSE)
}

// readCSV and splitData functions remain unchanged

func readCSV(filename string, limit int) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var data [][]float64
	for i, record := range records {
		if i == 0 { // Skip header
			continue
		}
		if limit > 0 && i > limit { // Limit the number of rows if specified
			break
		}
		row := make([]float64, len(record))
		for j, value := range record {
			if j == len(record)-1 { // Last column is the target
				if value == ">50K" {
					row[j] = 1
				} else {
					row[j] = 0
				}
			} else {
				floatValue, err := strconv.ParseFloat(value, 64)
				if err != nil {
					// If parsing fails, assign a default value
					floatValue = 0
				}
				row[j] = floatValue
			}
		}
		data = append(data, row)
	}

	return data, nil
}

func splitData(data [][]float64, trainRatio float64) ([][]float64, [][]float64) {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })

	splitIndex := int(float64(len(data)) * trainRatio)
	return data[:splitIndex], data[splitIndex:]
}
