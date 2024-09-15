// File: src/internal/collaborativefiltering/common.go

package collaborativefiltering

import (
	"encoding/csv"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
)

type MatrixFactorization struct {
	UserFactors    [][]float64
	ItemFactors    [][]float64
	NumFactors     int
	LearningRate   float64
	Regularization float64
	Epochs         int
}

type Review struct {
	UserID    string
	ProductID string
	Score     float64
}

func NewMatrixFactorization(
	numUsers, numItems, numFactors int,
	learningRate, regularization float64,
	epochs int,
) *MatrixFactorization {
	mf := &MatrixFactorization{
		NumFactors:     numFactors,
		LearningRate:   learningRate,
		Regularization: regularization,
		Epochs:         epochs,
	}

	mf.UserFactors = make([][]float64, numUsers)
	mf.ItemFactors = make([][]float64, numItems)

	for i := range mf.UserFactors {
		mf.UserFactors[i] = make([]float64, numFactors)
		for j := range mf.UserFactors[i] {
			mf.UserFactors[i][j] = rand.Float64() * 0.1
		}
	}

	for i := range mf.ItemFactors {
		mf.ItemFactors[i] = make([]float64, numFactors)
		for j := range mf.ItemFactors[i] {
			mf.ItemFactors[i][j] = rand.Float64() * 0.1
		}
	}

	return mf
}

func (mf *MatrixFactorization) predict(userID, itemID int) float64 {
	dot := 0.0
	for f := 0; f < mf.NumFactors; f++ {
		dot += mf.UserFactors[userID][f] * mf.ItemFactors[itemID][f]
	}
	return dot
}

func (mf *MatrixFactorization) Predict(userID, itemID int) float64 {
	return mf.predict(userID, itemID)
}

func (mf *MatrixFactorization) CalculateRMSE(ratings [][]float64) float64 {
	var sumSquaredError float64
	var count int

	for userID, userRatings := range ratings {
		for itemID, rating := range userRatings {
			if rating > 0 {
				prediction := mf.predict(userID, itemID)
				sumSquaredError += math.Pow(rating-prediction, 2)
				count++
			}
		}
	}

	return math.Sqrt(sumSquaredError / float64(count))
}

func ReadAmazonReviews(filename string, limit int) ([]Review, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var reviews []Review

	// Skip header
	if _, err := reader.Read(); err != nil {
		return nil, err
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		score, err := strconv.ParseFloat(record[6], 64)
		if err != nil {
			continue
		}

		reviews = append(reviews, Review{
			UserID:    record[1],
			ProductID: record[0],
			Score:     score,
		})

		if limit > 0 && len(reviews) >= limit {
			break
		}
	}

	return reviews, nil
}

func ConvertToMatrix(reviews []Review) ([][]float64, map[string]int, map[string]int) {
	userMap := make(map[string]int)
	productMap := make(map[string]int)

	for _, review := range reviews {
		if _, exists := userMap[review.UserID]; !exists {
			userMap[review.UserID] = len(userMap)
		}
		if _, exists := productMap[review.ProductID]; !exists {
			productMap[review.ProductID] = len(productMap)
		}
	}

	matrix := make([][]float64, len(userMap))
	for i := range matrix {
		matrix[i] = make([]float64, len(productMap))
	}

	for _, review := range reviews {
		userID := userMap[review.UserID]
		productID := productMap[review.ProductID]
		matrix[userID][productID] = review.Score
	}

	return matrix, userMap, productMap
}
