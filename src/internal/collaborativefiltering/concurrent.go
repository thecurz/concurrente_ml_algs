// File: src/internal/collaborativefiltering/concurrent.go

package collaborativefiltering

import (
	"sync"
)

type ConcurrentMatrixFactorization struct {
	*MatrixFactorization
}

func NewConcurrentMatrixFactorization(
	numUsers, numItems, numFactors int,
	learningRate, regularization float64,
	epochs int,
) *ConcurrentMatrixFactorization {
	return &ConcurrentMatrixFactorization{
		MatrixFactorization: NewMatrixFactorization(
			numUsers,
			numItems,
			numFactors,
			learningRate,
			regularization,
			epochs,
		),
	}
}

func (mf *ConcurrentMatrixFactorization) Train(ratings [][]float64) {
	var wg sync.WaitGroup
	numGoroutines := 4 // Adjust based on your system's capabilities

	for epoch := 0; epoch < mf.Epochs; epoch++ {
		wg.Add(numGoroutines)
		chunkSize := len(ratings) / numGoroutines

		for i := 0; i < numGoroutines; i++ {
			start := i * chunkSize
			end := start + chunkSize
			if i == numGoroutines-1 {
				end = len(ratings)
			}

			go func(start, end int) {
				defer wg.Done()
				mf.trainChunk(ratings[start:end])
			}(start, end)
		}

		wg.Wait()
	}
}

func (mf *ConcurrentMatrixFactorization) trainChunk(chunk [][]float64) {
	for userID, userRatings := range chunk {
		for itemID, rating := range userRatings {
			if rating > 0 {
				mf.updateFactors(userID, itemID, rating)
			}
		}
	}
}

func (mf *ConcurrentMatrixFactorization) updateFactors(userID, itemID int, rating float64) {
	prediction := mf.predict(userID, itemID)
	err := rating - prediction

	for f := 0; f < mf.NumFactors; f++ {
		userFactor := mf.UserFactors[userID][f]
		itemFactor := mf.ItemFactors[itemID][f]

		mf.UserFactors[userID][f] += mf.LearningRate * (err*itemFactor - mf.Regularization*userFactor)
		mf.ItemFactors[itemID][f] += mf.LearningRate * (err*userFactor - mf.Regularization*itemFactor)
	}
}
