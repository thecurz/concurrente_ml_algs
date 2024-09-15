// File: src/internal/collaborativefiltering/sequential.go

package collaborativefiltering

type SequentialMatrixFactorization struct {
	*MatrixFactorization
}

func NewSequentialMatrixFactorization(
	numUsers, numItems, numFactors int,
	learningRate, regularization float64,
	epochs int,
) *SequentialMatrixFactorization {
	return &SequentialMatrixFactorization{
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

func (mf *SequentialMatrixFactorization) Train(ratings [][]float64) {
	for epoch := 0; epoch < mf.Epochs; epoch++ {
		for userID, userRatings := range ratings {
			for itemID, rating := range userRatings {
				if rating > 0 {
					mf.updateFactors(userID, itemID, rating)
				}
			}
		}
	}
}

func (mf *SequentialMatrixFactorization) updateFactors(userID, itemID int, rating float64) {
	prediction := mf.predict(userID, itemID)
	err := rating - prediction

	for f := 0; f < mf.NumFactors; f++ {
		userFactor := mf.UserFactors[userID][f]
		itemFactor := mf.ItemFactors[itemID][f]

		mf.UserFactors[userID][f] += mf.LearningRate * (err*itemFactor - mf.Regularization*userFactor)
		mf.ItemFactors[itemID][f] += mf.LearningRate * (err*userFactor - mf.Regularization*itemFactor)
	}
}
