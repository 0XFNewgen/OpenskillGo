// Package openskill provides a fast, open-source Bayesian multi-player, multi-team rating and matchmaking system.
// It is an idiomatic Go port of openskill.py (v6.2.0), implementing all five Weng-Lin models:
// Plackett-Luce, Bradley-Terry Full Pairing, Bradley-Terry Partial Pairing,
// Thurstone-Mosteller Full Pairing, and Thurstone-Mosteller Partial Pairing.
package openskill

// Model represents any ranking model supported by openskill.
type Model interface {
	Rate(teams [][]Rating, opts ...RateOption) ([][]Rating, error)
	RateWithOptions(teams [][]Rating, options RateOptions) ([][]Rating, error)
	PredictWin(teams [][]Rating) ([]float64, error)
	PredictDraw(teams [][]Rating) (float64, error)
	PredictRank(teams [][]Rating) ([]RankProbability, error)
}

// DefaultModel is the default Plackett-Luce model used by package-level convenience functions.
var DefaultModel = NewPlackettLuce()

// Rate computes updated ratings for the given teams using the default Plackett-Luce model.
func Rate(teams [][]Rating, opts ...RateOption) ([][]Rating, error) {
	return DefaultModel.Rate(teams, opts...)
}

// RateWithOptions computes updated ratings using an explicit RateOptions struct with the default model.
func RateWithOptions(teams [][]Rating, options RateOptions) ([][]Rating, error) {
	return DefaultModel.RateWithOptions(teams, options)
}

// PredictWin computes win probabilities using the default model.
func PredictWinDefault(teams [][]Rating) ([]float64, error) {
	return DefaultModel.PredictWin(teams)
}

// PredictDrawDefault computes draw probability using the default model.
func PredictDrawDefault(teams [][]Rating) (float64, error) {
	return DefaultModel.PredictDraw(teams)
}

// PredictRankDefault computes rank probabilities using the default model.
func PredictRankDefault(teams [][]Rating) ([]RankProbability, error) {
	return DefaultModel.PredictRank(teams)
}
