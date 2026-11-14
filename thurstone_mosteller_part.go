package openskill

import (
	"fmt"
	"math"
	"sort"
)

// ThurstoneMostellerPart implements Algorithm 2 from Weng & Lin (2011) using Thurstone-Mosteller.
// It compares teams within a sliding window of adjacent ranks using Gaussian model distributions.
type ThurstoneMostellerPart struct {
	ModelConfig
}

// NewThurstoneMostellerPart creates a new ThurstoneMostellerPart model.
func NewThurstoneMostellerPart(opts ...ModelOption) *ThurstoneMostellerPart {
	cfg := DefaultModelConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return &ThurstoneMostellerPart{ModelConfig: cfg}
}

// Rating creates a new Rating using the model's default mu and sigma, unless overridden.
func (m *ThurstoneMostellerPart) Rating(mu, sigma *float64, name *string) Rating {
	defMu := m.Mu
	if mu != nil {
		defMu = *mu
	}
	defSigma := m.Sigma
	if sigma != nil {
		defSigma = *sigma
	}
	var n string
	if name != nil {
		n = *name
	}
	return NewRating(defMu, defSigma, n)
}

// CreateRating creates a Rating from a 2-element slice [mu, sigma].
func (m *ThurstoneMostellerPart) CreateRating(rating []float64, name *string) (Rating, error) {
	if len(rating) != 2 {
		return Rating{}, ErrInvalidRatingValues
	}
	var n string
	if name != nil {
		n = *name
	}
	return NewRating(rating[0], rating[1], n), nil
}

func (m *ThurstoneMostellerPart) compute(teams [][]Rating, ranks []float64, scores []float64, weights [][]float64, margin float64) [][]Rating {
	originalTeams := teams
	teamRatings := calculateTeamRatings(teams, ranks, m.Kappa, m.Balance)
	beta := m.Beta
	numTeams := len(teamRatings)
	window := m.WindowSize
	hasScores := len(scores) == numTeams

	result := make([][]Rating, numTeams)
	for i, teamI := range teamRatings {
		var omegaSum float64
		var deltaSum float64
		var comparisons int

		start := i - window
		if start < 0 {
			start = 0
		}
		end := i + window + 1
		if end > numTeams {
			end = numTeams
		}

		for q := start; q < end; q++ {
			if q == i {
				continue
			}

			teamQ := teamRatings[q]
			cIq := 2.0 * math.Sqrt(teamI.SigmaSquared+teamQ.SigmaSquared+2.0*beta*beta)
			deltaMu := (teamI.Mu - teamQ.Mu) / cIq
			sigmaSqToCIq := teamI.SigmaSquared / cIq

			var currentWeights []float64
			if len(weights) > 0 {
				currentWeights = weights[i]
			}
			gammaValue := m.Gamma(cIq, numTeams, teamI.Mu, teamI.SigmaSquared, teamI.Team, teamI.Rank, currentWeights)

			marginDivisor := 1.0
			if hasScores && i != q {
				scoreDiff := math.Abs(scores[i] - scores[q])
				if scoreDiff > 0 {
					if scoreDiff > margin && margin > 0.0 {
						marginDivisor = math.Log1p(scoreDiff / margin)
					}
				}
			}

			if teamQ.Rank > teamI.Rank {
				omegaSum += sigmaSqToCIq * V(deltaMu/marginDivisor, m.Epsilon/cIq)
				deltaSum += (gammaValue * sigmaSqToCIq / cIq) * W(deltaMu/marginDivisor, m.Epsilon/cIq)
			} else if teamQ.Rank < teamI.Rank {
				omegaSum += -sigmaSqToCIq * V(-deltaMu/marginDivisor, m.Epsilon/cIq)
				deltaSum += (gammaValue * sigmaSqToCIq / cIq) * W(-deltaMu/marginDivisor, m.Epsilon/cIq)
			} else {
				omegaSum += sigmaSqToCIq * Vt(deltaMu/marginDivisor, m.Epsilon/cIq)
				deltaSum += (gammaValue * sigmaSqToCIq / cIq) * Wt(deltaMu/marginDivisor, m.Epsilon/cIq)
			}

			comparisons++
		}

		var omega, delta float64
		if comparisons > 0 {
			omega = omegaSum / float64(comparisons)
			delta = deltaSum / float64(comparisons)
		}

		teamResult := make([]Rating, len(teamI.Team))
		for j, player := range teamI.Team {
			wVal := 1.0
			if len(weights) > 0 {
				wVal = weights[i][j]
			}

			newMu := player.Mu
			newSigma := player.Sigma

			if omega > 0 {
				newMu += (newSigma * newSigma / teamI.SigmaSquared) * omega * wVal
				newSigma *= math.Sqrt(math.Max(1.0-(newSigma*newSigma/teamI.SigmaSquared)*delta*wVal, m.Kappa))
			} else {
				newMu += (newSigma * newSigma / teamI.SigmaSquared) * omega / wVal
				newSigma *= math.Sqrt(math.Max(1.0-(newSigma*newSigma/teamI.SigmaSquared)*delta/wVal, m.Kappa))
			}

			modified := originalTeams[i][j].Clone()
			modified.Mu = newMu
			modified.Sigma = newSigma
			teamResult[j] = modified
		}
		result[i] = teamResult
	}

	return result
}

// Rate computes updated ratings for the given teams using Thurstone-Mosteller Partial Pairing.
func (m *ThurstoneMostellerPart) Rate(teams [][]Rating, opts ...RateOption) ([][]Rating, error) {
	var o RateOptions
	for _, opt := range opts {
		opt(&o)
	}
	return m.RateWithOptions(teams, o)
}

// RateWithOptions computes updated ratings using an explicit RateOptions structure.
func (m *ThurstoneMostellerPart) RateWithOptions(teams [][]Rating, options RateOptions) ([][]Rating, error) {
	if err := validateRateInputs(teams, options); err != nil {
		return nil, err
	}

	originalTeams := cloneTeams(teams)
	copiedTeams := cloneTeams(teams)

	tau := m.Tau
	if options.Tau != nil {
		tau = *options.Tau
	}
	tauSq := tau * tau
	for i := range copiedTeams {
		for j := range copiedTeams[i] {
			s := copiedTeams[i][j].Sigma
			copiedTeams[i][j].Sigma = math.Sqrt(s*s + tauSq)
		}
	}

	ranks := options.Ranks
	scores := options.Scores
	weights := copyWeights(options.Weights)
	margin := m.Margin
	if options.Margin != nil {
		margin = *options.Margin
	}

	if len(ranks) == 0 && len(scores) > 0 {
		negScores := make([]float64, len(scores))
		for i, s := range scores {
			negScores[i] = -s
		}
		ranks = CalculateRankings(len(copiedTeams), negScores)
	}

	if len(weights) > 0 && m.WeightBounds != nil {
		for i := range weights {
			weights[i] = Normalize(weights[i], m.WeightBounds.Min, m.WeightBounds.Max)
		}
	}

	var tenet []float64
	if len(ranks) > 0 {
		var orderedTeams [][]Rating
		orderedTeams, tenet = Unwind(ranks, copiedTeams)
		if len(weights) > 0 {
			weights, _ = Unwind(ranks, weights)
		}
		if len(scores) > 0 {
			scores, _ = Unwind(ranks, scores)
		}
		copiedTeams = orderedTeams
		sort.Float64s(ranks)
	}

	var processedResult [][]Rating
	if len(ranks) > 0 && len(tenet) > 0 {
		res := m.compute(copiedTeams, ranks, scores, weights, margin)
		unwoundResult, _ := Unwind(tenet, res)
		processedResult = unwoundResult
	} else {
		processedResult = m.compute(copiedTeams, nil, nil, weights, margin)
	}

	if options.LimitSigma != nil {
		m.LimitSigma = *options.LimitSigma
	}

	if m.LimitSigma {
		for i := range processedResult {
			for j := range processedResult[i] {
				if processedResult[i][j].Sigma > originalTeams[i][j].Sigma {
					processedResult[i][j].Sigma = originalTeams[i][j].Sigma
				}
			}
		}
	}

	return processedResult, nil
}

// PredictWin predicts match win odds for each team.
func (m *ThurstoneMostellerPart) PredictWin(teams [][]Rating) ([]float64, error) {
	return PredictWin(teams, m.Beta, m.Kappa, m.Balance)
}

// PredictDraw predicts the probability of a draw between teams.
func (m *ThurstoneMostellerPart) PredictDraw(teams [][]Rating) (float64, error) {
	return PredictDraw(teams, m.Beta, m.Kappa, m.Balance)
}

// PredictRank predicts the expected rank ordering and normalized win probabilities for all teams.
func (m *ThurstoneMostellerPart) PredictRank(teams [][]Rating) ([]RankProbability, error) {
	return PredictRank(teams, m.Beta, m.Kappa, m.Balance)
}

// Repr returns the representation string for ThurstoneMostellerPart.
func (m *ThurstoneMostellerPart) Repr() string {
	return fmt.Sprintf("ThurstoneMostellerPart(mu=%v, sigma=%v)", m.Mu, m.Sigma)
}

// String returns a human-readable description of the model parameters.
func (m *ThurstoneMostellerPart) String() string {
	return fmt.Sprintf("Thurstone-Mosteller Partial Pairing Model Parameters:\n\nmu: %v\nsigma: %v\n", m.Mu, m.Sigma)
}
