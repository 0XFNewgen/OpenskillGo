package openskill

import (
	"fmt"
	"math"
	"sort"
)

// BradleyTerryFull implements Algorithm 1 from Weng & Lin (2011).
// It models pairwise interactions between all teams using logistic comparison.
type BradleyTerryFull struct {
	ModelConfig
}

// NewBradleyTerryFull creates a new BradleyTerryFull model.
func NewBradleyTerryFull(opts ...ModelOption) *BradleyTerryFull {
	cfg := DefaultModelConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return &BradleyTerryFull{ModelConfig: cfg}
}

// Rating creates a new Rating using the model's default mu and sigma, unless overridden.
func (m *BradleyTerryFull) Rating(mu, sigma *float64, name *string) Rating {
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
func (m *BradleyTerryFull) CreateRating(rating []float64, name *string) (Rating, error) {
	if len(rating) != 2 {
		return Rating{}, ErrInvalidRatingValues
	}
	var n string
	if name != nil {
		n = *name
	}
	return NewRating(rating[0], rating[1], n), nil
}

func (m *BradleyTerryFull) compute(teams [][]Rating, ranks []float64, scores []float64, weights [][]float64, margin float64) [][]Rating {
	originalTeams := teams
	teamRatings := calculateTeamRatings(teams, ranks, m.Kappa, m.Balance)
	beta := m.Beta
	n := len(teamRatings)
	hasScores := len(scores) == n

	result := make([][]Rating, n)
	for i, teamI := range teamRatings {
		var omega float64
		var delta float64

		for q, teamQ := range teamRatings {
			if q == i {
				continue
			}

			marginDivisor := 1.0
			if hasScores && i != q {
				scoreDiff := math.Abs(scores[i] - scores[q])
				if scoreDiff > 0 {
					if scoreDiff > margin && margin > 0.0 {
						marginDivisor = math.Log1p(scoreDiff / margin)
					}
				}
			}

			cIq := math.Sqrt(teamI.SigmaSquared + teamQ.SigmaSquared + 2.0*beta*beta)
			pIq := 1.0 / (1.0 + math.Exp(((teamQ.Mu-teamI.Mu)/marginDivisor)/cIq))
			sigmaSquaredToCIq := teamI.SigmaSquared / cIq

			s := 0.0
			if teamQ.Rank > teamI.Rank {
				s = 1.0
			} else if teamQ.Rank == teamI.Rank {
				s = 0.5
			}

			omega += sigmaSquaredToCIq * (s - pIq)

			var currentWeights []float64
			if len(weights) > 0 {
				currentWeights = weights[i]
			}
			gammaValue := m.Gamma(cIq, n, teamI.Mu, teamI.SigmaSquared, teamI.Team, teamI.Rank, currentWeights)
			delta += ((gammaValue * sigmaSquaredToCIq) / cIq) * pIq * (1.0 - pIq)
		}

		teamResult := make([]Rating, len(teamI.Team))
		for j, player := range teamI.Team {
			w := 1.0
			if len(weights) > 0 {
				w = weights[i][j]
			}

			mu := player.Mu
			sigma := player.Sigma

			if omega >= 0 {
				mu += (sigma * sigma / teamI.SigmaSquared) * omega * w
				sigma *= math.Sqrt(math.Max(1.0-(sigma*sigma/teamI.SigmaSquared)*delta*w, m.Kappa))
			} else {
				mu += (sigma * sigma / teamI.SigmaSquared) * omega / w
				sigma *= math.Sqrt(math.Max(1.0-(sigma*sigma/teamI.SigmaSquared)*delta/w, m.Kappa))
			}

			modified := originalTeams[i][j].Clone()
			modified.Mu = mu
			modified.Sigma = sigma
			teamResult[j] = modified
		}
		result[i] = teamResult
	}

	return result
}

// Rate computes updated ratings for the given teams using Bradley-Terry Full Pairing.
func (m *BradleyTerryFull) Rate(teams [][]Rating, opts ...RateOption) ([][]Rating, error) {
	var o RateOptions
	for _, opt := range opts {
		opt(&o)
	}
	return m.RateWithOptions(teams, o)
}

// RateWithOptions computes updated ratings using an explicit RateOptions structure.
func (m *BradleyTerryFull) RateWithOptions(teams [][]Rating, options RateOptions) ([][]Rating, error) {
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
func (m *BradleyTerryFull) PredictWin(teams [][]Rating) ([]float64, error) {
	return PredictWin(teams, m.Beta, m.Kappa, m.Balance)
}

// PredictDraw predicts the probability of a draw between teams.
func (m *BradleyTerryFull) PredictDraw(teams [][]Rating) (float64, error) {
	return PredictDraw(teams, m.Beta, m.Kappa, m.Balance)
}

// PredictRank predicts the expected rank ordering and normalized win probabilities for all teams.
func (m *BradleyTerryFull) PredictRank(teams [][]Rating) ([]RankProbability, error) {
	return PredictRank(teams, m.Beta, m.Kappa, m.Balance)
}

// Repr returns the representation string for BradleyTerryFull.
func (m *BradleyTerryFull) Repr() string {
	return fmt.Sprintf("BradleyTerryFull(mu=%v, sigma=%v)", m.Mu, m.Sigma)
}

// String returns a human-readable description of the model parameters.
func (m *BradleyTerryFull) String() string {
	return fmt.Sprintf("Bradley-Terry Full Pairing Model Parameters:\n\nmu: %v\nsigma: %v\n", m.Mu, m.Sigma)
}
