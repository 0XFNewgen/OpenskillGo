package openskill

import (
	"fmt"
	"math"
	"sort"
)

// PlackettLuce implements Algorithm 4 from Weng & Lin (2011).
// It captures player performance across multiple dimensions, providing a rich framework for multi-team rankings.
type PlackettLuce struct {
	ModelConfig
}

// NewPlackettLuce creates a new PlackettLuce model with default or customized options.
func NewPlackettLuce(opts ...ModelOption) *PlackettLuce {
	cfg := DefaultModelConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return &PlackettLuce{ModelConfig: cfg}
}

// Rating creates a new Rating using the model's default mu and sigma, unless overridden.
func (m *PlackettLuce) Rating(mu, sigma *float64, name *string) Rating {
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
func (m *PlackettLuce) CreateRating(rating []float64, name *string) (Rating, error) {
	if len(rating) != 2 {
		return Rating{}, ErrInvalidRatingValues
	}
	var n string
	if name != nil {
		n = *name
	}
	return NewRating(rating[0], rating[1], n), nil
}

// C calculates the square root of collective team sigma (Procedure 3 from Weng & Lin 2011).
func (m *PlackettLuce) C(teamRatings []TeamRating) float64 {
	betaSq := m.Beta * m.Beta
	var collective float64
	for _, t := range teamRatings {
		collective += t.SigmaSquared + betaSq
	}
	return math.Sqrt(collective)
}

// SumQ calculates the sum of (mu / c) exponentiated for teams ranked below or equal.
func (m *PlackettLuce) SumQ(teamRatings []TeamRating, c float64, scores []float64, margin float64) []float64 {
	n := len(teamRatings)
	hasScores := len(scores) == n

	adjustedMu := make([]float64, n)
	for i, teamI := range teamRatings {
		adj := teamI.Mu
		if hasScores {
			var marginAdjustment float64
			var comparisonCount int
			for j, teamJ := range teamRatings {
				if i != j {
					scoreDiff := math.Abs(scores[i] - scores[j])
					if scoreDiff > 0 {
						marginFactor := 1.0
						if scoreDiff > margin && margin > 0.0 {
							marginFactor = math.Log1p(scoreDiff / margin)
						}
						if scores[i] > scores[j] {
							marginAdjustment -= (teamI.Mu - teamJ.Mu) * (marginFactor - 1.0)
						} else {
							marginAdjustment += (teamJ.Mu - teamI.Mu) * (marginFactor - 1.0)
						}
						comparisonCount++
					}
				}
			}
			if comparisonCount > 0 {
				adj += marginAdjustment / float64(comparisonCount)
			} else {
				adj += marginAdjustment
			}
		}
		adjustedMu[i] = adj
	}

	sumQ := make([]float64, n)
	for i := range teamRatings {
		summed := math.Exp(adjustedMu[i] / c)
		for q := range teamRatings {
			if teamRatings[i].Rank >= teamRatings[q].Rank {
				sumQ[q] += summed
			}
		}
	}
	return sumQ
}

// A counts the frequency of each rank among the teams.
func (m *PlackettLuce) A(teamRatings []TeamRating) []int {
	n := len(teamRatings)
	a := make([]int, n)
	for i := range teamRatings {
		count := 0
		for _, q := range teamRatings {
			if q.Rank == teamRatings[i].Rank {
				count++
			}
		}
		a[i] = count
	}
	return a
}

func (m *PlackettLuce) compute(teams [][]Rating, ranks []float64, scores []float64, weights [][]float64, margin float64) [][]Rating {
	originalTeams := teams
	teamRatings := calculateTeamRatings(teams, ranks, m.Kappa, m.Balance)
	c := m.C(teamRatings)
	sumQ := m.SumQ(teamRatings, c, scores, margin)
	a := m.A(teamRatings)

	n := len(teamRatings)
	hasScores := len(scores) == n

	result := make([][]Rating, n)
	for i, teamI := range teamRatings {
		var omega float64
		var delta float64

		adjustedMuI := teamI.Mu
		if hasScores {
			var marginAdjustment float64
			var comparisonCount int
			for j, teamJ := range teamRatings {
				if i != j {
					scoreDiff := math.Abs(scores[i] - scores[j])
					if scoreDiff > 0 {
						marginFactor := 1.0
						if scoreDiff > margin && margin > 0.0 {
							marginFactor = math.Log1p(scoreDiff / margin)
						}
						if scores[i] > scores[j] {
							marginAdjustment -= (teamI.Mu - teamJ.Mu) * (marginFactor - 1.0)
						} else {
							marginAdjustment += (teamJ.Mu - teamI.Mu) * (marginFactor - 1.0)
						}
						comparisonCount++
					}
				}
			}
			if comparisonCount > 0 {
				adjustedMuI += marginAdjustment / float64(comparisonCount)
			} else {
				adjustedMuI += marginAdjustment
			}
		}

		iMuOverC := math.Exp(adjustedMuI / c)

		for q, teamQ := range teamRatings {
			ratio := iMuOverC / sumQ[q]
			if teamQ.Rank <= teamI.Rank {
				delta += ratio * (1.0 - ratio) / float64(a[q])
				if q == i {
					omega += (1.0 - ratio) / float64(a[q])
				} else {
					omega -= ratio / float64(a[q])
				}
			}
		}

		omega *= teamI.SigmaSquared / c
		delta *= teamI.SigmaSquared / (c * c)

		var currentWeights []float64
		if len(weights) > 0 {
			currentWeights = weights[i]
		}
		gammaValue := m.Gamma(c, n, teamI.Mu, teamI.SigmaSquared, teamI.Team, teamI.Rank, currentWeights)
		delta *= gammaValue

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

// Rate computes updated skill ratings for the given teams using Plackett-Luce.
func (m *PlackettLuce) Rate(teams [][]Rating, opts ...RateOption) ([][]Rating, error) {
	var o RateOptions
	for _, opt := range opts {
		opt(&o)
	}
	return m.RateWithOptions(teams, o)
}

// RateWithOptions computes updated ratings using an explicit RateOptions structure.
func (m *PlackettLuce) RateWithOptions(teams [][]Rating, options RateOptions) ([][]Rating, error) {
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
func (m *PlackettLuce) PredictWin(teams [][]Rating) ([]float64, error) {
	return PredictWin(teams, m.Beta, m.Kappa, m.Balance)
}

// PredictDraw predicts the probability of a draw between teams.
func (m *PlackettLuce) PredictDraw(teams [][]Rating) (float64, error) {
	return PredictDraw(teams, m.Beta, m.Kappa, m.Balance)
}

// PredictRank predicts the expected rank ordering and normalized win probabilities for all teams.
func (m *PlackettLuce) PredictRank(teams [][]Rating) ([]RankProbability, error) {
	return PredictRank(teams, m.Beta, m.Kappa, m.Balance)
}

// Repr returns the representation string for PlackettLuce.
func (m *PlackettLuce) Repr() string {
	return fmt.Sprintf("PlackettLuce(mu=%v, sigma=%v)", m.Mu, m.Sigma)
}

// String returns a human-readable description of the model parameters.
func (m *PlackettLuce) String() string {
	return fmt.Sprintf("Plackett-Luce Model Parameters:\n\nmu: %v\nsigma: %v\n", m.Mu, m.Sigma)
}
