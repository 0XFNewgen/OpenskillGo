package openskill

import (
	"math"
	"sort"
)

// RankProbability represents the predicted placement and winning probability for a team.
type RankProbability struct {
	Rank        int     `json:"rank"`
	Probability float64 `json:"probability"`
}

// PredictWin calculates the probability of each team winning the match.
// Generalized from Ibstedt et al. (2019) to asymmetric multi-team games.
func PredictWin(teams [][]Rating, beta, kappa float64, balance bool) ([]float64, error) {
	if err := checkTeams(teams); err != nil {
		return nil, err
	}
	n := len(teams)
	teamRatings := calculateTeamRatings(teams, nil, kappa, balance)

	if n == 2 {
		a := teamRatings[0]
		b := teamRatings[1]
		denom := math.Sqrt(2*beta*beta + a.SigmaSquared + b.SigmaSquared)
		res := PhiMajor((a.Mu - b.Mu) / denom)
		return []float64{res, 1.0 - res}, nil
	}

	winProbabilities := make([]float64, n)
	for i := 0; i < n; i++ {
		var sumP float64
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			denom := math.Sqrt(2*beta*beta + teamRatings[i].SigmaSquared + teamRatings[j].SigmaSquared)
			sumP += PhiMajor((teamRatings[i].Mu - teamRatings[j].Mu) / denom)
		}
		winProbabilities[i] = sumP / float64(n-1)
	}

	var total float64
	for _, p := range winProbabilities {
		total += p
	}
	normalized := make([]float64, n)
	for i, p := range winProbabilities {
		normalized[i] = p / total
	}
	return normalized, nil
}

// PredictDraw calculates the probability of a draw among the given teams.
func PredictDraw(teams [][]Rating, beta, kappa float64, balance bool) (float64, error) {
	if err := checkTeams(teams); err != nil {
		return 0, err
	}
	n := len(teams)
	totalPlayerCount := 0
	for _, team := range teams {
		totalPlayerCount += len(team)
	}

	drawProbability := 1.0 / float64(totalPlayerCount)
	invCdf, err := PhiMajorInverse((1.0 + drawProbability) / 2.0)
	if err != nil {
		return 0, err
	}
	drawMargin := math.Sqrt(float64(totalPlayerCount)) * beta * invCdf

	teamRatings := calculateTeamRatings(teams, nil, kappa, balance)
	var pairwiseProbabilities []float64

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			muA := teamRatings[i].Mu
			sigmaA := teamRatings[i].SigmaSquared
			muB := teamRatings[j].Mu
			sigmaB := teamRatings[j].SigmaSquared
			denom := math.Sqrt(2*beta*beta + sigmaA + sigmaB)

			p := PhiMajor((drawMargin-muA+muB)/denom) - PhiMajor((muB-muA-drawMargin)/denom)
			pairwiseProbabilities = append(pairwiseProbabilities, p)
		}
	}

	if len(pairwiseProbabilities) == 0 {
		return 0, nil
	}

	var sum float64
	for _, p := range pairwiseProbabilities {
		sum += p
	}
	return sum / float64(len(pairwiseProbabilities)), nil
}

// PredictRank predicts the expected rank ordering and normalized win probabilities for all teams.
func PredictRank(teams [][]Rating, beta, kappa float64, balance bool) ([]RankProbability, error) {
	if err := checkTeams(teams); err != nil {
		return nil, err
	}
	n := len(teams)
	teamRatings := calculateTeamRatings(teams, nil, kappa, balance)

	winProbabilities := make([]float64, n)
	for i := 0; i < n; i++ {
		var sumP float64
		for j := 0; j < n; j++ {
			if i != j {
				denom := math.Sqrt(2*beta*beta + teamRatings[i].SigmaSquared + teamRatings[j].SigmaSquared)
				sumP += PhiMajor((teamRatings[i].Mu - teamRatings[j].Mu) / denom)
			}
		}
		winProbabilities[i] = sumP / float64(n-1)
	}

	var total float64
	for _, p := range winProbabilities {
		total += p
	}
	normalized := make([]float64, n)
	for i, p := range winProbabilities {
		normalized[i] = p / total
	}

	type teamProb struct {
		index int
		prob  float64
	}
	sortedTeams := make([]teamProb, n)
	for i, p := range normalized {
		sortedTeams[i] = teamProb{index: i, prob: p}
	}

	sort.SliceStable(sortedTeams, func(i, j int) bool {
		return sortedTeams[i].prob > sortedTeams[j].prob
	})

	ranks := make([]int, n)
	currentRank := 1
	for i := 0; i < n; i++ {
		if i > 0 && sortedTeams[i].prob < sortedTeams[i-1].prob {
			currentRank = i + 1
		}
		ranks[sortedTeams[i].index] = currentRank
	}

	result := make([]RankProbability, n)
	for i := 0; i < n; i++ {
		result[i] = RankProbability{
			Rank:        ranks[i],
			Probability: normalized[i],
		}
	}
	return result, nil
}
