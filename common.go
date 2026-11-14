package openskill

import (
	"fmt"
	"math"
	"sort"
)

func mathSqrt(x float64) float64 {
	return math.Sqrt(x)
}

// TeamRating represents the collective rating of a team during a match.
type TeamRating struct {
	Mu           float64
	SigmaSquared float64
	Team         []Rating
	Rank         int
}

func checkTeams(teams [][]Rating) error {
	if len(teams) < 2 {
		return ErrTooFewTeams
	}
	for _, team := range teams {
		if len(team) < 1 {
			return ErrEmptyTeam
		}
	}
	return nil
}

func validateRateInputs(teams [][]Rating, opts RateOptions) error {
	if err := checkTeams(teams); err != nil {
		return err
	}
	if len(opts.Ranks) > 0 && len(opts.Scores) > 0 {
		return ErrBothRanksAndScores
	}
	if len(opts.Ranks) > 0 && len(opts.Ranks) != len(teams) {
		return &RanksLengthError{RanksLen: len(opts.Ranks), TeamsLen: len(teams)}
	}
	if len(opts.Scores) > 0 && len(opts.Scores) != len(teams) {
		return &ScoresLengthError{ScoresLen: len(opts.Scores), TeamsLen: len(teams)}
	}
	if len(opts.Weights) > 0 {
		if len(opts.Weights) != len(teams) {
			return &WeightsLengthError{
				Message: fmt.Sprintf("argument 'weights' must have the same number of elements as 'teams', not %d", len(opts.Weights)),
			}
		}
		for i, teamWeights := range opts.Weights {
			if len(teamWeights) != len(teams[i]) {
				return &WeightsLengthError{
					Message: fmt.Sprintf("argument 'weights' must have the same number of elements as each team in 'teams', not %d", len(teamWeights)),
				}
			}
		}
	}
	return nil
}

// CalculateRankings assigns a 0-based rank to each team based on their score/rank value.
// The team with the lowest score value gets the first rank (0). Tied values receive identical ranks.
func CalculateRankings(numTeams int, ranks []float64) []float64 {
	if numTeams == 0 {
		return []float64{}
	}

	teamScores := make([]float64, numTeams)
	if len(ranks) == numTeams {
		copy(teamScores, ranks)
	} else {
		for i := 0; i < numTeams; i++ {
			teamScores[i] = float64(i)
		}
	}

	sortedScores := make([]float64, numTeams)
	copy(sortedScores, teamScores)
	sort.Float64s(sortedScores)

	rankMap := make(map[float64]int)
	for idx, val := range sortedScores {
		if _, exists := rankMap[val]; !exists {
			rankMap[val] = idx
		}
	}

	result := make([]float64, numTeams)
	for i, v := range teamScores {
		result[i] = float64(rankMap[v])
	}
	return result
}

// calculateTeamRatings computes team aggregate mu and sigma_squared, taking balance into account.
func calculateTeamRatings(game [][]Rating, ranks []float64, kappa float64, balance bool) []TeamRating {
	var calcRanks []float64
	if len(ranks) > 0 {
		calcRanks = CalculateRankings(len(game), ranks)
	} else {
		calcRanks = CalculateRankings(len(game), nil)
	}

	result := make([]TeamRating, len(game))
	for index, team := range game {
		maxOrdinal := 0.0
		if len(team) > 0 {
			maxOrdinal = team[0].Ordinal()
			for _, p := range team[1:] {
				ord := p.Ordinal()
				if ord > maxOrdinal {
					maxOrdinal = ord
				}
			}
		}

		var muSummed float64
		var sigmaSquaredSummed float64

		for _, player := range team {
			balanceWeight := 1.0
			if balance {
				ordinalDiff := maxOrdinal - player.Ordinal()
				balanceWeight = 1.0 + (ordinalDiff / (maxOrdinal + kappa))
			}
			muSummed += player.Mu * balanceWeight
			wSigma := player.Sigma * balanceWeight
			sigmaSquaredSummed += wSigma * wSigma
		}

		result[index] = TeamRating{
			Mu:           muSummed,
			SigmaSquared: sigmaSquaredSummed,
			Team:         team,
			Rank:         int(calcRanks[index]),
		}
	}
	return result
}

type unwoundItem[T any] struct {
	item      T
	origIndex int
	tenet     float64
}

// Unwind performs a stable sort of objects based on tenets, returning the sorted objects and the tenets (original indices).
func Unwind[T any](tenets []float64, objects []T) ([]T, []float64) {
	if len(objects) == 0 {
		return []T{}, []float64{}
	}

	items := make([]unwoundItem[T], len(objects))
	for i := range objects {
		items[i] = unwoundItem[T]{
			item:      objects[i],
			origIndex: i,
			tenet:     tenets[i],
		}
	}

	sort.SliceStable(items, func(i, j int) bool {
		return items[i].tenet < items[j].tenet
	})

	sortedObjects := make([]T, len(objects))
	returnedTenets := make([]float64, len(objects))
	for i, it := range items {
		sortedObjects[i] = it.item
		returnedTenets[i] = float64(it.origIndex)
	}
	return sortedObjects, returnedTenets
}

func cloneTeams(teams [][]Rating) [][]Rating {
	copied := make([][]Rating, len(teams))
	for i, team := range teams {
		copied[i] = make([]Rating, len(team))
		for j, player := range team {
			copied[i][j] = player.Clone()
		}
	}
	return copied
}

func copyWeights(weights [][]float64) [][]float64 {
	if weights == nil {
		return nil
	}
	cp := make([][]float64, len(weights))
	for i, w := range weights {
		cp[i] = make([]float64, len(w))
		copy(cp[i], w)
	}
	return cp
}
