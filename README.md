# 0XFOpenskill

[![Go Version](https://img.shields.io/badge/go-1.26.4-blue.svg)](https://golang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**0XFOpenskill** is a fast, pure Go implementation of the Bayesian multiplayer and multi-team skill rating system [openskill.py](https://github.com/vivekjoshy/openskill.py) (v6.2.0), based on the Weng-Lin (2011) Bayesian approximation models comparable to Microsoft TrueSkill™.

Designed for zero external dependencies, high numerical precision, and extreme throughput (>650,000 matches rated/second on a single CPU core).

---

## Features

- **Zero External Dependencies**: Pure Go using standard library (`math`, `crypto/rand`).
- **All 5 Weng-Lin Models Supported**:
  - `PlackettLuce`: Multidimensional abilities, default and fastest for asymmetric multi-team games.
  - `BradleyTerryFull`: Pairwise logistic comparison across all teams.
  - `BradleyTerryPart`: Pairwise logistic comparison within a sliding window of ranks for large ladders.
  - `ThurstoneMostellerFull`: Normal cumulative distribution modeling with full pairings.
  - `ThurstoneMostellerPart`: Sliding-window pairing with Gaussian distribution models.
- **Advanced Match Configuration**:
  - Placements via `WithRanks` (1 = 1st place, etc.) or `WithScores` (higher is better).
  - Victory margins (`WithMargin`) via logarithmic score difference scaling.
  - Individual player contribution weights (`WithWeights`).
  - Additive dynamics variance parameter (`WithTau`).
  - Standard deviation upper bounding (`WithLimitSigma`).
  - Outlier handling (`WithBalance`).
- **Predictive Analytics**:
  - `PredictWin`: Pairwise win probabilities generalized to $N$ teams.
  - `PredictDraw`: Draw probability calculation.
  - `PredictRank`: Predicted placement order with win probabilities.
- **100% Test Coverage & Accuracy**:
  - Verified against original `openskill.py` JSON test fixtures (`plackettluce.json`, `bradleyterryfull.json`, etc.).
  - Peter J. Acklam's inverse normal CDF algorithm with Halley refinement achieving machine precision ($10^{-16}$).

---

## Installation

```bash
go get 0XFOpenskill
```

*(Or copy into your internal `pkg/` or vendor directory)*

---

## Quick Start

### Basic 1v1 Rating Update

```go
package main

import (
	"fmt"
	"0XFOpenskill"
)

func main() {
	// Create two players with default rating (mu = 25.0, sigma = 8.333)
	playerA := openskill.NewRating(25.0, 25.0/3.0, "PlayerA")
	playerB := openskill.NewRating(25.0, 25.0/3.0, "PlayerB")

	// Player A beats Player B (Team 1 at index 0 won)
	teams := [][]openskill.Rating{{playerA}, {playerB}}
	updated, err := openskill.Rate(teams)
	if err != nil {
		panic(err)
	}

	newA := updated[0][0]
	newB := updated[1][0]

	fmt.Printf("Winner (A): Mu=%.3f, Sigma=%.3f, Ordinal=%.3f\n", newA.Mu, newA.Sigma, newA.Ordinal())
	fmt.Printf("Loser  (B): Mu=%.3f, Sigma=%.3f, Ordinal=%.3f\n", newB.Mu, newB.Sigma, newB.Ordinal())
}
```

### Multiplayer & Multi-Team (e.g. 3v3 or 4 Teams)

```go
model := openskill.NewPlackettLuce()

team1 := []openskill.Rating{model.Rating(nil, nil, nil), model.Rating(nil, nil, nil)}
team2 := []openskill.Rating{model.Rating(nil, nil, nil), model.Rating(nil, nil, nil)}
team3 := []openskill.Rating{model.Rating(nil, nil, nil), model.Rating(nil, nil, nil)}

// Team 2 won (rank 1), Team 1 took 2nd (rank 2), Team 3 took 3rd (rank 3)
updated, err := model.Rate(
	[][]openskill.Rating{team1, team2, team3},
	openskill.WithRanks([]float64{2, 1, 3}),
)
```

### Match Scores with Victory Margin

```go
model := openskill.NewPlackettLuce(openskill.WithMargin(2.0))

// Scores: Team 1 scored 10, Team 2 scored 5, Team 3 scored 2
updated, err := model.Rate(
	[][]openskill.Rating{team1, team2, team3},
	openskill.WithScores([]float64{10, 5, 2}),
)
```

### Individual Player Contribution Weights

```go
// Team 1 player 0 did 70% of work, player 1 did 30%
weights := [][]float64{
	{2.0, 1.0},
	{1.0, 1.0},
}

updated, err := model.Rate(
	[][]openskill.Rating{team1, team2},
	openskill.WithWeights(weights),
)
```

### Match Predictions

```go
model := openskill.NewPlackettLuce()

// Predict win probabilities
winOdds, _ := model.PredictWin([][]openskill.Rating{team1, team2})
fmt.Printf("Team 1 Win: %.1f%%, Team 2 Win: %.1f%%\n", winOdds[0]*100, winOdds[1]*100)

// Predict draw probability
drawProb, _ := model.PredictDraw([][]openskill.Rating{team1, team2})
fmt.Printf("Draw probability: %.1f%%\n", drawProb*100)

// Predict ranks
predictedRanks, _ := model.PredictRank([][]openskill.Rating{team1, team2, team3})
for i, r := range predictedRanks {
    fmt.Printf("Team %d: Expected Rank %d (Prob: %.2f%%)\n", i+1, r.Rank, r.Probability*100)
}
```

---

## Choosing a Model

| Model | Time Complexity | Best Suited For |
|---|---|---|
| `PlackettLuce` | $\mathcal{O}(n)$ | Default choice; team games, fast matchmaking, general multiplayer |
| `BradleyTerryFull` | $\mathcal{O}(n^2)$ | All-vs-all round-robin style games |
| `BradleyTerryPart` | $\mathcal{O}(n \cdot w)$ | Large tournaments with windowed ladder matching |
| `ThurstoneMostellerFull` | $\mathcal{O}(n^2)$ | Games with strong draw probabilities and normal distribution priors |
| `ThurstoneMostellerPart` | $\mathcal{O}(n \cdot w)$ | Large ladders with draws |

All models implement the `openskill.Model` interface:

```go
type Model interface {
	Rate(teams [][]Rating, opts ...RateOption) ([][]Rating, error)
	RateWithOptions(teams [][]Rating, options RateOptions) ([][]Rating, error)
	PredictWin(teams [][]Rating) ([]float64, error)
	PredictDraw(teams [][]Rating) (float64, error)
	PredictRank(teams [][]Rating) ([]RankProbability, error)
}
```

---

## Benchmarks

Benchmark performed on AMD Ryzen 5 (single core):

```
BenchmarkPlackettLuceRate-12              	 816,342 ops	  1,519 ns/op  (~658,000 matches/sec)
BenchmarkBradleyTerryFullRate-12          	1,000,000 ops	  1,788 ns/op  (~560,000 matches/sec)
BenchmarkThurstoneMostellerFullRate-12    	1,000,000 ops	  1,783 ns/op  (~560,000 matches/sec)
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
Based on the research paper by Ruby C. Weng and Chih-Jen Lin (2011) and [openskill.py](https://github.com/vivekjoshy/openskill.py) by Vivek Joshy.
