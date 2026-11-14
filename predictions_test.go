package openskill

import (
	"sort"
	"testing"
)

func TestPredictWin(t *testing.T) {
	model := NewPlackettLuce()
	r := func(mu, sigma float64) Rating {
		return NewRating(mu, sigma)
	}

	a1 := r(25.0, 25.0/3.0)
	a2 := r(32.444, 5.123)
	b1 := r(73.381, 1.421)
	b2 := r(25.188, 6.211)

	team1 := []Rating{a1, a2}
	team2 := []Rating{b1, b2}

	// 5-team match
	probs5, err := model.PredictWin([][]Rating{team1, team2, {a2}, {a1}, {b1}})
	if err != nil {
		t.Fatalf("predict win error: %v", err)
	}
	var sum5 float64
	for _, p := range probs5 {
		sum5 += p
	}
	if !approxEqual(sum5, 1.0, 1e-4) {
		t.Errorf("expected sum of probabilities 1.0, got %v", sum5)
	}

	// 2-team match
	probs2, err := model.PredictWin([][]Rating{team1, team2})
	if err != nil {
		t.Fatalf("predict win error: %v", err)
	}
	var sum2 float64
	for _, p := range probs2 {
		sum2 += p
	}
	if !approxEqual(sum2, 1.0, 1e-4) {
		t.Errorf("expected sum of probabilities 1.0, got %v", sum2)
	}

	// Error when < 2 teams
	if _, err := model.PredictWin([][]Rating{team1}); err == nil {
		t.Errorf("expected error for 1 team")
	}
}

func TestPredict5pImpostor(t *testing.T) {
	model := NewPlackettLuce()
	a1 := NewRating(25.0, 25.0/3.0)
	a2 := NewRating(32.444, 5.123)

	probs, err := model.PredictWin([][]Rating{{a1}, {a1}, {a1}, {a2}, {a1}})
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	p4 := probs[3] // team 4 has a2
	sortedProbs := make([]float64, len(probs))
	copy(sortedProbs, probs)
	sort.Float64s(sortedProbs)

	if sortedProbs[4] != p4 {
		t.Errorf("highest win probability was not the impostor player: %v vs %v", sortedProbs[4], p4)
	}
}

func TestPredictDraw(t *testing.T) {
	model := NewPlackettLuce()
	a1 := NewRating(25.0, 25.0/3.0)
	a2 := NewRating(32.444, 1.123)
	b1 := NewRating(35.881, 0.0001)
	b2 := NewRating(25.188, 0.00001)

	team1 := []Rating{a1, a2}
	team2 := []Rating{b1, b2}

	p1, err := model.PredictDraw([][]Rating{team1, team2})
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if !approxEqual(p1, 0.1919967, 1e-4) {
		t.Errorf("expected 0.1919967, got %v", p1)
	}

	p2, err := model.PredictDraw([][]Rating{team1, team2, {a1}, {a2}, {b1}})
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if !approxEqual(p2, 0.0603735, 1e-4) {
		t.Errorf("expected 0.0603735, got %v", p2)
	}

	pIdentical, err := model.PredictDraw([][]Rating{{b1}, {b1}})
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if !approxEqual(pIdentical, 0.5, 1e-4) {
		t.Errorf("expected 0.5, got %v", pIdentical)
	}

	if _, err := model.PredictDraw([][]Rating{team1}); err == nil {
		t.Errorf("expected error for 1 team")
	}
}

func TestPredictRank(t *testing.T) {
	model := NewPlackettLuce()
	a1 := NewRating(34, 0.25)
	a2 := NewRating(32, 0.25)
	a3 := NewRating(30, 0.25)

	b1 := NewRating(24, 0.5)
	b2 := NewRating(22, 0.5)
	b3 := NewRating(20, 0.5)

	team1 := []Rating{a1, b1}
	team2 := []Rating{a2, b2}
	team3 := []Rating{a3, b3}

	ranks, err := model.PredictRank([][]Rating{team1, team2, team3})
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	var totalProb float64
	for _, rp := range ranks {
		totalProb += rp.Probability
	}
	if !approxEqual(totalProb, 1.0, 1e-4) {
		t.Errorf("expected sum of probabilities 1.0, got %v", totalProb)
	}

	identicalRanks, err := model.PredictRank([][]Rating{team1, team1, team1})
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	var totalIdentical float64
	for _, rp := range identicalRanks {
		totalIdentical += rp.Probability
	}
	if !approxEqual(totalIdentical, 1.0, 1e-4) {
		t.Errorf("expected sum of probabilities 1.0, got %v", totalIdentical)
	}

	if _, err := model.PredictRank([][]Rating{team1}); err == nil {
		t.Errorf("expected error for 1 team")
	}
}

func TestWeightBounds(t *testing.T) {
	// Default bounds (1.0, 2.0)
	modelDefault := NewPlackettLuce(WithWeightBounds(1.0, 2.0))
	team1Def := []Rating{NewRating(25, 8.333), NewRating(25, 8.333), NewRating(25, 8.333)}
	team2Def := []Rating{NewRating(25, 8.333), NewRating(25, 8.333), NewRating(25, 8.333)}
	weights := [][]float64{{0.5, 1.0, 2.0}, {1.0, 1.0, 1.0}}

	resDef, err := modelDefault.Rate([][]Rating{team1Def, team2Def}, WithRanks([]float64{1, 2}), WithWeights(weights))
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	// Narrow bounds (0.9, 1.1)
	modelNarrow := NewPlackettLuce(WithWeightBounds(0.9, 1.1))
	team1Narr := []Rating{NewRating(25, 8.333), NewRating(25, 8.333), NewRating(25, 8.333)}
	team2Narr := []Rating{NewRating(25, 8.333), NewRating(25, 8.333), NewRating(25, 8.333)}

	resNarr, err := modelNarrow.Rate([][]Rating{team1Narr, team2Narr}, WithRanks([]float64{1, 2}), WithWeights(weights))
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	defaultSpread := resDef[0][2].Mu - resDef[0][0].Mu
	narrowSpread := resNarr[0][2].Mu - resNarr[0][0].Mu
	if !(narrowSpread < defaultSpread) {
		t.Errorf("narrowSpread (%v) should be less than defaultSpread (%v)", narrowSpread, defaultSpread)
	}

	// Without weight bounds
	modelNoBounds := NewPlackettLuce(WithoutWeightBounds())
	team1Raw := []Rating{NewRating(25, 8.333), NewRating(25, 8.333), NewRating(25, 8.333)}
	team2Raw := []Rating{NewRating(25, 8.333), NewRating(25, 8.333), NewRating(25, 8.333)}
	rawWeights := [][]float64{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}}

	resNoBounds, err := modelNoBounds.Rate([][]Rating{team1Raw, team2Raw}, WithRanks([]float64{1, 2}), WithWeights(rawWeights))
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	muChange0 := resNoBounds[0][0].Mu - 25.0
	muChange1 := resNoBounds[0][1].Mu - 25.0
	muChange2 := resNoBounds[0][2].Mu - 25.0
	if !approxEqual(muChange0, muChange1, 1e-4) || !approxEqual(muChange1, muChange2, 1e-4) {
		t.Errorf("mu changes should be identical with uniform raw weights: %v, %v, %v", muChange0, muChange1, muChange2)
	}
}

func TestValidationErrors(t *testing.T) {
	model := NewPlackettLuce()
	t1 := []Rating{NewRating(25, 8.333)}
	t2 := []Rating{NewRating(25, 8.333), NewRating(25, 8.333)}

	// Too few teams
	if _, err := model.Rate([][]Rating{t1}); err != ErrTooFewTeams {
		t.Errorf("expected ErrTooFewTeams, got %v", err)
	}

	// Empty team
	if _, err := model.Rate([][]Rating{t1, {}}); err != ErrEmptyTeam {
		t.Errorf("expected ErrEmptyTeam, got %v", err)
	}

	// Both ranks and scores
	if _, err := model.Rate([][]Rating{t1, t2}, WithRanks([]float64{1, 2}), WithScores([]float64{10, 5})); err != ErrBothRanksAndScores {
		t.Errorf("expected ErrBothRanksAndScores, got %v", err)
	}

	// Ranks length mismatch
	if _, err := model.Rate([][]Rating{t1, t2}, WithRanks([]float64{1})); err == nil {
		t.Errorf("expected ranks length mismatch error")
	}

	// Scores length mismatch
	if _, err := model.Rate([][]Rating{t1, t2}, WithScores([]float64{10})); err == nil {
		t.Errorf("expected scores length mismatch error")
	}

	// Weights outer length mismatch
	if _, err := model.Rate([][]Rating{t1, t2}, WithWeights([][]float64{{1}})); err == nil {
		t.Errorf("expected weights outer length mismatch error")
	}

	// Weights inner length mismatch
	if _, err := model.Rate([][]Rating{t1, t2}, WithWeights([][]float64{{1}, {1}})); err == nil {
		t.Errorf("expected weights inner length mismatch error")
	}
}

func TestConveniencePackageAPI(t *testing.T) {
	p1 := NewRating(25, 8.333)
	p2 := NewRating(25, 8.333)

	updated, err := Rate([][]Rating{{p1}, {p2}})
	if err != nil {
		t.Fatalf("Rate error: %v", err)
	}
	if len(updated) != 2 || len(updated[0]) != 1 || len(updated[1]) != 1 {
		t.Fatalf("unexpected updated structure")
	}
	if updated[0][0].Mu <= p1.Mu {
		t.Errorf("winner mu should increase")
	}
	if updated[1][0].Mu >= p2.Mu {
		t.Errorf("loser mu should decrease")
	}

	// Test default predictions
	winProbs, err := PredictWinDefault([][]Rating{{p1}, {p2}})
	if err != nil || len(winProbs) != 2 {
		t.Errorf("PredictWinDefault failed")
	}
	drawProb, err := PredictDrawDefault([][]Rating{{p1}, {p2}})
	if err != nil || drawProb <= 0 {
		t.Errorf("PredictDrawDefault failed")
	}
	rankProbs, err := PredictRankDefault([][]Rating{{p1}, {p2}})
	if err != nil || len(rankProbs) != 2 {
		t.Errorf("PredictRankDefault failed")
	}
}

func BenchmarkPlackettLuceRate(b *testing.B) {
	model := NewPlackettLuce()
	t1 := []Rating{NewRating(25, 8.333), NewRating(25, 8.333)}
	t2 := []Rating{NewRating(25, 8.333), NewRating(25, 8.333)}
	teams := [][]Rating{t1, t2}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = model.Rate(teams)
	}
}

func BenchmarkBradleyTerryFullRate(b *testing.B) {
	model := NewBradleyTerryFull()
	t1 := []Rating{NewRating(25, 8.333), NewRating(25, 8.333)}
	t2 := []Rating{NewRating(25, 8.333), NewRating(25, 8.333)}
	teams := [][]Rating{t1, t2}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = model.Rate(teams)
	}
}

func BenchmarkThurstoneMostellerFullRate(b *testing.B) {
	model := NewThurstoneMostellerFull()
	t1 := []Rating{NewRating(25, 8.333), NewRating(25, 8.333)}
	t2 := []Rating{NewRating(25, 8.333), NewRating(25, 8.333)}
	teams := [][]Rating{t1, t2}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = model.Rate(teams)
	}
}
