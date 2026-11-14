package openskill

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

type expectedPlayerData struct {
	Mu    float64 `json:"mu"`
	Sigma float64 `json:"sigma"`
}

type modelTestFixture struct {
	Model struct {
		Mu    float64 `json:"mu"`
		Sigma float64 `json:"sigma"`
	} `json:"model"`
	Normal     map[string][]expectedPlayerData `json:"normal"`
	Ranks      map[string][]expectedPlayerData `json:"ranks"`
	Scores     map[string][]expectedPlayerData `json:"scores"`
	Margins    map[string][]expectedPlayerData `json:"margins"`
	LimitSigma map[string][]expectedPlayerData `json:"limit_sigma"`
	Ties       map[string][]expectedPlayerData `json:"ties"`
	Weights    map[string][]expectedPlayerData `json:"weights"`
	Balance    map[string][]expectedPlayerData `json:"balance"`
}

func loadFixture(t *testing.T, filename string) modelTestFixture {
	t.Helper()
	data, err := os.ReadFile("testdata/" + filename)
	if err != nil {
		t.Fatalf("failed to read fixture %s: %v", filename, err)
	}
	var fix modelTestFixture
	if err := json.Unmarshal(data, &fix); err != nil {
		t.Fatalf("failed to parse fixture %s: %v", filename, err)
	}
	return fix
}

func checkExpected(t *testing.T, sectionName string, expected map[string][]expectedPlayerData, actual [][]Rating) {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("[%s] team count mismatch: got %d, expected %d", sectionName, len(actual), len(expected))
	}
	for teamIdx, team := range actual {
		teamKey := fmt.Sprintf("team_%d", teamIdx+1)
		expTeam, ok := expected[teamKey]
		if !ok {
			t.Fatalf("[%s] key %s not found in expected", sectionName, teamKey)
		}
		if len(team) != len(expTeam) {
			t.Fatalf("[%s] team %s size mismatch: got %d, expected %d", sectionName, teamKey, len(team), len(expTeam))
		}
		for pIdx, player := range team {
			expP := expTeam[pIdx]
			if !approxEqual(player.Mu, expP.Mu, 1e-4) {
				t.Errorf("[%s] %s player %d Mu mismatch: got %v, expected %v (diff: %v)",
					sectionName, teamKey, pIdx, player.Mu, expP.Mu, player.Mu-expP.Mu)
			}
			if !approxEqual(player.Sigma, expP.Sigma, 1e-4) {
				t.Errorf("[%s] %s player %d Sigma mismatch: got %v, expected %v (diff: %v)",
					sectionName, teamKey, pIdx, player.Sigma, expP.Sigma, player.Sigma-expP.Sigma)
			}
		}
	}
}

func testModelFixture(t *testing.T, modelName string, fixtureFile string, factory func(mu, sigma float64, opts ...ModelOption) Model) {
	fix := loadFixture(t, fixtureFile)
	mu := fix.Model.Mu
	sigma := fix.Model.Sigma

	r := func() Rating {
		return NewRating(mu, sigma)
	}

	// 1. Normal
	{
		m := factory(mu, sigma)
		team1 := []Rating{r()}
		team2 := []Rating{r(), r()}
		res, err := m.Rate([][]Rating{team1, team2})
		if err != nil {
			t.Fatalf("[%s normal] error: %v", modelName, err)
		}
		checkExpected(t, modelName+" normal", fix.Normal, res)
	}

	// 2. Ranks
	{
		m := factory(mu, sigma)
		team1 := []Rating{r()}
		team2 := []Rating{r(), r()}
		team3 := []Rating{r()}
		team4 := []Rating{r(), r()}
		res, err := m.Rate([][]Rating{team1, team2, team3, team4}, WithRanks([]float64{2, 1, 4, 3}))
		if err != nil {
			t.Fatalf("[%s ranks] error: %v", modelName, err)
		}
		checkExpected(t, modelName+" ranks", fix.Ranks, res)
	}

	// 3. Scores
	{
		m := factory(mu, sigma)
		team1 := []Rating{r()}
		team2 := []Rating{r(), r()}
		res, err := m.Rate([][]Rating{team1, team2}, WithScores([]float64{1, 2}))
		if err != nil {
			t.Fatalf("[%s scores] error: %v", modelName, err)
		}
		checkExpected(t, modelName+" scores", fix.Scores, res)
	}

	// 4. Margins
	{
		m := factory(mu, sigma, WithMargin(2.0))
		game := [][]Rating{
			{r(), r()},
			{r(), r()},
			{r(), r()},
			{r(), r()},
			{r(), r()},
		}
		res, err := m.Rate(game,
			WithScores([]float64{10, 5, 5, 2, 1}),
			WithWeights([][]float64{
				{1, 2},
				{2, 1},
				{1, 2},
				{3, 1},
				{1, 2},
			}),
		)
		if err != nil {
			t.Fatalf("[%s margins] error: %v", modelName, err)
		}
		checkExpected(t, modelName+" margins", fix.Margins, res)
	}

	// 5. Limit Sigma
	{
		m := factory(mu, sigma)
		team1 := []Rating{r()}
		team2 := []Rating{r(), r()}
		team3 := []Rating{r(), r(), r()}
		res, err := m.Rate([][]Rating{team1, team2, team3},
			WithRanks([]float64{2, 1, 3}),
			WithRateLimitSigma(true),
		)
		if err != nil {
			t.Fatalf("[%s limit_sigma] error: %v", modelName, err)
		}
		checkExpected(t, modelName+" limit_sigma", fix.LimitSigma, res)
	}

	// 6. Ties
	{
		m := factory(mu, sigma)
		team1 := []Rating{r()}
		team2 := []Rating{r(), r()}
		team3 := []Rating{r(), r(), r()}
		res, err := m.Rate([][]Rating{team1, team2, team3}, WithRanks([]float64{1, 2, 1}))
		if err != nil {
			t.Fatalf("[%s ties] error: %v", modelName, err)
		}
		checkExpected(t, modelName+" ties", fix.Ties, res)
	}

	// 7. Weights
	{
		m := factory(mu, sigma)
		team1 := []Rating{r(), r(), r()}
		team2 := []Rating{r(), r()}
		team3 := []Rating{r(), r(), r()}
		team4 := []Rating{r(), r()}
		res, err := m.Rate([][]Rating{team1, team2, team3, team4},
			WithRanks([]float64{2, 1, 4, 3}),
			WithWeights([][]float64{
				{2, 0, 0},
				{1, 2},
				{0, 0, 1},
				{0, 1},
			}),
		)
		if err != nil {
			t.Fatalf("[%s weights] error: %v", modelName, err)
		}
		checkExpected(t, modelName+" weights", fix.Weights, res)
	}

	// 8. Balance
	{
		m := factory(mu, sigma, WithBalance(true))
		team1 := []Rating{r(), r()}
		team2 := []Rating{r(), r()}
		res, err := m.Rate([][]Rating{team1, team2}, WithRanks([]float64{1, 2}))
		if err != nil {
			t.Fatalf("[%s balance] error: %v", modelName, err)
		}
		checkExpected(t, modelName+" balance", fix.Balance, res)
	}
}

func TestPlackettLuceFixture(t *testing.T) {
	testModelFixture(t, "PlackettLuce", "plackettluce.json", func(mu, sigma float64, opts ...ModelOption) Model {
		allOpts := append([]ModelOption{WithMu(mu), WithSigma(sigma)}, opts...)
		return NewPlackettLuce(allOpts...)
	})
}

func TestBradleyTerryFullFixture(t *testing.T) {
	testModelFixture(t, "BradleyTerryFull", "bradleyterryfull.json", func(mu, sigma float64, opts ...ModelOption) Model {
		allOpts := append([]ModelOption{WithMu(mu), WithSigma(sigma)}, opts...)
		return NewBradleyTerryFull(allOpts...)
	})
}

func TestBradleyTerryPartFixture(t *testing.T) {
	testModelFixture(t, "BradleyTerryPart", "bradleyterrypart.json", func(mu, sigma float64, opts ...ModelOption) Model {
		allOpts := append([]ModelOption{WithMu(mu), WithSigma(sigma)}, opts...)
		return NewBradleyTerryPart(allOpts...)
	})
}

func TestThurstoneMostellerFullFixture(t *testing.T) {
	testModelFixture(t, "ThurstoneMostellerFull", "thurstonemostellerfull.json", func(mu, sigma float64, opts ...ModelOption) Model {
		allOpts := append([]ModelOption{WithMu(mu), WithSigma(sigma)}, opts...)
		return NewThurstoneMostellerFull(allOpts...)
	})
}

func TestThurstoneMostellerPartFixture(t *testing.T) {
	testModelFixture(t, "ThurstoneMostellerPart", "thurstonemostellerpart.json", func(mu, sigma float64, opts ...ModelOption) Model {
		allOpts := append([]ModelOption{WithMu(mu), WithSigma(sigma)}, opts...)
		return NewThurstoneMostellerPart(allOpts...)
	})
}
