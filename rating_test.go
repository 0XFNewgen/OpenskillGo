package openskill

import (
	"encoding/json"
	"testing"
)

func TestRatingDefaults(t *testing.T) {
	model := NewPlackettLuce(WithMu(30), WithSigma(10))
	r := model.Rating(nil, nil, nil)
	if r.Mu != 30.0 {
		t.Errorf("expected mu 30, got %v", r.Mu)
	}
	if r.Sigma != 10.0 {
		t.Errorf("expected sigma 10, got %v", r.Sigma)
	}
	if r.ID == "" {
		t.Errorf("expected non-empty ID")
	}

	// Default ordinal: mu - 3 * sigma
	expectedOrdinal := 30.0 - 3.0*10.0
	if r.Ordinal() != expectedOrdinal {
		t.Errorf("expected ordinal %v, got %v", expectedOrdinal, r.Ordinal())
	}
}

func TestRatingOverrides(t *testing.T) {
	model := NewPlackettLuce(WithMu(30), WithSigma(10))

	mu40 := 40.0
	r1 := model.Rating(&mu40, nil, nil)
	if r1.Mu != 40.0 || r1.Sigma != 10.0 {
		t.Errorf("r1 mismatch: %v, %v", r1.Mu, r1.Sigma)
	}

	sig15 := 15.0
	name := "Vivek Joshy"
	r2 := model.Rating(nil, &sig15, &name)
	if r2.Mu != 30.0 || r2.Sigma != 15.0 || r2.Name != name {
		t.Errorf("r2 mismatch: %v, %v, %v", r2.Mu, r2.Sigma, r2.Name)
	}
}

func TestRatingComparison(t *testing.T) {
	model := NewPlackettLuce()
	rA := model.Rating(nil, nil, nil)
	rB := model.Rating(nil, nil, nil)

	if !rA.LessOrEqual(rB) || !rA.GreaterOrEqual(rB) {
		t.Errorf("reflexive comparison failed")
	}

	mu1, sig1 := 32.124, 1.421
	mu2, sig2 := 23.84, 3.443
	rHigh := model.Rating(&mu1, &sig1, nil)
	rLow := model.Rating(&mu2, &sig2, nil)

	if !rHigh.Greater(rLow) {
		t.Errorf("expected rHigh > rLow")
	}
	if !rLow.Less(rHigh) {
		t.Errorf("expected rLow < rHigh")
	}
	if rHigh.Equal(rLow) {
		t.Errorf("expected rHigh != rLow")
	}

	rHighClone := rHigh.Clone()
	if !rHigh.Equal(rHighClone) {
		t.Errorf("expected clone to be equal")
	}
}

func TestRatingJSON(t *testing.T) {
	r := NewRating(25.0, 8.333, "Player1")
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var parsed Rating
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if parsed.ID != r.ID || parsed.Name != r.Name || parsed.Mu != r.Mu || parsed.Sigma != r.Sigma {
		t.Errorf("json roundtrip mismatch: %+v vs %+v", parsed, r)
	}
}
