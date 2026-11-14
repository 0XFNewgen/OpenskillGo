package openskill

import (
	"reflect"
	"testing"
)

func TestCalculateRankings(t *testing.T) {
	if len(CalculateRankings(0, nil)) != 0 {
		t.Errorf("expected empty")
	}

	r1 := CalculateRankings(4, nil)
	if !reflect.DeepEqual(r1, []float64{0, 1, 2, 3}) {
		t.Errorf("r1 failed: %v", r1)
	}

	r2 := CalculateRankings(2, []float64{0, 0})
	if !reflect.DeepEqual(r2, []float64{0, 0}) {
		t.Errorf("r2 failed: %v", r2)
	}

	r3 := CalculateRankings(4, []float64{1, 2, 3, 4})
	if !reflect.DeepEqual(r3, []float64{0, 1, 2, 3}) {
		t.Errorf("r3 failed: %v", r3)
	}

	r4 := CalculateRankings(4, []float64{1, 1, 3, 4})
	if !reflect.DeepEqual(r4, []float64{0, 0, 2, 3}) {
		t.Errorf("r4 failed: %v", r4)
	}

	r5 := CalculateRankings(4, []float64{1, 2, 3, 3})
	if !reflect.DeepEqual(r5, []float64{0, 1, 2, 2}) {
		t.Errorf("r5 failed: %v", r5)
	}

	r6 := CalculateRankings(4, []float64{1, 2, 2, 4})
	if !reflect.DeepEqual(r6, []float64{0, 1, 1, 3}) {
		t.Errorf("r6 failed: %v", r6)
	}

	r7 := CalculateRankings(5, []float64{14, 32, 47, 47, 48})
	if !reflect.DeepEqual(r7, []float64{0, 1, 2, 2, 4}) {
		t.Errorf("r7 failed: %v", r7)
	}
}

func TestCalculateTeamRatings(t *testing.T) {
	model := NewPlackettLuce()
	team1 := []Rating{model.Rating(nil, nil, nil)}
	team2 := []Rating{model.Rating(nil, nil, nil), model.Rating(nil, nil, nil)}

	res := calculateTeamRatings([][]Rating{team1, team2}, nil, model.Kappa, model.Balance)
	if !approxEqual(res[0].Mu, 25.0, 1e-4) {
		t.Errorf("team 1 mu failed: %v", res[0].Mu)
	}
	if !approxEqual(res[1].Mu, 50.0, 1e-4) {
		t.Errorf("team 2 mu failed: %v", res[1].Mu)
	}
	if !approxEqual(res[0].SigmaSquared, 69.44444, 1e-4) {
		t.Errorf("team 1 sigma_sq failed: %v", res[0].SigmaSquared)
	}
	if !approxEqual(res[1].SigmaSquared, 138.88888, 1e-4) {
		t.Errorf("team 2 sigma_sq failed: %v", res[1].SigmaSquared)
	}
	if res[0].Rank != 0 || res[1].Rank != 1 {
		t.Errorf("ranks failed: %v, %v", res[0].Rank, res[1].Rank)
	}
}

func TestUnwind(t *testing.T) {
	out0, tenet0 := Unwind[string]([]float64{}, []string{})
	if len(out0) != 0 || len(tenet0) != 0 {
		t.Errorf("empty failed")
	}

	src1 := []string{"a"}
	out1, tenet1 := Unwind([]float64{0}, src1)
	if !reflect.DeepEqual(out1, []string{"a"}) || !reflect.DeepEqual(tenet1, []float64{0}) {
		t.Errorf("1 item failed")
	}

	src2 := []string{"b", "a"}
	out2, tenet2 := Unwind([]float64{1, 0}, src2)
	if !reflect.DeepEqual(out2, []string{"a", "b"}) || !reflect.DeepEqual(tenet2, []float64{1, 0}) {
		t.Errorf("2 items failed")
	}

	src3 := []string{"b", "c", "a"}
	out3, tenet3 := Unwind([]float64{1, 2, 0}, src3)
	if !reflect.DeepEqual(out3, []string{"a", "b", "c"}) || !reflect.DeepEqual(tenet3, []float64{2, 0, 1}) {
		t.Errorf("3 items failed")
	}

	src4 := []string{"b", "d", "c", "a"}
	out4, tenet4 := Unwind([]float64{1, 3, 2, 0}, src4)
	if !reflect.DeepEqual(out4, []string{"a", "b", "c", "d"}) || !reflect.DeepEqual(tenet4, []float64{3, 0, 2, 1}) {
		t.Errorf("4 items failed")
	}

	// Can undo the ranking:
	restored, _ := Unwind(tenet4, out4)
	if !reflect.DeepEqual(restored, src4) {
		t.Errorf("restore failed: %v vs %v", restored, src4)
	}
}
