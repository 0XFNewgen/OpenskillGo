package openskill

import (
	"math"
	"reflect"
	"testing"
)

const floatTolerance = 1e-5

func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestUnaryMinus(t *testing.T) {
	if UnaryMinus(1) != -1 {
		t.Errorf("expected -1, got %v", UnaryMinus(1))
	}
	if UnaryMinus(-1) != 1 {
		t.Errorf("expected 1, got %v", UnaryMinus(-1))
	}
	if UnaryMinus(0) != 0 {
		t.Errorf("expected 0, got %v", UnaryMinus(0))
	}
	if UnaryMinus(1.0) != -1.0 {
		t.Errorf("expected -1.0, got %v", UnaryMinus(1.0))
	}
	if UnaryMinus(-1.0) != 1.0 {
		t.Errorf("expected 1.0, got %v", UnaryMinus(-1.0))
	}
}

func TestMatrixTranspose(t *testing.T) {
	m1 := [][]int{{1, 2, 3}}
	e1 := [][]int{{1}, {2}, {3}}
	if !reflect.DeepEqual(MatrixTranspose(m1), e1) {
		t.Errorf("m1 failed")
	}

	m2 := [][]int{{1, 2, 3}, {4, 5, 6}}
	e2 := [][]int{{1, 4}, {2, 5}, {3, 6}}
	if !reflect.DeepEqual(MatrixTranspose(m2), e2) {
		t.Errorf("m2 failed")
	}

	m3 := [][]int{{1, 2}, {3, 4}}
	e3 := [][]int{{1, 3}, {2, 4}}
	if !reflect.DeepEqual(MatrixTranspose(m3), e3) {
		t.Errorf("m3 failed")
	}
}

func TestNormalize(t *testing.T) {
	checkSlice := func(got, expected []float64) {
		t.Helper()
		if len(got) != len(expected) {
			t.Fatalf("length mismatch: got %d, expected %d", len(got), len(expected))
		}
		for i := range got {
			if !approxEqual(got[i], expected[i], 1e-7) {
				t.Errorf("idx %d: got %v, expected %v", i, got[i], expected[i])
			}
		}
	}

	checkSlice(Normalize([]float64{1, 2, 3}, 0, 1), []float64{0.0, 0.5, 1.0})
	checkSlice(Normalize([]float64{1, 2, 3}, 0, 100), []float64{0.0, 50.0, 100.0})
	checkSlice(Normalize([]float64{1, 2, 3}, 0, 10), []float64{0.0, 5.0, 10.0})
	checkSlice(Normalize([]float64{1, 2, 3}, 1, 0), []float64{1.0, 0.5, 0.0})
	checkSlice(Normalize([]float64{1, 1, 1}, 0, 1), []float64{0.0, 0.0, 0.0})
}

func TestV(t *testing.T) {
	if !approxEqual(V(1, 2), 1.525135276160981, floatTolerance) {
		t.Errorf("V(1, 2) failed: %v", V(1, 2))
	}
	if !approxEqual(V(0, 2), 2.373215532822843, floatTolerance) {
		t.Errorf("V(0, 2) failed: %v", V(0, 2))
	}
	if !approxEqual(V(0, -1), 0.2875999709391784, floatTolerance) {
		t.Errorf("V(0, -1) failed: %v", V(0, -1))
	}
	if !approxEqual(V(0, 10), 10, floatTolerance) {
		t.Errorf("V(0, 10) failed: %v", V(0, 10))
	}
}

func TestVt(t *testing.T) {
	if !approxEqual(Vt(-1000, -100), 1100, floatTolerance) {
		t.Errorf("Vt(-1000, -100) failed: %v", Vt(-1000, -100))
	}
	if !approxEqual(Vt(1000, -100), -1100, floatTolerance) {
		t.Errorf("Vt(1000, -100) failed: %v", Vt(1000, -100))
	}
	if !approxEqual(Vt(-1000, 1000), 0.79788, floatTolerance) {
		t.Errorf("Vt(-1000, 1000) failed: %v", Vt(-1000, 1000))
	}
	if !approxEqual(Vt(0, 1000), 0, floatTolerance) {
		t.Errorf("Vt(0, 1000) failed: %v", Vt(0, 1000))
	}
}

func TestW(t *testing.T) {
	if !approxEqual(W(1, 2), 0.800902334429651, floatTolerance) {
		t.Errorf("W(1, 2) failed: %v", W(1, 2))
	}
	if !approxEqual(W(0, 2), 0.885720899585924, floatTolerance) {
		t.Errorf("W(0, 2) failed: %v", W(0, 2))
	}
	if !approxEqual(W(0, -1), 0.3703137142233946, floatTolerance) {
		t.Errorf("W(0, -1) failed: %v", W(0, -1))
	}
	if !approxEqual(W(0, 10), 0, floatTolerance) {
		t.Errorf("W(0, 10) failed: %v", W(0, 10))
	}
	if !approxEqual(W(-1, 10), 1, floatTolerance) {
		t.Errorf("W(-1, 10) failed: %v", W(-1, 10))
	}
}

func TestWt(t *testing.T) {
	if !approxEqual(Wt(1, 2), 0.3838582646421707, floatTolerance) {
		t.Errorf("Wt(1, 2) failed: %v", Wt(1, 2))
	}
	if !approxEqual(Wt(0, 2), 0.2262586964500768, floatTolerance) {
		t.Errorf("Wt(0, 2) failed: %v", Wt(0, 2))
	}
	if !approxEqual(Wt(0, -1), 1, floatTolerance) {
		t.Errorf("Wt(0, -1) failed: %v", Wt(0, -1))
	}
	if !approxEqual(Wt(0, 0), 1.0, floatTolerance) {
		t.Errorf("Wt(0, 0) failed: %v", Wt(0, 0))
	}
	if !approxEqual(Wt(0, 10), 0, floatTolerance) {
		t.Errorf("Wt(0, 10) failed: %v", Wt(0, 10))
	}
}

func TestLadderPairs(t *testing.T) {
	r0 := LadderPairs([]int{})
	if len(r0) != 1 || len(r0[0]) != 0 {
		t.Errorf("expected [[]], got %v", r0)
	}

	r1 := LadderPairs([]int{1})
	if len(r1) != 1 || len(r1[0]) != 0 {
		t.Errorf("expected [[]], got %v", r1)
	}

	r2 := LadderPairs([]int{1, 2})
	if !reflect.DeepEqual(r2, [][]int{{2}, {1}}) {
		t.Errorf("expected [[2], [1]], got %v", r2)
	}

	r3 := LadderPairs([]int{1, 2, 3})
	if !reflect.DeepEqual(r3, [][]int{{2}, {1, 3}, {2}}) {
		t.Errorf("expected [[2], [1, 3], [2]], got %v", r3)
	}

	r4 := LadderPairs([]int{1, 2, 3, 4})
	if !reflect.DeepEqual(r4, [][]int{{2}, {1, 3}, {2, 4}, {3}}) {
		t.Errorf("expected [[2], [1, 3], [2, 4], [3]], got %v", r4)
	}
}
