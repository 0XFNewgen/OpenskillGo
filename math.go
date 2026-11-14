package openskill

import (
	"errors"
	"math"
)

// FloatEpsilon is machine epsilon for float64, equivalent to sys.float_info.epsilon in Python.
const FloatEpsilon = 2.220446049250313e-16

// Coefficients for Peter J. Acklam's inverse normal CDF algorithm.
var (
	acklamA = [6]float64{
		-3.969683028665376e+01,
		2.209460984245205e+02,
		-2.759285104469687e+02,
		1.383577518672690e+02,
		-3.066479806614716e+01,
		2.506628277459239e+00,
	}
	acklamB = [5]float64{
		-5.447609879822406e+01,
		1.615858368580409e+02,
		-1.556989798598866e+02,
		6.680131188771972e+01,
		-1.328068155288572e+01,
	}
	acklamC = [6]float64{
		-7.784894002430293e-03,
		-3.223964580411365e-01,
		-2.400758277161838e+00,
		-2.549732539343734e+00,
		4.374664141464968e+00,
		2.938163982698783e+00,
	}
	acklamD = [4]float64{
		7.784695709041462e-03,
		3.224671290700398e-01,
		2.445134137142996e+00,
		3.754408661907416e+00,
	}
)

const (
	acklamPLow  = 0.02425
	acklamPHigh = 1.0 - acklamPLow
)

// PhiMajor computes the Standard Normal Cumulative Distribution Function (CDF).
func PhiMajor(x float64) float64 {
	return 0.5 * math.Erfc(-x/math.Sqrt2)
}

// PhiMinor computes the Standard Normal Probability Density Function (PDF).
func PhiMinor(x float64) float64 {
	return math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
}

// PhiMajorInverse computes the Inverse Standard Normal Cumulative Distribution Function (Quantile function).
// It implements Peter J. Acklam's rational approximation refined by Halley's rational method.
func PhiMajorInverse(p float64) (float64, error) {
	if p <= 0.0 || p >= 1.0 {
		return 0, errors.New("p must be in the range 0.0 < p < 1.0")
	}

	var x float64
	if p < acklamPLow {
		q := math.Sqrt(-2.0 * math.Log(p))
		num := ((((acklamC[0]*q+acklamC[1])*q+acklamC[2])*q+acklamC[3])*q+acklamC[4])*q + acklamC[5]
		den := (((acklamD[0]*q+acklamD[1])*q+acklamD[2])*q+acklamD[3])*q + 1.0
		x = num / den
	} else if p <= acklamPHigh {
		q := p - 0.5
		r := q * q
		num := (((((acklamA[0]*r+acklamA[1])*r+acklamA[2])*r+acklamA[3])*r+acklamA[4])*r+acklamA[5])*q
		den := ((((acklamB[0]*r+acklamB[1])*r+acklamB[2])*r+acklamB[3])*r+acklamB[4])*r + 1.0
		x = num / den
	} else {
		q := math.Sqrt(-2.0 * math.Log(1.0-p))
		num := ((((acklamC[0]*q+acklamC[1])*q+acklamC[2])*q+acklamC[3])*q+acklamC[4])*q + acklamC[5]
		den := (((acklamD[0]*q+acklamD[1])*q+acklamD[2])*q+acklamD[3])*q + 1.0
		x = -(num / den)
	}

	// Refinement using one step of Halley's rational method
	if math.Abs(x) < 25.0 {
		e := 0.5*math.Erfc(-x/math.Sqrt2) - p
		u := e * math.Sqrt(2*math.Pi) * math.Exp(0.5*x*x)
		x = x - u/(1.0+0.5*x*u)
	}

	return x, nil
}

// MustPhiMajorInverse is a helper that panics if p is out of range (useful where inputs are mathematically guaranteed).
func MustPhiMajorInverse(p float64) float64 {
	res, err := PhiMajorInverse(p)
	if err != nil {
		panic(err)
	}
	return res
}

// V is the function V as defined in Weng & Lin (2011).
func V(x, t float64) float64 {
	xt := x - t
	denom := PhiMajor(xt)
	if denom < FloatEpsilon {
		return -xt
	}
	return PhiMinor(xt) / denom
}

// W is the function W as defined in Weng & Lin (2011).
func W(x, t float64) float64 {
	xt := x - t
	denom := PhiMajor(xt)
	if denom < FloatEpsilon {
		if x < 0 {
			return 1.0
		}
		return 0.0
	}
	vx := V(x, t)
	return vx * (vx + xt)
}

// Vt is the function \tilde{V} as defined in Weng & Lin (2011).
func Vt(x, t float64) float64 {
	xx := math.Abs(x)
	b := PhiMajor(t-xx) - PhiMajor(-t-xx)
	if b < 1e-5 {
		if x < 0 {
			return -x - t
		}
		return -x + t
	}
	a := PhiMinor(-t-xx) - PhiMinor(t-xx)
	if x < 0 {
		return -a / b
	}
	return a / b
}

// Wt is the function \tilde{W} as defined in Weng & Lin (2011).
func Wt(x, t float64) float64 {
	xx := math.Abs(x)
	b := PhiMajor(t-xx) - PhiMajor(-t-xx)
	if b < FloatEpsilon {
		return 1.0
	}
	valVt := Vt(x, t)
	return ((t-xx)*PhiMinor(t-xx)+(t+xx)*PhiMinor(-t-xx))/b + valVt*valVt
}

// Normalize normalizes a vector of float64 to a target range of values [targetMin, targetMax].
func Normalize(vector []float64, targetMin, targetMax float64) []float64 {
	if len(vector) == 0 {
		return []float64{}
	}
	if len(vector) == 1 {
		return []float64{targetMax}
	}

	sourceMin := vector[0]
	sourceMax := vector[0]
	for _, v := range vector[1:] {
		if v < sourceMin {
			sourceMin = v
		}
		if v > sourceMax {
			sourceMax = v
		}
	}

	sourceRange := sourceMax - sourceMin
	targetRange := targetMax - targetMin

	if sourceRange == 0 {
		sourceRange = 0.0001
	}

	scaled := make([]float64, len(vector))
	for i, val := range vector {
		scaled[i] = (((val - sourceMin) / sourceRange) * targetRange) + targetMin
	}
	return scaled
}

// MatrixTranspose transposes a 2D matrix of any type.
func MatrixTranspose[T any](matrix [][]T) [][]T {
	if len(matrix) == 0 {
		return [][]T{}
	}
	rows := len(matrix)
	cols := len(matrix[0])
	result := make([][]T, cols)
	for i := 0; i < cols; i++ {
		result[i] = make([]T, rows)
		for j := 0; j < rows; j++ {
			result[i][j] = matrix[j][i]
		}
	}
	return result
}

// UnaryMinus negates a float64 value.
func UnaryMinus(number float64) float64 {
	return -number
}

// LadderPairs returns a list of pairs of elements that are adjacent in the ladder.
func LadderPairs[T any](teams []T) [][]T {
	n := len(teams)
	if n <= 1 {
		return [][]T{nil}
	}

	result := make([][]T, n)
	for i := 0; i < n; i++ {
		var neighbors []T
		if i > 0 {
			neighbors = append(neighbors, teams[i-1])
		}
		if i+1 < n {
			neighbors = append(neighbors, teams[i+1])
		}
		result[i] = neighbors
	}
	return result
}
