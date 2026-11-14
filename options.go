package openskill

// WeightBounds holds the minimum and maximum boundaries for normalizing team player weights.
type WeightBounds struct {
	Min float64
	Max float64
}

// GammaFunc defines the signature of a custom gamma regularization function.
type GammaFunc func(
	c float64,
	k int,
	mu float64,
	sigmaSquared float64,
	team []Rating,
	rank int,
	weights []float64,
) float64

// DefaultGamma is the standard gamma function from Weng & Lin (2011).
func DefaultGamma(
	c float64,
	k int,
	mu float64,
	sigmaSquared float64,
	team []Rating,
	rank int,
	weights []float64,
) float64 {
	return mathSqrt(sigmaSquared) / c
}

// ModelConfig holds common configuration parameters for ranking models.
type ModelConfig struct {
	Mu           float64
	Sigma        float64
	Beta         float64
	Kappa        float64
	Gamma        GammaFunc
	Tau          float64
	Margin       float64
	LimitSigma   bool
	Balance      bool
	WeightBounds *WeightBounds
	WindowSize   int
	Epsilon      float64
}

// DefaultModelConfig returns a ModelConfig populated with openskill standard defaults.
func DefaultModelConfig() ModelConfig {
	return ModelConfig{
		Mu:           25.0,
		Sigma:        25.0 / 3.0,
		Beta:         25.0 / 6.0,
		Kappa:        0.0001,
		Gamma:        DefaultGamma,
		Tau:          25.0 / 300.0,
		Margin:       0.0,
		LimitSigma:   false,
		Balance:      false,
		WeightBounds: &WeightBounds{Min: 1.0, Max: 2.0},
		WindowSize:   4,
		Epsilon:      0.1,
	}
}

// ModelOption configures a ModelConfig.
type ModelOption func(*ModelConfig)

// WithMu sets the default initial mean skill.
func WithMu(mu float64) ModelOption {
	return func(c *ModelConfig) {
		c.Mu = mu
	}
}

// WithSigma sets the default standard deviation of skill prior.
func WithSigma(sigma float64) ModelOption {
	return func(c *ModelConfig) {
		c.Sigma = sigma
	}
}

// WithBeta sets the uncertainty hyperparameter beta.
func WithBeta(beta float64) ModelOption {
	return func(c *ModelConfig) {
		c.Beta = beta
	}
}

// WithKappa sets the variance lower-bound regularizer kappa.
func WithKappa(kappa float64) ModelOption {
	return func(c *ModelConfig) {
		c.Kappa = kappa
	}
}

// WithGamma sets a custom gamma function.
func WithGamma(gamma GammaFunc) ModelOption {
	return func(c *ModelConfig) {
		c.Gamma = gamma
	}
}

// WithTau sets the additive dynamics parameter tau.
func WithTau(tau float64) ModelOption {
	return func(c *ModelConfig) {
		c.Tau = tau
	}
}

// WithMargin sets the victory margin threshold.
func WithMargin(margin float64) ModelOption {
	return func(c *ModelConfig) {
		c.Margin = margin
	}
}

// WithLimitSigma sets whether to prevent sigma from growing beyond initial values.
func WithLimitSigma(limit bool) ModelOption {
	return func(c *ModelConfig) {
		c.LimitSigma = limit
	}
}

// WithBalance sets whether to balance teams by emphasizing rating outliers.
func WithBalance(balance bool) ModelOption {
	return func(c *ModelConfig) {
		c.Balance = balance
	}
}

// WithWeightBounds sets custom normalization bounds for player weights.
func WithWeightBounds(min, max float64) ModelOption {
	return func(c *ModelConfig) {
		c.WeightBounds = &WeightBounds{Min: min, Max: max}
	}
}

// WithoutWeightBounds disables weight normalization, using raw weights as provided.
func WithoutWeightBounds() ModelOption {
	return func(c *ModelConfig) {
		c.WeightBounds = nil
	}
}

// WithWindowSize sets the sliding comparison window size for partial pairing models.
func WithWindowSize(size int) ModelOption {
	return func(c *ModelConfig) {
		c.WindowSize = size
	}
}

// WithEpsilon sets the draw margin epsilon for Thurstone-Mosteller models.
func WithEpsilon(eps float64) ModelOption {
	return func(c *ModelConfig) {
		c.Epsilon = eps
	}
}

// RateOptions contains options for a single Rate operation.
type RateOptions struct {
	Ranks      []float64
	Scores     []float64
	Weights    [][]float64
	Tau        *float64
	LimitSigma *bool
	Margin     *float64
}

// RateOption configures RateOptions.
type RateOption func(*RateOptions)

// WithRanks specifies the team placements (lower is better, e.g. 1 = 1st place).
func WithRanks(ranks []float64) RateOption {
	return func(o *RateOptions) {
		o.Ranks = ranks
	}
}

// WithScores specifies the match scores (higher is better, converted to ranks).
func WithScores(scores []float64) RateOption {
	return func(o *RateOptions) {
		o.Scores = scores
	}
}

// WithWeights specifies the individual player weights in each team.
func WithWeights(weights [][]float64) RateOption {
	return func(o *RateOptions) {
		o.Weights = weights
	}
}

// WithRateTau temporarily overrides tau for this rating calculation.
func WithRateTau(tau float64) RateOption {
	return func(o *RateOptions) {
		o.Tau = &tau
	}
}

// WithRateLimitSigma overrides limitSigma for this rating calculation.
func WithRateLimitSigma(limit bool) RateOption {
	return func(o *RateOptions) {
		o.LimitSigma = &limit
	}
}

// WithRateMargin overrides margin for this rating calculation.
func WithRateMargin(margin float64) RateOption {
	return func(o *RateOptions) {
		o.Margin = &margin
	}
}
