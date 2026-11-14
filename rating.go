package openskill

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
)

// Rating represents an individual player's skill belief.
// It models skill as a Gaussian distribution with mean Mu and standard deviation Sigma.
type Rating struct {
	ID    string  `json:"id"`
	Name  string  `json:"name,omitempty"`
	Mu    float64 `json:"mu"`
	Sigma float64 `json:"sigma"`
}

// Type aliases for model-specific ratings, matching openskill.py naming.
type (
	PlackettLuceRating           = Rating
	BradleyTerryFullRating       = Rating
	BradleyTerryPartRating       = Rating
	ThurstoneMostellerFullRating = Rating
	ThurstoneMostellerPartRating = Rating
)

// GenerateUUID generates a random v4 hex UUID string (lowercase, 32 hex chars without hyphens).
func GenerateUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40 // Version 4
	b[8] = (b[8] & 0x3f) | 0x80 // Variant 10xx
	return hex.EncodeToString(b[:])
}

// NewRating creates a new Rating with the given mu, sigma, and optional name.
func NewRating(mu, sigma float64, name ...string) Rating {
	var n string
	if len(name) > 0 {
		n = name[0]
	}
	return Rating{
		ID:    GenerateUUID(),
		Name:  n,
		Mu:    mu,
		Sigma: sigma,
	}
}

// OrdinalOptions holds parameters for calculating the ordinal skill rating.
type OrdinalOptions struct {
	Z      float64
	Alpha  float64
	Target float64
}

// OrdinalOption configures an OrdinalOptions struct.
type OrdinalOption func(*OrdinalOptions)

// WithZ sets the confidence multiplier z (number of standard deviations to subtract from the mean).
func WithZ(z float64) OrdinalOption {
	return func(o *OrdinalOptions) {
		o.Z = z
	}
}

// WithAlpha sets the overall scaling factor alpha.
func WithAlpha(alpha float64) OrdinalOption {
	return func(o *OrdinalOptions) {
		o.Alpha = alpha
	}
}

// WithTarget sets the target shift value.
func WithTarget(target float64) OrdinalOption {
	return func(o *OrdinalOptions) {
		o.Target = target
	}
}

// Ordinal returns a single scalar value representing the player's conservative skill estimate.
// Defaults: z=3.0 (99.7% confidence), alpha=1.0, target=0.0.
// Formula: alpha * ((Mu - z * Sigma) + (target / alpha))
func (r Rating) Ordinal(opts ...OrdinalOption) float64 {
	o := OrdinalOptions{
		Z:      3.0,
		Alpha:  1.0,
		Target: 0.0,
	}
	for _, opt := range opts {
		opt(&o)
	}
	return o.Alpha * ((r.Mu - o.Z*r.Sigma) + (o.Target / o.Alpha))
}

// Clone creates an exact deep copy of the rating, retaining the same ID, Name, Mu, and Sigma.
func (r Rating) Clone() Rating {
	return Rating{
		ID:    r.ID,
		Name:  r.Name,
		Mu:    r.Mu,
		Sigma: r.Sigma,
	}
}

// Equal returns true if both ratings have the same Mu and Sigma.
func (r Rating) Equal(other Rating) bool {
	return r.Mu == other.Mu && r.Sigma == other.Sigma
}

// Less compares two ratings based on their default Ordinal values.
func (r Rating) Less(other Rating) bool {
	return r.Ordinal() < other.Ordinal()
}

// Greater compares two ratings based on their default Ordinal values.
func (r Rating) Greater(other Rating) bool {
	return r.Ordinal() > other.Ordinal()
}

// LessOrEqual compares two ratings based on their default Ordinal values.
func (r Rating) LessOrEqual(other Rating) bool {
	return r.Ordinal() <= other.Ordinal()
}

// GreaterOrEqual compares two ratings based on their default Ordinal values.
func (r Rating) GreaterOrEqual(other Rating) bool {
	return r.Ordinal() >= other.Ordinal()
}

// String returns a human-readable representation of the player's rating.
func (r Rating) String() string {
	if r.Name != "" {
		return fmt.Sprintf("Player Data:\n\nid: %s\nname: %s\nmu: %v\nsigma: %v\n", r.ID, r.Name, r.Mu, r.Sigma)
	}
	return fmt.Sprintf("Player Data:\n\nid: %s\nmu: %v\nsigma: %v\n", r.ID, r.Mu, r.Sigma)
}

// Repr returns a Python-style representation string.
func (r Rating) Repr() string {
	return fmt.Sprintf("Rating(mu=%v, sigma=%v)", r.Mu, r.Sigma)
}
