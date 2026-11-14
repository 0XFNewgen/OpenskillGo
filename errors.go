package openskill

import (
	"errors"
	"fmt"
)

var (
	// ErrTooFewTeams is returned when less than 2 teams are provided.
	ErrTooFewTeams = errors.New("argument 'teams' must have at least 2 teams")

	// ErrEmptyTeam is returned when a team has no players.
	ErrEmptyTeam = errors.New("argument 'teams' must have at least 1 player per team")

	// ErrBothRanksAndScores is returned when both ranks and scores are provided.
	ErrBothRanksAndScores = errors.New("cannot accept both 'ranks' and 'scores' arguments at the same time")

	// ErrInvalidRatingValues is returned when rating slice has invalid length or format.
	ErrInvalidRatingValues = errors.New("rating slice must contain exactly 2 values [mu, sigma]")
)

// RanksLengthError indicates that ranks count does not match teams count.
type RanksLengthError struct {
	RanksLen int
	TeamsLen int
}

func (e *RanksLengthError) Error() string {
	return fmt.Sprintf("argument 'ranks' must have the same number of elements as 'teams' (%d != %d)", e.RanksLen, e.TeamsLen)
}

// ScoresLengthError indicates that scores count does not match teams count.
type ScoresLengthError struct {
	ScoresLen int
	TeamsLen  int
}

func (e *ScoresLengthError) Error() string {
	return fmt.Sprintf("argument 'scores' must have the same number of elements as 'teams' (%d != %d)", e.ScoresLen, e.TeamsLen)
}

// WeightsLengthError indicates a mismatch in weights dimension.
type WeightsLengthError struct {
	Message string
}

func (e *WeightsLengthError) Error() string {
	return e.Message
}
