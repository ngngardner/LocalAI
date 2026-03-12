package schema

import "time"

type TranscriptionWord struct {
	Start       float64 `json:"start"`
	End         float64 `json:"end"`
	Word        string  `json:"word"`
	Probability float64 `json:"probability,omitempty"`
}

type TranscriptionSegment struct {
	Id      int                `json:"id"`
	Start   time.Duration      `json:"start"`
	End     time.Duration      `json:"end"`
	Text    string             `json:"text"`
	Tokens  []int              `json:"tokens"`
	Speaker string             `json:"speaker,omitempty"`
	Words   []TranscriptionWord `json:"words,omitempty"`
}

type TranscriptionResult struct {
	Segments []TranscriptionSegment `json:"segments,omitempty"`
	Text     string                 `json:"text"`
}
