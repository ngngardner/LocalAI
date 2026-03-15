package format

import (
	"fmt"
	"strings"

	"github.com/mudler/LocalAI/core/schema"
)

func TranscriptionResponse(tr *schema.TranscriptionResult, resFmt schema.TranscriptionResponseFormatType) string {
	var out string
	if resFmt == schema.TranscriptionResponseFormatLrc {
		out = "[by:LocalAI]\n[re:LocalAI]\n"
	} else if resFmt == schema.TranscriptionResponseFormatVtt {
		out = "WEBVTT"
	}

	for i, s := range tr.Segments {
		switch resFmt {
		case schema.TranscriptionResponseFormatLrc:
			m := int64(s.Start * 1000)
			out += fmt.Sprintf("\n[%02d:%02d:%02d] %s", m/60000, (m/1000)%60, (m%1000)/10, strings.TrimSpace(s.Text))
		case schema.TranscriptionResponseFormatSrt:
			out += fmt.Sprintf("\n\n%d\n%s --> %s\n%s", i+1, secondsToTimecode(s.Start, ','), secondsToTimecode(s.End, ','), strings.TrimSpace(s.Text))
		case schema.TranscriptionResponseFormatVtt:
			out += fmt.Sprintf("\n\n%s --> %s\n%s\n", secondsToTimecode(s.Start, '.'), secondsToTimecode(s.End, '.'), strings.TrimSpace(s.Text))
		case schema.TranscriptionResponseFormatText:
			fallthrough
		default:
			out += fmt.Sprintf("\n%s", strings.TrimSpace(s.Text))
		}
	}

	return out
}

func secondsToTimecode(seconds float64, millisSeparator rune) string {
	m := int64(seconds * 1000)
	return fmt.Sprintf("%02d:%02d:%02d%c%03d", m/3600000, (m/60000)%60, (m/1000)%60, millisSeparator, m%1000)
}
