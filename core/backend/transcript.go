package backend

import (
	"context"
	"fmt"
	"time"

	"github.com/mudler/LocalAI/core/config"
	"github.com/mudler/LocalAI/core/trace"
	"github.com/mudler/LocalAI/core/schema"

	"github.com/mudler/LocalAI/pkg/grpc/proto"
	"github.com/mudler/LocalAI/pkg/model"
)

func ModelTranscription(audio, language string, translate, diarize bool, prompt string, ml *model.ModelLoader, modelConfig config.ModelConfig, appConfig *config.ApplicationConfig) (*schema.TranscriptionResult, error) {
	if modelConfig.Backend == "" {
		modelConfig.Backend = model.WhisperBackend
	}

	opts := ModelOptions(modelConfig, appConfig)

	transcriptionModel, err := ml.Load(opts...)
	if err != nil {
		return nil, err
	}

	if transcriptionModel == nil {
		return nil, fmt.Errorf("could not load transcription model")
	}

	var startTime time.Time
	if appConfig.EnableTracing {
		trace.InitBackendTracingIfEnabled(appConfig.TracingMaxItems)
		startTime = time.Now()
	}

	r, err := transcriptionModel.AudioTranscription(context.Background(), &proto.TranscriptRequest{
		Dst:            audio,
		Language:       language,
		Translate:      translate,
		Diarize:        diarize,
		Threads:        uint32(*modelConfig.Threads),
		Prompt:         prompt,
		WordTimestamps: true,
	})
	if err != nil {
		if appConfig.EnableTracing {
			trace.RecordBackendTrace(trace.BackendTrace{
				Timestamp: startTime,
				Duration:  time.Since(startTime),
				Type:      trace.BackendTraceTranscription,
				ModelName: modelConfig.Name,
				Backend:   modelConfig.Backend,
				Summary:   trace.TruncateString(audio, 200),
				Error:     err.Error(),
				Data: map[string]any{
					"audio_file": audio,
					"language":   language,
					"translate":  translate,
					"diarize":    diarize,
					"prompt":     prompt,
				},
			})
		}
		return nil, err
	}
	tr := &schema.TranscriptionResult{
		Text: r.Text,
	}
	for _, s := range r.Segments {
		var tks []int
		for _, t := range s.Tokens {
			tks = append(tks, int(t))
		}
		var words []schema.TranscriptionWord
		for _, w := range s.Words {
			words = append(words, schema.TranscriptionWord{
				Start:       float64(w.Start) / 1e9,
				End:         float64(w.End) / 1e9,
				Word:        w.Word,
				Probability: float64(w.Probability),
			})
		}
		tr.Segments = append(tr.Segments,
			schema.TranscriptionSegment{
				Text:    s.Text,
				Id:      int(s.Id),
				Start:   float64(s.Start) / 1e9,
				End:     float64(s.End) / 1e9,
				Tokens:  tks,
				Speaker: s.Speaker,
				Words:   words,
			})
	}

	if appConfig.EnableTracing {
		trace.RecordBackendTrace(trace.BackendTrace{
			Timestamp: startTime,
			Duration:  time.Since(startTime),
			Type:      trace.BackendTraceTranscription,
			ModelName: modelConfig.Name,
			Backend:   modelConfig.Backend,
			Summary:   trace.TruncateString(audio+" -> "+tr.Text, 200),
			Data: map[string]any{
				"audio_file":     audio,
				"language":       language,
				"translate":      translate,
				"diarize":        diarize,
				"prompt":         prompt,
				"result_text":    tr.Text,
				"segments_count": len(tr.Segments),
			},
		})
	}

	return tr, err
}
