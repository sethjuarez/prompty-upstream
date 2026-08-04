package wire

import (
	model "prompty/model"
)

// PartSpec is the provider-neutral view of one emitted content part.
type PartSpec struct {
	// Kind is "text", "image", "file" or "audio".
	Kind string
	// Value is the text for a text part and the source (URL, data URI or raw
	// base64 payload) for every other kind.
	Value string
	// MediaType is the declared MIME type, empty when the author omitted it.
	MediaType string
	// Detail is the image detail hint, empty when unset.
	Detail string
}

// InspectPart projects an emitted content part. The second result is false for
// values that are not content parts.
func InspectPart(v interface{}) (PartSpec, bool) {
	switch p := v.(type) {
	case model.TextPart:
		return PartSpec{Kind: "text", Value: p.Value}, true
	case *model.TextPart:
		if p == nil {
			return PartSpec{}, false
		}
		return InspectPart(*p)

	case model.ImagePart:
		spec := PartSpec{Kind: "image", Value: p.Source}
		if p.MediaType != nil {
			spec.MediaType = *p.MediaType
		}
		if p.Detail != nil {
			spec.Detail = *p.Detail
		}
		return spec, true
	case *model.ImagePart:
		if p == nil {
			return PartSpec{}, false
		}
		return InspectPart(*p)

	case model.AudioPart:
		spec := PartSpec{Kind: "audio", Value: p.Source}
		if p.MediaType != nil {
			spec.MediaType = *p.MediaType
		}
		return spec, true
	case *model.AudioPart:
		if p == nil {
			return PartSpec{}, false
		}
		return InspectPart(*p)

	case model.FilePart:
		spec := PartSpec{Kind: "file", Value: p.Source}
		if p.MediaType != nil {
			spec.MediaType = *p.MediaType
		}
		return spec, true
	case *model.FilePart:
		if p == nil {
			return PartSpec{}, false
		}
		return InspectPart(*p)

	default:
		return PartSpec{}, false
	}
}

// InspectParts projects a message's parts, skipping unrecognised entries.
func InspectParts(parts []interface{}) []PartSpec {
	if len(parts) == 0 {
		return nil
	}
	out := make([]PartSpec, 0, len(parts))
	for _, part := range parts {
		if spec, ok := InspectPart(part); ok {
			out = append(out, spec)
		}
	}
	return out
}

// TextOf concatenates every text part of a message with no separator.
//
// This differs deliberately from the emitted Message.Text helper, which joins
// with a newline. Provider wire formats treat consecutive text blocks as one
// continuous string — the shared process vector anthropic_multiple_text_blocks
// requires "Here is the answer:" + " The weather..." to concatenate without an
// inserted newline.
func TextOf(msg model.Message) string {
	return joinTextParts(msg.Parts)
}

func joinTextParts(parts []interface{}) string {
	out := ""
	for _, part := range parts {
		if spec, ok := InspectPart(part); ok && spec.Kind == "text" {
			out += spec.Value
		}
	}
	return out
}

// IsSoleText reports whether the message is exactly one text part, and returns
// that text. Providers collapse this shape to a plain `content` string instead
// of a one-element block array.
func IsSoleText(msg model.Message) (string, bool) {
	if len(msg.Parts) != 1 {
		return "", false
	}
	spec, ok := InspectPart(msg.Parts[0])
	if !ok || spec.Kind != "text" {
		return "", false
	}
	return spec.Value, true
}

// IsSystemRole reports whether a role addresses the system prompt channel.
// Both providers hoist these messages out of the conversation array.
func IsSystemRole(role model.Role) bool {
	return role == model.RoleSystem || role == model.RoleDeveloper
}

// MimeToAudioFormat maps an audio MIME type to the short format token OpenAI
// expects in input_audio.format (§7.1.2). Unmapped audio/* types have their
// prefix stripped; anything else falls back to wav.
func MimeToAudioFormat(mime string) string {
	switch mime {
	case "audio/wav", "audio/x-wav":
		return "wav"
	case "audio/mpeg", "audio/mp3":
		return "mp3"
	case "audio/mp4":
		return "mp4"
	case "audio/ogg":
		return "ogg"
	case "audio/flac":
		return "flac"
	case "audio/webm":
		return "webm"
	case "audio/pcm":
		return "pcm"
	case "":
		return "wav"
	default:
		if len(mime) > 6 && mime[:6] == "audio/" {
			return mime[6:]
		}
		return "wav"
	}
}

// TextMessage builds a single-text-part message with optional metadata.
func TextMessage(role model.Role, text string, metadata map[string]interface{}) model.Message {
	return model.Message{
		Role:     role,
		Parts:    []interface{}{model.TextPart{Kind: "text", Value: text}},
		Metadata: metadata,
	}
}
