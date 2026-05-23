# podcast-transcribe-zh

**Chinese (Mandarin) and English podcast transcription with speaker diarization.**

中文國語 / 英文 Podcast 多人語者轉錄 — 自動辨識說話人、輸出 SRT 字幕、智慧合併段落。

## ✨ Features

- 🎙 **Speaker diarization** — auto-labels SPEAKER_00 / SPEAKER_01 …
- 🌏 **Mandarin Chinese**: Whisper `large-v3-turbo` (high accuracy)
- 🇬🇧 **English**: Whisper `distil-large-v3` (fast)
- 📺 **SRT / Markdown / JSON output** — ready for YouTube subtitles or Show Notes
- 🔀 **Smart segment merging** — collapses fragmented Whisper output by speaker
- 🔤 **Hotword injection** for proper nouns / 人名

## ⚠️ Language support

| Language | Code | Quality |
|---|---|---|
| Mandarin Chinese 中文國語 | `zh` | ⭐⭐⭐⭐⭐ Excellent |
| English | `en` | ⭐⭐⭐⭐⭐ Excellent |
| Japanese | `ja` | ⭐⭐⭐⭐ Good |
| Korean | `ko` | ⭐⭐⭐⭐ Good |

**Not supported / 不支援**: Taiwanese Hokkien (台語/閩南語), Cantonese (粵語), other Chinese dialects. Whisper does not natively support these; transcripts will be garbled. Use a dialect-specific fine-tune instead.

## 📥 Inputs

| Field | Type | Default | Notes |
|---|---|---|---|
| `audio` | file | required | mp3 / wav / m4a / mp4, recommend < 60 min |
| `language` | str | `zh` | `zh` / `en` / `ja` / `ko` |
| `hotwords` | str | `""` | Proper nouns (e.g. `蔡康永 黃詹 OpenAI Anthropic`) |
| `enable_diarization` | bool | `true` | Set `false` to skip speaker labels (~30% faster) |
| `gap_threshold` | float | `1.5` | Seconds; merge same-speaker segs within this gap |
| `output_format` | str | `srt` | `srt` / `json` / `plain` |
| `hf_token` | str | `""` | Required if `enable_diarization=true` |

## 🔑 HuggingFace token setup (one-time)

Diarization uses `pyannote/speaker-diarization-3.1`, a gated model. Before first use:

1. Get a token at https://huggingface.co/settings/tokens (Read scope)
2. Accept terms at https://hf.co/pyannote/speaker-diarization-3.1
3. Accept terms at https://hf.co/pyannote/segmentation-3.0
4. Paste your token into the `hf_token` field

Set `enable_diarization=false` to skip diarization entirely (pure transcription, no token needed).

## 📤 Output schema

```json
{
  "segments": [
    {"start": 0.0, "end": 5.2, "speaker": "SPEAKER_00", "text": "簡單來講就是不放心"},
    {"start": 5.5, "end": 12.1, "speaker": "SPEAKER_01", "text": "不放心"}
  ],
  "n_segments": 5,
  "n_speakers": 2,
  "language_detected": "zh",
  "elapsed_seconds": 21.0,
  "model_used": "large-v3-turbo",
  "srt": "1\n00:00:00,000 --> 00:00:05,200\n[SPEAKER_00] 簡單來講就是不放心\n\n..."
}
```

## 🚀 Quick start (Python SDK)

```python
import replicate

output = replicate.run(
    "vm6eji6m4/podcast-transcribe-zh:latest",
    input={
        "audio": open("episode.mp3", "rb"),
        "language": "zh",
        "hotwords": "蔡康永 黃詹 OpenAI",
        "hf_token": "hf_xxxxxxxxxxxx",
    },
)
print(output["srt"])
```

## 🎯 Use cases

- Mandarin podcast Show Notes 自動產出
- 英文 podcast 翻譯前置處理
- YouTube 中文字幕（SRT 直上）
- 多人會議逐字稿
- 訪談類影音自動分段

## 🛠 Hardware & performance

Runs on Nvidia T4 (16GB). Typical timings:
- 19s English (jfk.flac): ~25s total (cold start: ~90s)
- 30s Mandarin podcast: ~25s total
- 60min podcast: ~5-8 minutes (estimate)

## 📜 License & source

MIT. Source: https://github.com/vm6eji6m4/replicate-podcast-transcribe-zh

Built on `pyannote.audio 3.3.0` + `faster-whisper 1.1.1` + `torch 2.3.1`.
