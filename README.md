# Whisper Chinese Pro

**Mandarin Chinese + English + Japanese + Korean podcast transcription with speaker diarization.**

中文 / 英文 / 日文 / 韓文 Podcast 多人語者轉錄 — Whisper large-v3-turbo + pyannote 3.3，含字級時間戳、信度評分、智慧段落合併。

## ✨ Why this model

| Feature | This model | thomasmol/whisper-diarization | victor-upmeet/whisperx |
|---|---|---|---|
| Whisper version | **large-v3-turbo** + distil-large-v3 (en) | large-v3-turbo | large-v3 |
| Diarization | pyannote 3.3 | pyannote 3.3 | pyannote 3.1 |
| Auto language detect | ✅ | ✅ | ✅ |
| Word-level timestamps | ✅ | ✅ | ✅ |
| Word-level confidence | ✅ | ✅ | — |
| Segment confidence (`avg_logprob`) | ✅ | ✅ | — |
| 3 input methods (file/URL/Base64) | ✅ | ✅ | URL only |
| **Smart segment merging** by speaker | ✅ (configurable gap) | manual | manual |
| **Realtime factor** in output | ✅ | — | — |
| **Speaker count control** | ✅ (1-10 or auto) | ✅ | — |
| Output formats | SRT / VTT / JSON / plain | JSON | JSON / SRT |
| GPU | T4 (cheaper) | A40 (faster) | A40 |
| **Optimized for Chinese podcast** | ✅ (vocab from 教育部辭典) | — | — |

## 🌏 Language support

| Language | Code | Quality | Model |
|---|---|---|---|
| Mandarin Chinese 國語 | `zh` | ⭐⭐⭐⭐⭐ | `large-v3-turbo` |
| English | `en` | ⭐⭐⭐⭐⭐ | `distil-large-v3` (super fast) |
| Japanese | `ja` | ⭐⭐⭐⭐ | `large-v3-turbo` |
| Korean | `ko` | ⭐⭐⭐⭐ | `large-v3-turbo` |
| **Auto-detect** | _empty_ | ✅ | `large-v3-turbo` |

⚠️ **NOT supported by Whisper natively**: Taiwanese Hokkien (台語/閩南語), Cantonese (粵語), Hakka (客語). Whisper has no acoustic model for these. Output will be garbled. Use dialect-specific fine-tunes instead.

## 📥 Inputs

| Field | Type | Default | Notes |
|---|---|---|---|
| `audio` | file | — | mp3 / wav / m4a / mp4, recommend < 60 min |
| `file_url` | str | "" | Public HTTPS URL (alternative to `audio`) |
| `file_string` | str | "" | Base64-encoded audio (alternative) |
| `language` | str | "" (auto-detect) | `zh` / `en` / `ja` / `ko` |
| `num_speakers` | int | 0 (auto) | Force exact count (1-10) |
| `prompt` | str | "" | Proper nouns / vocabulary to bias Whisper |
| `enable_diarization` | bool | true | Set false to skip speaker labels |
| `gap_threshold` | float | 1.5 | Merge same-speaker segments within N seconds |
| `word_timestamps` | bool | false | Include per-word start/end + probability |
| `output_format` | str | `srt` | `srt` / `vtt` / `json` / `plain` |
| `hf_token` | **Secret** | none | Required if `enable_diarization=true`. Masked in UI. |

## 🔑 First-time setup (one-time, 2 min)

1. Get a HuggingFace token: https://huggingface.co/settings/tokens (Read scope)
2. Accept terms: https://hf.co/pyannote/speaker-diarization-3.1
3. Accept terms: https://hf.co/pyannote/segmentation-3.0
4. Paste your token into the `hf_token` field (masked, never exposed in examples)

If you don't need speaker labels, set `enable_diarization=false` and no token is needed.

## 📤 Output schema

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "speaker": "SPEAKER_00",
      "text": "簡單來講就是不放心",
      "avg_logprob": -0.12,
      "words": [
        {"start": 0.0, "end": 0.4, "word": "簡單", "probability": 0.98},
        ...
      ]
    }
  ],
  "n_segments": 5,
  "n_speakers": 2,
  "language_detected": "zh",
  "language_probability": 0.998,
  "duration_seconds": 30.0,
  "elapsed_seconds": 21.0,
  "realtime_factor": 1.43,
  "model_used": "large-v3-turbo",
  "srt": "1\n00:00:00,000 --> 00:00:05,200\n[SPEAKER_00] 簡單來講就是不放心\n\n..."
}
```

## 🚀 Quick start

### Python SDK
```python
import replicate

output = replicate.run(
    "vm6eji6m4/whisper-chinese-pro:latest",
    input={
        "audio": open("episode.mp3", "rb"),
        "language": "zh",
        "prompt": "蔡康永, OpenAI, Anthropic",
        "hf_token": "hf_xxxxxxxxxxxx",
    },
)
print(output["srt"])
```

### cURL (URL input)
```bash
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "file_url": "https://example.com/podcast.mp3",
      "language": "zh",
      "num_speakers": 2,
      "hf_token": "hf_xxx"
    }
  }' \
  https://api.replicate.com/v1/predictions
```

### Node.js
```javascript
import Replicate from "replicate";
const replicate = new Replicate();
const out = await replicate.run("vm6eji6m4/whisper-chinese-pro:latest", {
  input: { file_url: "https://example.com/audio.mp3", language: "zh", hf_token: "hf_xxx" }
});
console.log(out.srt);
```

## 🎯 Use cases

- 中文 Podcast Show Notes 自動產出
- 英文 Podcast → 翻譯前置處理（先精準逐字稿，再翻譯）
- YouTube 中文字幕（SRT 直上）
- 多人會議 / 訪談 / 圓桌對話分段
- 客服錄音分析（speaker label + 信度評分）
- 內容創作者 SRT 字幕生產線

## 🛠 Hardware & speed

Runs on **Nvidia T4** (16GB VRAM). Real timings:

| Audio length | Cold start | Warm |
|---|---|---|
| 19s English (jfk.flac) | ~110s total | ~10s |
| 30s Mandarin podcast | ~120s total | ~25s |
| 5 min podcast | ~150s | ~50s |
| 30 min podcast (estimate) | ~280s | ~180s |
| 60 min podcast (estimate) | ~520s | ~420s |

Cold start ~90s (load Whisper + pyannote weights). Subsequent requests within ~15 min are warm.

## 💡 Tips

- **Auto-detect language**: leave `language` empty if you're not sure. Detection accuracy ~99% on clean audio > 10s.
- **Force speaker count**: if you know exactly N speakers, set `num_speakers=N` for cleaner diarization.
- **Hotwords for Chinese names**: pass `prompt="蔡康永, 朱平, 楊照"` to fix proper noun transcription.
- **Skip diarization for monologues**: set `enable_diarization=false` saves ~30% time + no HF token needed.
- **`word_timestamps=true`** for karaoke / sync apps.

## 📜 License & source

MIT. Source: https://github.com/vm6eji6m4/replicate-podcast-transcribe-zh

## 🙋 About

Built by 國裕 (Guoyu) — solo dev focused on AI infrastructure for Chinese content creators. Reach out at vm6eji6m4@gmail.com for custom transcription needs (台語 fine-tunes, on-premise deployments, podcast SaaS).

Tech stack: `pyannote-audio==3.3.0` · `huggingface-hub==0.23.5` · `torch==2.3.1+cu121` · `faster-whisper==1.1.1` · `numpy==1.26.4`.
