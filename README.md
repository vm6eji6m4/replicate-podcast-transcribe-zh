# podcast-transcribe-zh

Chinese / Taiwanese (Hokkien) podcast transcription with **speaker diarization**.

- **Whisper**: `large-v3-turbo` for zh/ja/ko, `distil-large-v3` for en
- **Diarization**: pyannote 3.1 (sequential VRAM mgmt for 16GB cards)
- **Hotword injection** for proper nouns / Taiwanese
- Output: SRT / JSON segments with `[SPEAKER_xx]` labels

## Inputs

| Param | Type | Default | Notes |
|---|---|---|---|
| `audio` | file | required | mp3 / wav / m4a |
| `language` | str | `zh` | zh / en / ja / ko |
| `hotwords` | str | `""` | proper nouns to bias Whisper |
| `enable_diarization` | bool | `true` | turn off to save ~30% time |
| `gap_threshold` | float | `1.5` | merge same-speaker gap (sec) |
| `output_format` | str | `srt` | `srt` / `json` / `plain` |

## Setup secret

Set `HF_TOKEN` in your Replicate model settings (required for pyannote 3.1 gated model).

## Source

Built on top of [content-transcribe-saas](https://github.com/vm6eji6m4/content-transcribe-saas) (private).
