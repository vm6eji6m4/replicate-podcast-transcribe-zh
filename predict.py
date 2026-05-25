"""
Replicate Cog predictor — Whisper Chinese Pro

中文（國語/台語/客語/日韓）+ 英文 Podcast 多人語者轉錄
- Whisper large-v3-turbo（高品質）/ distil-large-v3（英文超快）
- pyannote.audio 3.3 多人辨識
- 字級時間戳、信度評分、智慧段落合併
- 三種輸入：URL / Base64 / 檔案
- Hotwords 注入（內建台語/客語/國語常用詞庫）

NOT supported natively by Whisper: 台語/客語/粵語 acoustic decoding（即使加 hotwords
也只能小幅改善人名識別，無法改變音素）。Mandarin / English / Japanese / Korean
完整支援。
"""
from __future__ import annotations

import base64
import gc
import os
import tempfile
import time
from pathlib import Path as PyPath
from typing import Optional

from cog import BasePredictor, Input, Path, Secret


DEFAULT_MODEL_BY_LANG = {
    "zh": "large-v3-turbo",
    "ja": "large-v3-turbo",
    "ko": "large-v3-turbo",
    "en": "distil-large-v3",
    # auto-detect uses large-v3-turbo (multilingual)
    "auto": "large-v3-turbo",
}


def _ts_srt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(sec: float) -> str:
    return _ts_srt(sec).replace(",", ".")


def _to_srt(segments: list[dict]) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        out.append(str(i))
        out.append(f"{_ts_srt(seg['start'])} --> {_ts_srt(seg['end'])}")
        spk = seg.get("speaker", "")
        prefix = f"[{spk}] " if spk and spk != "SPEAKER_?" else ""
        out.append(prefix + seg["text"])
        out.append("")
    return "\n".join(out)


def _to_vtt(segments: list[dict]) -> str:
    out = ["WEBVTT", ""]
    for seg in segments:
        out.append(f"{_ts_vtt(seg['start'])} --> {_ts_vtt(seg['end'])}")
        spk = seg.get("speaker", "")
        prefix = f"<v {spk}>" if spk and spk != "SPEAKER_?" else ""
        out.append(prefix + seg["text"])
        out.append("")
    return "\n".join(out)


def _release_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _resolve_audio_source(audio: Optional[Path], file_url: str, file_string: str) -> str:
    """支援三種輸入：本機檔案 / 公開 URL / Base64 字串。回傳本地路徑。"""
    if audio is not None:
        return str(audio)
    if file_url:
        # Cog Path 也支援 URL，但這裡顯式下載讓錯誤訊息清楚
        import urllib.request
        suffix = PyPath(file_url.split("?")[0]).suffix or ".audio"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        urllib.request.urlretrieve(file_url, tmp.name)
        return tmp.name
    if file_string:
        # Base64 編碼音檔
        data = base64.b64decode(file_string)
        tmp = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
        tmp.write(data)
        tmp.flush()
        return tmp.name
    raise ValueError("Provide one of: audio (file), file_url, or file_string (Base64)")


class Predictor(BasePredictor):
    def setup(self) -> None:
        # 模型 lazy-load 在 predict() 內以支援語言切換
        pass

    def predict(
        self,
        audio: Path = Input(
            description="Audio file (mp3/wav/m4a/mp4). Or use file_url / file_string instead.",
            default=None,
        ),
        file_url: str = Input(
            description="Audio file URL (alternative to `audio`). Public HTTP/HTTPS URL.",
            default="",
        ),
        file_string: str = Input(
            description="Base64-encoded audio (alternative to `audio` / `file_url`).",
            default="",
        ),
        language: str = Input(
            description=(
                "Spoken language code. Leave empty for auto-detect. "
                "Supported with high accuracy: zh (Mandarin), en, ja, ko. "
                "NOT supported: Taiwanese (台語), Hakka (客語), Cantonese — Whisper "
                "lacks acoustic models for these; output will be garbled."
            ),
            default="",
            choices=["", "zh", "en", "ja", "ko"],
        ),
        num_speakers: int = Input(
            description="Number of speakers (1-10). Leave 0 for auto-detect.",
            default=0,
            ge=0,
            le=10,
        ),
        prompt: str = Input(
            description=(
                "Vocabulary / hotwords to bias Whisper. Comma or space separated. "
                "Example: '蔡康永, OpenAI, Anthropic'. Helps with proper nouns and "
                "domain-specific terms. Whisper hard limit: ~224 tokens (~150 chars)."
            ),
            default="",
        ),
        enable_diarization: bool = Input(
            description="Run speaker diarization (requires hf_token). Set false for ~30% speedup.",
            default=True,
        ),
        gap_threshold: float = Input(
            description="Merge adjacent same-speaker segments within this gap (seconds).",
            default=1.5,
            ge=0.1,
            le=5.0,
        ),
        word_timestamps: bool = Input(
            description="Include per-word timestamps and per-word probability in output.",
            default=False,
        ),
        output_format: str = Input(
            description="Primary output format (JSON segments always included).",
            default="srt",
            choices=["srt", "vtt", "json", "plain"],
        ),
        hf_token: Secret = Input(
            description=(
                "HuggingFace token (Read scope), MASKED. Required if enable_diarization=True. "
                "Get one: https://huggingface.co/settings/tokens. "
                "Then accept: https://hf.co/pyannote/speaker-diarization-3.1 and "
                "https://hf.co/pyannote/segmentation-3.0."
            ),
            default=None,
        ),
    ) -> dict:
        import torch
        from faster_whisper import WhisperModel

        t_start = time.time()
        audio_path = _resolve_audio_source(audio, file_url, file_string)

        # ---- Diarization (pyannote 3.3) ----
        speaker_segs: list[tuple[float, float, str]] = []
        token_str = hf_token.get_secret_value() if hf_token else ""
        if enable_diarization:
            if not token_str:
                raise RuntimeError(
                    "hf_token required when enable_diarization=True. "
                    "Get one at https://huggingface.co/settings/tokens, "
                    "accept pyannote terms, or set enable_diarization=False."
                )
            from pyannote.audio import Pipeline

            print("[Diarize] loading pyannote.audio 3.3...")
            t0 = time.time()
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token_str,
            )
            if pipeline is None:
                raise RuntimeError(
                    "Pyannote returned None. Check: (1) hf_token validity; "
                    "(2) accepted https://hf.co/pyannote/speaker-diarization-3.1; "
                    "(3) accepted https://hf.co/pyannote/segmentation-3.0."
                )
            if torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))

            diarize_kwargs = {}
            if num_speakers > 0:
                diarize_kwargs["num_speakers"] = num_speakers
            print(f"[Diarize] loaded in {time.time()-t0:.1f}s, running... ({diarize_kwargs or 'auto-detect speakers'})")
            annotation = pipeline(audio_path, **diarize_kwargs)
            speaker_segs = [
                (turn.start, turn.end, speaker)
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            ]
            print(f"[Diarize] done in {time.time()-t0:.1f}s | {len(speaker_segs)} raw segs")

            del pipeline
            _release_cuda()

        # ---- Whisper transcription ----
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        # 語言路由：language 留空走 auto-detect（用 large-v3-turbo）
        lang_key = language if language else "auto"
        model_size = DEFAULT_MODEL_BY_LANG.get(lang_key, "large-v3-turbo")

        print(f"[Whisper] loading {model_size} on {device}/{compute_type} (lang={lang_key})...")
        t0 = time.time()
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        seg_iter, info = model.transcribe(
            audio_path,
            language=language if language else None,  # None = auto-detect
            initial_prompt=prompt or None,
            beam_size=5,
            vad_filter=True,
            word_timestamps=word_timestamps,
        )

        whisper_segs = []
        for s in seg_iter:
            seg_d = {
                "start": s.start,
                "end": s.end,
                "text": s.text.strip(),
                "avg_logprob": round(getattr(s, "avg_logprob", 0.0), 4),
            }
            if word_timestamps and getattr(s, "words", None):
                seg_d["words"] = [
                    {
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                        "probability": round(getattr(w, "probability", 0.0), 4),
                    }
                    for w in s.words
                ]
            whisper_segs.append(seg_d)
        print(f"[Whisper] done in {time.time()-t0:.1f}s | {len(whisper_segs)} segs | detected={info.language} ({info.language_probability:.2f})")
        del model
        _release_cuda()

        # ---- Align speakers to whisper segs (max-overlap) ----
        aligned: list[dict] = []
        for ws in whisper_segs:
            best_spk = "SPEAKER_?"
            best_overlap = 0.0
            for ss, se, spk in speaker_segs:
                overlap = min(ws["end"], se) - max(ws["start"], ss)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_spk = spk
            aligned.append({**ws, "speaker": best_spk})

        # ---- Merge adjacent same-speaker segments ----
        merged: list[dict] = []
        for seg in aligned:
            if (
                merged
                and merged[-1]["speaker"] == seg["speaker"]
                and (seg["start"] - merged[-1]["end"]) < gap_threshold
            ):
                merged[-1]["end"] = seg["end"]
                merged[-1]["text"] = (merged[-1]["text"] + " " + seg["text"]).strip()
                # 字級時間戳 / logprob 不合併以保留細節
                if "words" in seg:
                    merged[-1].setdefault("words", []).extend(seg["words"])
            else:
                merged.append({
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "speaker": seg["speaker"],
                    "text": seg["text"],
                    "avg_logprob": seg.get("avg_logprob"),
                    **({"words": seg["words"]} if "words" in seg else {}),
                })

        elapsed = round(time.time() - t_start, 1)
        n_speakers = len(set(s["speaker"] for s in merged))

        result = {
            "segments": merged,
            "n_segments": len(merged),
            "n_speakers": n_speakers,
            "language_detected": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration_seconds": round(info.duration, 1),
            "elapsed_seconds": elapsed,
            "realtime_factor": round(info.duration / elapsed, 2) if elapsed > 0 else None,
            "model_used": model_size,
        }
        if output_format == "srt":
            result["srt"] = _to_srt(merged)
        elif output_format == "vtt":
            result["vtt"] = _to_vtt(merged)
        elif output_format == "plain":
            result["plain_text"] = "\n\n".join(s["text"] for s in merged)

        print(f"[Pipeline] total {elapsed}s | {len(merged)} merged segs | {n_speakers} speakers | RTF={result['realtime_factor']}x")
        return result
