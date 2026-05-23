"""
Replicate Cog predictor — Chinese/Taiwanese Podcast Transcribe
Whisper large-v3-turbo (zh/ja/ko) + distil-large-v3 (en) + pyannote 3.3 diarization
"""
from __future__ import annotations

import gc
import os
import time
from typing import Optional

from cog import BasePredictor, Input, Path


DEFAULT_MODEL_BY_LANG = {
    "zh": "large-v3-turbo",
    "ja": "large-v3-turbo",
    "ko": "large-v3-turbo",
    "en": "distil-large-v3",
}


def _ts_srt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


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


def _release_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


class Predictor(BasePredictor):
    def setup(self) -> None:
        # Lazy-load models in predict() to allow per-call language selection.
        pass

    def predict(
        self,
        audio: Path = Input(description="Audio file (mp3/wav/m4a/mp4), recommended < 60 min"),
        language: str = Input(
            description="Language code. zh/ja/ko use large-v3-turbo, en uses distil-large-v3.",
            default="zh",
            choices=["zh", "en", "ja", "ko"],
        ),
        hotwords: str = Input(
            description="Proper nouns / Taiwanese terms to bias Whisper (e.g. '黃詹 蔡瀾 百靈果').",
            default="",
        ),
        enable_diarization: bool = Input(
            description="Run speaker diarization (requires hf_token). Disable to skip ~30% time.",
            default=True,
        ),
        gap_threshold: float = Input(
            description="Merge adjacent same-speaker segments within this gap (seconds).",
            default=1.5,
            ge=0.1,
            le=5.0,
        ),
        output_format: str = Input(
            description="Primary output format. JSON segments are always included.",
            default="srt",
            choices=["srt", "json", "plain"],
        ),
        hf_token: str = Input(
            description=(
                "HuggingFace token (Read scope). Required if enable_diarization=True. "
                "Get one at https://huggingface.co/settings/tokens. "
                "Then accept terms at https://hf.co/pyannote/speaker-diarization-3.1 "
                "and https://hf.co/pyannote/segmentation-3.0."
            ),
            default="",
        ),
    ) -> dict:
        import torch
        from faster_whisper import WhisperModel

        t_start = time.time()
        audio_path = str(audio)

        # ---- Diarization (pyannote 3.3) ----
        speaker_segs: list[tuple[float, float, str]] = []
        if enable_diarization:
            if not hf_token:
                raise RuntimeError(
                    "hf_token required when enable_diarization=True. "
                    "Get one at https://huggingface.co/settings/tokens, "
                    "accept pyannote model terms, or set enable_diarization=False."
                )
            from pyannote.audio import Pipeline

            print("[Diarize] loading pyannote.audio 3.3 pipeline...")
            t0 = time.time()
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            if pipeline is None:
                raise RuntimeError(
                    "Pyannote returned None. Likely causes: "
                    "(1) hf_token invalid; "
                    "(2) you haven't accepted https://hf.co/pyannote/speaker-diarization-3.1 terms; "
                    "(3) you haven't accepted https://hf.co/pyannote/segmentation-3.0 terms."
                )
            if torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))

            print(f"[Diarize] loaded in {time.time()-t0:.1f}s, running...")
            annotation = pipeline(audio_path)
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
        model_size = DEFAULT_MODEL_BY_LANG.get(language, "large-v3-turbo")

        print(f"[Whisper] loading {model_size} on {device}/{compute_type}...")
        t0 = time.time()
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        seg_iter, info = model.transcribe(
            audio_path,
            language=language,
            initial_prompt=hotwords or None,
            beam_size=5,
            vad_filter=True,
        )
        whisper_segs = [
            {"start": s.start, "end": s.end, "text": s.text.strip()} for s in seg_iter
        ]
        print(f"[Whisper] done in {time.time()-t0:.1f}s | {len(whisper_segs)} segs | lang={info.language}")
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
            else:
                merged.append({
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "speaker": seg["speaker"],
                    "text": seg["text"],
                })

        elapsed = round(time.time() - t_start, 1)
        n_speakers = len(set(s["speaker"] for s in merged))

        result = {
            "segments": merged,
            "n_segments": len(merged),
            "n_speakers": n_speakers,
            "language_detected": info.language,
            "elapsed_seconds": elapsed,
            "model_used": model_size,
        }
        if output_format == "srt":
            result["srt"] = _to_srt(merged)
        elif output_format == "plain":
            result["plain_text"] = "\n\n".join(s["text"] for s in merged)

        print(f"[Pipeline] total {elapsed}s | {len(merged)} merged segs | {n_speakers} speakers")
        return result
