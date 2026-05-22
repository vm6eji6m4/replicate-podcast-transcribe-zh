"""
Replicate Cog predictor — Podcast / 多人語者轉錄
中文/台語預設用 large-v3-turbo + pyannote diarization
"""
from __future__ import annotations

import gc
import json
import os
import time
from typing import Optional

from cog import BasePredictor, Input, Path


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


DEFAULT_MODEL_BY_LANG = {
    "zh": "large-v3-turbo",
    "ja": "large-v3-turbo",
    "ko": "large-v3-turbo",
    "en": "distil-large-v3",
}


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Replicate 啟動時呼叫。pyannote / whisper 模型在第一次 predict 才 lazy load。"""
        pass

    def predict(
        self,
        audio: Path = Input(description="音檔（mp3/wav/m4a，建議 < 60 分鐘）"),
        language: str = Input(
            description="語言代碼。zh/ja/ko 用 large-v3-turbo；en 用 distil-large-v3",
            default="zh",
            choices=["zh", "en", "ja", "ko"],
        ),
        hotwords: str = Input(
            description="熱詞/專有名詞（選填，會注入 Whisper initial_prompt，提升人名/台語準確度）",
            default="",
        ),
        enable_diarization: bool = Input(
            description="是否做多人語者辨識（關掉可省 ~30% 時間）",
            default=True,
        ),
        gap_threshold: float = Input(
            description="合併同 speaker 相鄰段的最大間隔秒數",
            default=1.5,
            ge=0.1,
            le=5.0,
        ),
        output_format: str = Input(
            description="主要輸出格式（會同時回傳 JSON 詳細結果）",
            default="srt",
            choices=["srt", "json", "plain"],
        ),
        hf_token: str = Input(
            description=(
                "HuggingFace token（diarization 需要）。"
                "申請：https://huggingface.co/settings/tokens（Read 即可）。"
                "首次使用要先接受兩個模型條款："
                "https://hf.co/pyannote/speaker-diarization-3.1 + "
                "https://hf.co/pyannote/segmentation-3.0"
            ),
            default="",
        ),
    ) -> dict:
        """跑 diarize → transcribe → align → merge，回傳 dict。"""
        from pyannote.audio import Pipeline as PyannotePipeline
        from faster_whisper import WhisperModel
        import torch

        audio_path = str(audio)
        t_start = time.time()

        # ---- Diarization ----
        speaker_segs: list[tuple[float, float, str]] = []
        if enable_diarization:
            if not hf_token:
                raise RuntimeError(
                    "hf_token required for diarization. "
                    "Get one at https://huggingface.co/settings/tokens, "
                    "then accept pyannote model terms. "
                    "Or pass enable_diarization=False to skip."
                )
            print("[Diarize] loading pyannote...")
            t0 = time.time()
            # cog.yaml 已 pin huggingface-hub<0.24，use_auth_token kwarg 可用。
            # 同時設環境變數作雙重保險。
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            if pipeline is None:
                raise RuntimeError(
                    "Pyannote returned None — usually means: "
                    "(1) hf_token invalid, "
                    "(2) you haven't accepted pyannote/speaker-diarization-3.1 terms at https://hf.co/pyannote/speaker-diarization-3.1, "
                    "(3) you haven't accepted pyannote/segmentation-3.0 terms at https://hf.co/pyannote/segmentation-3.0"
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

        # ---- Whisper ----
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
        print(f"[Whisper] done in {time.time()-t0:.1f}s | {len(whisper_segs)} segs | detected={info.language}")
        del model
        _release_cuda()

        # ---- Align speakers to whisper segs ----
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

        # ---- Merge adjacent same-speaker segs ----
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

        total_elapsed = time.time() - t_start
        n_speakers = len(set(s["speaker"] for s in merged))

        # ---- Build output ----
        result = {
            "segments": merged,
            "n_segments": len(merged),
            "n_speakers": n_speakers,
            "language_detected": info.language,
            "elapsed_seconds": round(total_elapsed, 1),
            "model_used": model_size,
        }
        if output_format == "srt":
            result["srt"] = _to_srt(merged)
        elif output_format == "plain":
            result["plain_text"] = "\n\n".join(s["text"] for s in merged)
        # json 已內含

        print(f"[Pipeline] total {total_elapsed:.1f}s | {len(merged)} merged segs | {n_speakers} speakers")
        return result
