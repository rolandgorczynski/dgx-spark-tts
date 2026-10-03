"""HTTP API for Chatterbox Multilingual V3.

The model generates a whole chunk in one pass (max_new_tokens=1000 inside
the upstream generate()). Streaming here is sentence-chunk WAV, not
token-level audio. A short sentence is one WAV. A long article is split,
synthesized piece by piece, and written as one WAV stream.
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS

T3_MODEL = os.environ.get("CHATTERBOX_T3_MODEL", "v3")
DEFAULT_LANGUAGE = os.environ.get("TTS_LANGUAGE", "de").lower()
CHUNK_CHARS = int(os.environ.get("TTS_CHUNK_CHARS", "400"))
VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/voices"))
ALLOW_CPU = os.environ.get("TTS_ALLOW_CPU", "").lower() in {"1", "true", "yes"}
if torch.cuda.is_available():
    DEVICE = "cuda"
elif ALLOW_CPU:
    DEVICE = "cpu"
else:
    raise SystemExit(
        "CUDA is not visible in this container. On the Spark, install the "
        "NVIDIA Container Toolkit and keep the GPU reservation in compose. "
        "Set TTS_ALLOW_CPU=1 only for a slow CPU smoke test."
    )

_model: Optional[ChatterboxMultilingualTTS] = None
_lock = threading.Lock()
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")


def load_model() -> ChatterboxMultilingualTTS:
    global _model
    if _model is None:
        _model = ChatterboxMultilingualTTS.from_pretrained(DEVICE, t3_model=T3_MODEL)
    return _model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(title="spark-tts", version="1.0.0", lifespan=lifespan)


class SpeechRequest(BaseModel):
    input: Optional[str] = None
    text: Optional[str] = None
    language: Optional[str] = None
    voice: Optional[str] = Field(
        default=None, description="builtin, or a filename in /voices"
    )
    stream: bool = False
    response_format: str = "wav"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    seed: Optional[int] = None


def resolve_text(body: SpeechRequest) -> str:
    raw = body.input if body.input is not None else body.text
    if raw is None or not raw.strip():
        raise HTTPException(status_code=400, detail="input (or text) is required")
    text = raw.strip()
    if len(text) > 100_000:
        raise HTTPException(status_code=400, detail="text longer than 100000 characters")
    return text


def resolve_language(body: SpeechRequest) -> str:
    language = (body.language or DEFAULT_LANGUAGE).lower()
    if language not in SUPPORTED_LANGUAGES:
        known = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise HTTPException(
            status_code=400,
            detail=f"unsupported language '{language}'. supported: {known}",
        )
    return language


def resolve_voice(voice: Optional[str]) -> Optional[str]:
    if voice is None or voice in {"", "default", "builtin"}:
        return None
    name = Path(voice).name
    if name != voice or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="voice must be a file name, not a path")
    candidate = VOICES_DIR / name
    if not candidate.is_file():
        for ext in (".wav", ".mp3", ".flac"):
            alt = VOICES_DIR / f"{name}{ext}"
            if alt.is_file():
                candidate = alt
                break
    if not candidate.is_file():
        raise HTTPException(
            status_code=400, detail=f"voice file not found in /voices: {name}"
        )
    return str(candidate)


def split_text(text: str, limit: int) -> list[str]:
    parts = [p.strip() for p in _SENTENCE.split(text) if p.strip()]
    if not parts:
        return [text]
    chunks: list[str] = []
    buf = ""
    for part in parts:
        while len(part) > limit:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(part[:limit])
            part = part[limit:].strip()
        if not part:
            continue
        if buf and len(buf) + 1 + len(part) > limit:
            chunks.append(buf)
            buf = part
        else:
            buf = f"{buf} {part}".strip()
    if buf:
        chunks.append(buf)
    return chunks


def to_float(wav: torch.Tensor) -> np.ndarray:
    audio = wav.detach().float().cpu().numpy().reshape(-1)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def crossfade(pieces: list[np.ndarray], sample_rate: int, ms: int = 20) -> np.ndarray:
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    out = pieces[0]
    n = max(1, int(sample_rate * ms / 1000))
    for nxt in pieces[1:]:
        if len(out) < n or len(nxt) < n:
            out = np.concatenate([out, nxt])
            continue
        fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
        fade_in = 1.0 - fade_out
        mixed = out[-n:] * fade_out + nxt[:n] * fade_in
        out = np.concatenate([out[:-n], mixed, nxt[n:]])
    return out


def wav_header(sample_rate: int, data_bytes: int) -> bytes:
    byte_rate = sample_rate * 2
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        byte_rate,
        2,
        16,
        b"data",
        data_bytes,
    )


def pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def synthesize(
    text: str, language: str, voice_path: Optional[str], body: SpeechRequest
) -> list[np.ndarray]:
    model = load_model()
    pieces: list[np.ndarray] = []
    with _lock:
        if body.seed is not None:
            torch.manual_seed(body.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(body.seed)
        for piece in split_text(text, CHUNK_CHARS):
            wav = model.generate(
                piece,
                language_id=language,
                audio_prompt_path=voice_path,
                exaggeration=body.exaggeration,
                cfg_weight=body.cfg_weight,
                temperature=body.temperature,
            )
            pieces.append(to_float(wav))
    return pieces


def stream_wav(
    text: str, language: str, voice_path: Optional[str], body: SpeechRequest
) -> Iterator[bytes]:
    model = load_model()
    sample_rate = int(model.sr)
    # Oversized data length so a client can start reading before the last chunk.
    yield wav_header(sample_rate, 0x7FFFFF00)
    with _lock:
        if body.seed is not None:
            torch.manual_seed(body.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(body.seed)
        for piece in split_text(text, CHUNK_CHARS):
            wav = model.generate(
                piece,
                language_id=language,
                audio_prompt_path=voice_path,
                exaggeration=body.exaggeration,
                cfg_weight=body.cfg_weight,
                temperature=body.temperature,
            )
            yield pcm16(to_float(wav))


@app.get("/health")
def health():
    loaded = _model is not None
    return {
        "status": "ok" if loaded else "loading",
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "t3_model": T3_MODEL,
        "default_language": DEFAULT_LANGUAGE,
        "sample_rate": int(_model.sr) if loaded else None,
    }


@app.get("/v1/languages")
def languages():
    return {"languages": SUPPORTED_LANGUAGES, "default": DEFAULT_LANGUAGE}


@app.post("/v1/audio/speech")
@app.post("/tts")
def speech(body: SpeechRequest):
    text = resolve_text(body)
    language = resolve_language(body)
    voice_path = resolve_voice(body.voice)
    fmt = body.response_format.lower()
    if fmt not in {"wav", "mp3"}:
        raise HTTPException(status_code=400, detail="response_format must be wav or mp3")
    if body.stream and fmt != "wav":
        raise HTTPException(status_code=400, detail="stream supports wav only")

    model = load_model()
    sample_rate = int(model.sr)

    if body.stream:
        return StreamingResponse(
            stream_wav(text, language, voice_path, body),
            media_type="audio/wav",
            headers={"X-Sample-Rate": str(sample_rate), "X-Language": language},
        )

    pieces = synthesize(text, language, voice_path, body)
    audio = crossfade(pieces, sample_rate)
    pcm = pcm16(audio)
    if fmt == "wav":
        payload = wav_header(sample_rate, len(pcm)) + pcm
        media = "audio/wav"
    else:
        payload = encode_mp3(pcm, sample_rate)
        media = "audio/mpeg"
    return Response(
        content=payload,
        media_type=media,
        headers={"X-Sample-Rate": str(sample_rate), "X-Language": language},
    )


def encode_mp3(pcm: bytes, sample_rate: int) -> bytes:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-f",
            "mp3",
            "-b:a",
            "192k",
            "pipe:1",
        ],
        input=pcm,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", errors="replace")[-500:]
        raise HTTPException(status_code=500, detail=f"ffmpeg mp3 encode failed: {detail}")
    return proc.stdout
