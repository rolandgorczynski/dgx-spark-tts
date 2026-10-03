"""HTTP API for Chatterbox Multilingual V3.

The model generates one chunk per call (max_new_tokens=1000 inside generate()).
A stream is sentence-sized audio, not token-level audio. The first sentence is
synthesized alone so playback can start, then later sentences are packed up to
TTS_CHUNK_CHARS. Progressive clients should request Ogg Opus
(``response_format=ogg``, ``stream=true``): chunked HTTP, no Content-Length.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from tts_audio import (
    MODEL_SAMPLE_RATE,
    SPEED_MAX,
    SPEED_MIN,
    VOICE_EXTENSIONS,
    PcmCrossfader,
    apply_speed_pcm,
    encode_mp3,
    encode_ogg,
    iter_pcm_stream,
    list_voices as _list_voice_files,
    pcm16,
    split_text,
    wav_header,
)

log = logging.getLogger("spark-tts")

T3_MODEL = os.environ.get("CHATTERBOX_T3_MODEL", "v3")
DEFAULT_LANGUAGE = os.environ.get("TTS_LANGUAGE", "de").lower()
CHUNK_CHARS = int(os.environ.get("TTS_CHUNK_CHARS", "400"))
VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/voices"))
ALLOW_CPU = os.environ.get("TTS_ALLOW_CPU", "").lower() in {"1", "true", "yes"}

# Pinned to resemble-ai/chatterbox@5de7a54 (Multilingual V3). Checked again
# when the checkpoint loads.
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}

CUDA_MESSAGE = (
    "CUDA is not visible in this container. On the Spark, install the "
    "NVIDIA Container Toolkit and keep the GPU reservation in compose. "
    "Set TTS_ALLOW_CPU=1 only for a slow CPU smoke test."
)

_model = None
_device: Optional[str] = None
_lock = threading.Lock()
# Tests inject an iterator of s16le mono PCM at MODEL_SAMPLE_RATE.
_pcm_override = None


class _PassthroughWatermarker:
    """Identity stand-in for Perth's implicit watermarker.

    Chatterbox calls ``perth.PerthImplicitWatermarker()`` with no skip flag.
    resemble-perth sets that name to ``None`` when the implicit watermarker
    (or a native dependency) fails to import, which is the aarch64 / GB10
    failure mode. Synthesis still needs ``apply_watermark`` to return audio.
    """

    def apply_watermark(self, wav, sample_rate=None, **_kwargs):
        return wav


def _ensure_perth_watermarker() -> None:
    """Install a passthrough if Perth's watermarker class did not import."""
    import perth

    watermarker = getattr(perth, "PerthImplicitWatermarker", None)
    if callable(watermarker):
        return
    log.warning(
        "PerthImplicitWatermarker is not callable (%r); "
        "TTS will run without an audio watermark",
        watermarker,
    )
    perth.PerthImplicitWatermarker = _PassthroughWatermarker


def _skip_model() -> bool:
    return os.environ.get("TTS_SKIP_MODEL", "").lower() in {"1", "true", "yes"}


def _api_token() -> str:
    return os.environ.get("TTS_API_TOKEN", "").strip()


def load_model():
    global _model, _device
    if _model is not None:
        return _model
    import torch
    from chatterbox.mtl_tts import SUPPORTED_LANGUAGES as upstream
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    missing = set(upstream) - set(SUPPORTED_LANGUAGES)
    if missing:
        SUPPORTED_LANGUAGES.update({code: upstream[code] for code in missing})

    if torch.cuda.is_available():
        _device = "cuda"
    elif ALLOW_CPU:
        _device = "cpu"
    else:
        raise SystemExit(CUDA_MESSAGE)

    # Must run after perth is importable and before Chatterbox constructs
    # the watermarker inside from_pretrained. Upstream has no disable flag.
    _ensure_perth_watermarker()
    _model = ChatterboxMultilingualTTS.from_pretrained(_device, t3_model=T3_MODEL)
    return _model


def active_sample_rate() -> int:
    if _model is not None:
        return int(_model.sr)
    return MODEL_SAMPLE_RATE


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not _skip_model():
        load_model()
    yield


app = FastAPI(title="spark-tts", version="1.1.0", lifespan=lifespan)


class SpeechRequest(BaseModel):
    input: Optional[str] = None
    text: Optional[str] = None
    language: Optional[str] = None
    voice: Optional[str] = Field(
        default=None, description="builtin, or a filename in /voices"
    )
    stream: bool = False
    response_format: str = "wav"
    speed: float = Field(
        default=1.0,
        ge=SPEED_MIN,
        le=SPEED_MAX,
        description="Pitch-preserving speaking rate. 1.0 is the model output.",
    )
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    seed: Optional[int] = None


class VoiceInfo(BaseModel):
    id: str
    name: str
    builtin: bool


class VoicesResponse(BaseModel):
    voices: list[VoiceInfo]


def require_token(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    expected = _api_token()
    if not expected:
        return
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    elif x_api_key:
        supplied = x_api_key.strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


api = APIRouter(dependencies=[Depends(require_token)])


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
    if name != voice or name in {".", ".."} or "/" in voice or "\\" in voice:
        raise HTTPException(status_code=400, detail="voice must be a file name, not a path")
    candidate = VOICES_DIR / name
    if not candidate.is_file():
        for ext in sorted(VOICE_EXTENSIONS):
            alt = VOICES_DIR / f"{name}{ext}"
            if alt.is_file():
                candidate = alt
                break
    if not candidate.is_file():
        raise HTTPException(
            status_code=404, detail=f"voice file not found in /voices: {name}"
        )
    return str(candidate)


def list_voices() -> list[dict]:
    return _list_voice_files(VOICES_DIR)


def to_float(wav) -> np.ndarray:
    audio = wav.detach().float().cpu().numpy().reshape(-1)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def iter_synthesis_pcm(
    text: str,
    language: str,
    voice_path: Optional[str],
    body: SpeechRequest,
    *,
    eager_first: bool,
) -> Iterator[bytes]:
    """Yield s16le mono PCM. The override (tests) bypasses the model."""
    if _pcm_override is not None:
        yield from _pcm_override(text, language, voice_path, body)
        return
    if _skip_model():
        raise HTTPException(status_code=503, detail="TTS model is not loaded")

    model = load_model()
    sample_rate = int(model.sr)
    fader = PcmCrossfader(sample_rate)
    parts = split_text(text, CHUNK_CHARS, eager_first=eager_first)
    log.info(
        "synth language=%s voice=%s speed=%.2f chunks=%s eager=%s chars=%s",
        language,
        body.voice or "builtin",
        body.speed,
        len(parts),
        eager_first,
        len(text),
    )
    with _lock:
        if body.seed is not None:
            import torch

            torch.manual_seed(body.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(body.seed)
        for index, piece in enumerate(parts, start=1):
            wav = model.generate(
                piece,
                language_id=language,
                audio_prompt_path=voice_path,
                exaggeration=body.exaggeration,
                cfg_weight=body.cfg_weight,
                temperature=body.temperature,
            )
            ready = fader.push(to_float(wav))
            if ready.size:
                log.info(
                    "synth chunk %s/%s ready samples=%s",
                    index,
                    len(parts),
                    int(ready.size),
                )
                yield pcm16(ready)
        tail = fader.flush()
        if tail.size:
            yield pcm16(tail)


def _media_type(fmt: str) -> str:
    return {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}[fmt]


def _audio_headers(sample_rate: int, language: str, speed: float) -> dict[str, str]:
    return {
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
        "Accept-Ranges": "none",
        "X-Sample-Rate": str(sample_rate),
        "X-Language": language,
        "X-Speed": f"{speed:.2f}",
    }


@app.get("/health")
def health():
    loaded = _model is not None
    if loaded:
        status = "ok"
    elif _skip_model():
        status = "skipped"
    else:
        status = "loading"
    return {
        "status": status,
        "device": _device,
        "cuda": _device == "cuda",
        "t3_model": T3_MODEL,
        "default_language": DEFAULT_LANGUAGE,
        "sample_rate": int(_model.sr) if loaded else MODEL_SAMPLE_RATE,
        "progressive_format": "ogg",
        "speed": {
            "min": SPEED_MIN,
            "max": SPEED_MAX,
            "default": 1.0,
            "method": "ffmpeg-atempo",
        },
    }


@api.get("/v1/languages")
def languages():
    return {"languages": SUPPORTED_LANGUAGES, "default": DEFAULT_LANGUAGE}


@api.get("/v1/voices", response_model=VoicesResponse)
def voices():
    return {"voices": list_voices()}


@api.post("/v1/audio/speech")
@api.post("/tts")
def speech(body: SpeechRequest):
    text = resolve_text(body)
    language = resolve_language(body)
    voice_path = resolve_voice(body.voice)
    fmt = body.response_format.lower()
    if fmt not in {"wav", "mp3", "ogg"}:
        raise HTTPException(
            status_code=400, detail="response_format must be wav, ogg, or mp3"
        )
    if body.stream and fmt == "mp3":
        raise HTTPException(
            status_code=400,
            detail="stream supports ogg (players) and wav (download). mp3 is not streamed",
        )
    if _pcm_override is None and _skip_model():
        raise HTTPException(status_code=503, detail="TTS model is not loaded")
    if _pcm_override is None:
        load_model()

    sample_rate = active_sample_rate()
    headers = _audio_headers(sample_rate, language, body.speed)

    if body.stream:
        chunks = iter_synthesis_pcm(
            text, language, voice_path, body, eager_first=True
        )
        return StreamingResponse(
            iter_pcm_stream(chunks, sample_rate, body.speed, fmt),
            media_type=_media_type(fmt),
            headers=headers,
        )

    try:
        pcm = b"".join(
            iter_synthesis_pcm(text, language, voice_path, body, eager_first=False)
        )
        pcm = apply_speed_pcm(pcm, sample_rate, body.speed)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if fmt == "wav":
        payload = wav_header(sample_rate, len(pcm)) + pcm
    elif fmt == "mp3":
        try:
            payload = encode_mp3(pcm, sample_rate)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    else:
        try:
            payload = encode_ogg(pcm, sample_rate, speed=1.0)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=payload, media_type=_media_type(fmt), headers=headers)


app.include_router(api)
