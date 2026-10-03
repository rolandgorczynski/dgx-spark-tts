"""Audio chunking and progressive encoding for the Spark TTS server.

Chatterbox Multilingual has no speaking-rate argument. ``speed`` is applied
with ffmpeg ``atempo`` (pitch-preserving, 0.5–2.0). 1.0 leaves samples as
the model emitted them.

Ogg Opus is the progressive format. With ``-probesize 32 -analyzeduration 0``
ffmpeg emits pages as soon as the first PCM chunk arrives, and ExoPlayer
treats a chunked response (no Content-Length) as an unseekable Ogg stream:
playback starts on the first pages instead of waiting for the last sentence.

WAV streams still use a placeholder data length (``0x7FFFFF00``). Media3's
WavExtractor treats that length as the real end of the file, so a long article
looks like hours of audio. Keep WAV for ``curl``; point players at Ogg.
"""

from __future__ import annotations

import re
import struct
import subprocess
import threading
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

MODEL_SAMPLE_RATE = 24000  # chatterbox.models.s3gen.const.S3GEN_SR
SPEED_MIN = 0.5
SPEED_MAX = 2.0
# Large positive signed length. Players that read until the socket closes
# still hear audio; ExoPlayer should use Ogg instead (see module docstring).
STREAMING_WAV_DATA_BYTES = 0x7FFFFF00
VOICE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

_SENTENCE = re.compile(r"(?<=[.!?…])\s+|\n+")


def list_voices(directory: Path) -> list[dict]:
    """Builtin voice plus audio files in ``directory`` (filename is the id)."""
    voices: list[dict] = [{"id": "builtin", "name": "Eingebaut", "builtin": True}]
    if not directory.is_dir():
        return voices
    names: list[str] = []
    for path in directory.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in VOICE_EXTENSIONS:
            continue
        names.append(path.name)
    for name in sorted(names, key=str.lower):
        voices.append({"id": name, "name": Path(name).stem, "builtin": False})
    return voices


def split_text(text: str, limit: int, *, eager_first: bool = False) -> list[str]:
    """Split text on sentence endings and newlines, then pack up to ``limit``.

    ``eager_first`` keeps the first piece alone so a stream can start after one
    sentence instead of waiting to fill ``limit`` characters.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    parts = [p.strip() for p in _SENTENCE.split(normalized) if p.strip()]
    if not parts:
        return [normalized]
    pieces: list[str] = []
    for part in parts:
        pieces.extend(_cut_long(part, limit))
    if not pieces:
        return [normalized]
    if eager_first:
        if len(pieces) == 1:
            return pieces
        rest = _pack(pieces[1:], limit)
        return [pieces[0], *rest]
    return _pack(pieces, limit)


def _cut_long(part: str, limit: int) -> list[str]:
    out: list[str] = []
    while len(part) > limit:
        cut = part.rfind(" ", 0, limit + 1)
        if cut < max(1, limit // 2):
            cut = limit
        piece = part[:cut].strip()
        if piece:
            out.append(piece)
        part = part[cut:].strip()
    if part:
        out.append(part)
    return out


def _pack(pieces: list[str], limit: int) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for part in pieces:
        if buf and len(buf) + 1 + len(part) > limit:
            chunks.append(buf)
            buf = part
        elif buf:
            buf = f"{buf} {part}"
        else:
            buf = part
    if buf:
        chunks.append(buf)
    return chunks


def pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def wav_header(sample_rate: int, data_bytes: int, *, channels: int = 1, bits: int = 16) -> bytes:
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_bytes,
    )


def streaming_wav_header(sample_rate: int) -> bytes:
    return wav_header(sample_rate, STREAMING_WAV_DATA_BYTES)


class PcmCrossfader:
    """20 ms overlap-add that matches a one-shot crossfade, but emits early.

    The first chunk is returned immediately except for the overlap tail, so a
    stream does not wait for the next sentence to play the current one.
    """

    def __init__(self, sample_rate: int, ms: int = 20):
        self.n = max(1, int(sample_rate * ms / 1000))
        self.pending: Optional[np.ndarray] = None

    def push(self, audio: np.ndarray) -> np.ndarray:
        block = np.asarray(audio, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self.pending is None:
            merged = block
        elif self.pending.size < self.n or block.size < self.n:
            merged = np.concatenate([self.pending, block])
        else:
            fade_out = np.linspace(1.0, 0.0, self.n, dtype=np.float32)
            fade_in = 1.0 - fade_out
            mixed = self.pending[-self.n :] * fade_out + block[: self.n] * fade_in
            merged = np.concatenate([self.pending[: -self.n], mixed, block[self.n :]])
        if merged.size <= self.n:
            self.pending = merged
            return np.zeros(0, dtype=np.float32)
        self.pending = merged[-self.n :].copy()
        return merged[: -self.n].copy()

    def flush(self) -> np.ndarray:
        tail = self.pending
        self.pending = None
        if tail is None or tail.size == 0:
            return np.zeros(0, dtype=np.float32)
        return tail.copy()


def crossfade(pieces: list[np.ndarray], sample_rate: int, ms: int = 20) -> np.ndarray:
    fader = PcmCrossfader(sample_rate, ms=ms)
    parts: list[np.ndarray] = []
    for piece in pieces:
        ready = fader.push(piece)
        if ready.size:
            parts.append(ready)
    tail = fader.flush()
    if tail.size:
        parts.append(tail)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def needs_tempo(speed: float) -> bool:
    return abs(speed - 1.0) >= 0.01


def atempo_filter(speed: float) -> str:
    """ffmpeg atempo chain. Each stage accepts 0.5–2.0."""
    remaining = float(speed)
    if remaining <= 0:
        raise ValueError("speed must be positive")
    stages: list[str] = []
    while remaining > SPEED_MAX + 1e-6:
        stages.append("atempo=2.0")
        remaining /= 2.0
    while remaining < SPEED_MIN - 1e-6:
        stages.append("atempo=0.5")
        remaining /= 0.5
    stages.append(f"atempo={remaining:.4f}")
    return ",".join(stages)


def _ffmpeg_base(sample_rate: int) -> list[str]:
    # probesize/analyzeduration stop ffmpeg from buffering the whole pipe
    # before it writes the first packet. Without them, pages appear only at EOF.
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        "-fflags",
        "+nobuffer",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
    ]


def iter_ogg_opus(
    pcm_chunks: Iterator[bytes], sample_rate: int, speed: float = 1.0
) -> Iterator[bytes]:
    """Encode s16le mono PCM to Ogg Opus, flushing pages per input chunk."""
    cmd = _ffmpeg_base(sample_rate)
    if needs_tempo(speed):
        cmd += ["-filter:a", atempo_filter(speed)]
    cmd += [
        "-c:a",
        "libopus",
        "-b:a",
        "32k",
        "-application",
        "voip",
        "-frame_duration",
        "20",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
        "-f",
        "ogg",
        "-page_duration",
        "20000",
        "-flush_packets",
        "1",
        "pipe:1",
    ]
    yield from _pipe_ffmpeg(cmd, pcm_chunks)


def iter_atempo_pcm(
    pcm_chunks: Iterator[bytes], sample_rate: int, speed: float
) -> Iterator[bytes]:
    cmd = _ffmpeg_base(sample_rate)
    cmd += [
        "-filter:a",
        atempo_filter(speed),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-flush_packets",
        "1",
        "pipe:1",
    ]
    yield from _pipe_ffmpeg(cmd, pcm_chunks)


def apply_speed_pcm(pcm: bytes, sample_rate: int, speed: float) -> bytes:
    if not pcm or not needs_tempo(speed):
        return pcm
    return b"".join(iter_atempo_pcm(iter([pcm]), sample_rate, speed))


def encode_mp3(pcm: bytes, sample_rate: int) -> bytes:
    cmd = _ffmpeg_base(sample_rate) + [
        "-f",
        "mp3",
        "-b:a",
        "192k",
        "pipe:1",
    ]
    return b"".join(_pipe_ffmpeg(cmd, iter([pcm])))


def encode_ogg(pcm: bytes, sample_rate: int, speed: float = 1.0) -> bytes:
    return b"".join(iter_ogg_opus(iter([pcm]), sample_rate, speed))


def iter_pcm_stream(
    pcm_chunks: Iterator[bytes], sample_rate: int, speed: float, response_format: str
) -> Iterator[bytes]:
    fmt = response_format.lower()
    if fmt == "wav" and not needs_tempo(speed):
        yield streaming_wav_header(sample_rate)
        yield from pcm_chunks
        return
    if fmt == "wav":
        yield streaming_wav_header(sample_rate)
        yield from iter_atempo_pcm(pcm_chunks, sample_rate, speed)
        return
    if fmt == "ogg":
        yield from iter_ogg_opus(pcm_chunks, sample_rate, speed)
        return
    raise ValueError(f"unsupported stream format: {response_format}")


def _pipe_ffmpeg(cmd: list[str], pcm_chunks: Iterator[bytes]) -> Iterator[bytes]:
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed") from exc

    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    stderr_parts: list[bytes] = []
    feed_error: list[BaseException] = []

    def drain() -> None:
        stderr_parts.append(proc.stderr.read())

    def feed() -> None:
        try:
            for chunk in pcm_chunks:
                if proc.poll() is not None:
                    break
                if not chunk:
                    continue
                proc.stdin.write(chunk)
                proc.stdin.flush()
        except Exception as exc:
            feed_error.append(exc)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    drainer = threading.Thread(target=drain, name="ffmpeg-stderr", daemon=True)
    feeder = threading.Thread(target=feed, name="ffmpeg-feed", daemon=True)
    drainer.start()
    feeder.start()
    produced = False
    try:
        while True:
            block = proc.stdout.read(4096)
            if not block:
                break
            produced = True
            yield block
        feeder.join(timeout=30)
        if feed_error and not produced:
            raise RuntimeError("audio encode failed") from feed_error[0]
        if not produced:
            detail = b"".join(stderr_parts).decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(
                f"ffmpeg produced no audio (exit {proc.returncode}): {detail}"
            )
    finally:
        if proc.poll() is None:
            proc.kill()
        feeder.join(timeout=5)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        drainer.join(timeout=2)
