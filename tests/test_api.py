"""API and streaming tests. They synthesize PCM locally and do not load the model."""

from __future__ import annotations

import math
import socket
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from tts_audio import (
    MODEL_SAMPLE_RATE,
    STREAMING_WAV_DATA_BYTES,
    PcmCrossfader,
    iter_ogg_opus,
    iter_pcm_stream,
    split_text,
)


def tone(sample_rate: int, seconds: float, freq: float = 440.0) -> bytes:
    count = int(sample_rate * seconds)
    frames = bytearray()
    for i in range(count):
        sample = int(8000 * math.sin(2 * math.pi * freq * i / sample_rate))
        frames += struct.pack("<h", sample)
    return bytes(frames)


def ogg_duration(data: bytes) -> float:
    with tempfile.NamedTemporaryFile(suffix=".ogg") as handle:
        handle.write(data)
        handle.flush()
        out = subprocess.check_output(
            [
                "ffprobe",
                "-hide_banner",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                handle.name,
            ],
            text=True,
        )
    return float(out.strip())


@pytest.fixture
def client():
    server._pcm_override = None
    with TestClient(server.app) as test_client:
        yield test_client
    server._pcm_override = None


def test_split_eager_first_sentence_is_not_packed():
    text = "Erster Satz ist kurz. " + ("Wort " * 80) + "Ende."
    chunks = split_text(text, 400, eager_first=True)
    assert chunks[0] == "Erster Satz ist kurz."
    assert len(chunks) >= 2
    packed = split_text("Eins. Zwei. Drei.", 400, eager_first=False)
    assert packed == ["Eins. Zwei. Drei."]
    eager = split_text("Eins. Zwei. Drei.", 400, eager_first=True)
    assert eager[0] == "Eins."
    assert "Zwei." in eager[1]


def test_split_newlines_and_long_sentence():
    chunks = split_text("Zeile eins\nZeile zwei. Und drei.", 400, eager_first=True)
    assert chunks[0] == "Zeile eins"
    assert "Zeile zwei." in chunks[1]
    long = "wort" * 100
    pieces = split_text(long, 40, eager_first=True)
    assert all(len(part) <= 40 for part in pieces)
    assert "".join(pieces).replace(" ", "") == long


def test_crossfade_matches_reference():
    def reference(pieces: list[np.ndarray], n: int) -> np.ndarray:
        out = pieces[0]
        for nxt in pieces[1:]:
            if len(out) < n or len(nxt) < n:
                out = np.concatenate([out, nxt])
                continue
            fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
            fade_in = 1.0 - fade_out
            mixed = out[-n:] * fade_out + nxt[:n] * fade_in
            out = np.concatenate([out[:-n], mixed, nxt[n:]])
        return out

    rng = np.random.default_rng(0)
    pieces = [rng.normal(size=size).astype(np.float32) for size in (5, 50, 10, 800, 30)]
    fader = PcmCrossfader(sample_rate=500, ms=20)  # 10-sample overlap
    assert fader.n == 10
    streamed = [fader.push(piece) for piece in pieces]
    streamed.append(fader.flush())
    got = np.concatenate([part for part in streamed if part.size])
    assert np.allclose(got, reference(pieces, 10))


def test_ogg_pages_arrive_before_second_pcm_chunk():
    release = threading.Event()
    state = {"second": False}

    def chunks():
        yield tone(MODEL_SAMPLE_RATE, 0.5, 440)
        release.wait(timeout=8)
        state["second"] = True
        yield tone(MODEL_SAMPLE_RATE, 0.5, 660)

    collected = bytearray()

    def consume():
        for block in iter_ogg_opus(chunks(), MODEL_SAMPLE_RATE, 1.0):
            collected.extend(block)

    worker = threading.Thread(target=consume)
    worker.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and collected.count(b"OggS") < 4:
            time.sleep(0.02)
        assert collected.count(b"OggS") >= 4
        assert state["second"] is False
    finally:
        release.set()
        worker.join(timeout=8)
    assert worker.is_alive() is False
    assert collected.startswith(b"OggS")
    assert 0.8 <= ogg_duration(collected) <= 1.3


def test_speed_stretches_ogg_duration():
    def once(speed: float) -> float:
        data = b"".join(
            iter_ogg_opus(iter([tone(MODEL_SAMPLE_RATE, 1.0)]), MODEL_SAMPLE_RATE, speed)
        )
        return ogg_duration(data)

    normal = once(1.0)
    faster = once(2.0)
    assert 0.85 <= normal <= 1.25
    assert 0.35 <= faster <= 0.7
    assert faster < normal * 0.7


def test_wav_stream_header_before_second_chunk():
    release = threading.Event()

    def chunks():
        yield tone(MODEL_SAMPLE_RATE, 0.05)
        assert release.wait(timeout=2)
        yield tone(MODEL_SAMPLE_RATE, 0.05)

    gen = iter_pcm_stream(chunks(), MODEL_SAMPLE_RATE, 1.0, "wav")
    header = next(gen)
    assert header.startswith(b"RIFF")
    assert struct.unpack_from("<I", header, 40)[0] == STREAMING_WAV_DATA_BYTES
    first_pcm = next(gen)
    assert len(first_pcm) == int(MODEL_SAMPLE_RATE * 0.05) * 2
    release.set()
    assert b"".join(gen)


def test_voices_lists_builtin_and_files(client, tmp_path: Path, monkeypatch):
    (tmp_path / "Anna.wav").write_bytes(b"RIFF")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / ".hidden.wav").write_bytes(b"RIFF")
    (tmp_path / "jan.ogg").write_bytes(b"OggS")
    monkeypatch.setattr(server, "VOICES_DIR", tmp_path)
    response = client.get("/v1/voices")
    assert response.status_code == 200
    voices = response.json()["voices"]
    assert voices[0] == {"id": "builtin", "name": "Eingebaut", "builtin": True}
    assert [item["id"] for item in voices[1:]] == ["Anna.wav", "jan.ogg"]


def test_missing_voice_and_path_rejected(client, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(server, "VOICES_DIR", tmp_path)
    missing = client.post("/v1/audio/speech", json={"input": "Hallo.", "voice": "nope.wav"})
    assert missing.status_code == 404
    traversal = client.post(
        "/v1/audio/speech", json={"input": "Hallo.", "voice": "../secrets.wav"}
    )
    assert traversal.status_code == 400


def test_language_polish_and_speed_bounds(client):
    server._pcm_override = lambda *_args, **_kwargs: iter([tone(MODEL_SAMPLE_RATE, 0.1)])
    ok = client.post(
        "/tts",
        json={"text": "Dzień dobry. To jest test.", "language": "pl", "response_format": "wav"},
    )
    assert ok.status_code == 200
    assert ok.headers["x-language"] == "pl"
    assert ok.content[8:12] == b"WAVE"
    data_bytes = struct.unpack_from("<I", ok.content, 40)[0]
    assert data_bytes == len(ok.content) - 44

    bad = client.post("/v1/audio/speech", json={"input": "x", "language": "xx"})
    assert bad.status_code == 400
    too_fast = client.post("/v1/audio/speech", json={"input": "x", "speed": 3})
    assert too_fast.status_code == 422


def test_nonstream_speed_shortens_wav(client):
    pcm = tone(MODEL_SAMPLE_RATE, 1.0)

    def override(*_args, **_kwargs):
        yield pcm

    server._pcm_override = override
    normal = client.post(
        "/v1/audio/speech",
        json={"input": "Hallo.", "response_format": "wav", "speed": 1},
    )
    faster = client.post(
        "/v1/audio/speech",
        json={"input": "Hallo.", "response_format": "wav", "speed": 2},
    )
    assert normal.status_code == 200
    assert faster.status_code == 200
    normal_pcm = len(normal.content) - 44
    faster_pcm = len(faster.content) - 44
    assert normal_pcm == len(pcm)
    assert 0.4 * normal_pcm < faster_pcm < 0.7 * normal_pcm


def test_stream_mp3_rejected_ogg_accepted(client):
    server._pcm_override = lambda *_a, **_k: iter([tone(MODEL_SAMPLE_RATE, 0.2)])
    rejected = client.post(
        "/v1/audio/speech",
        json={"input": "Hallo.", "stream": True, "response_format": "mp3"},
    )
    assert rejected.status_code == 400
    accepted = client.post(
        "/v1/audio/speech",
        json={"input": "Erster Satz. Zweiter.", "stream": True, "response_format": "ogg"},
    )
    assert accepted.status_code == 200
    assert accepted.headers["content-type"].startswith("audio/ogg")
    assert "content-length" not in {key.lower() for key in accepted.headers}
    assert accepted.content.startswith(b"OggS")
    assert ogg_duration(accepted.content) > 0.05


def test_optional_token(client, monkeypatch):
    monkeypatch.setenv("TTS_API_TOKEN", "secret-token")
    assert client.get("/health").status_code == 200
    assert client.get("/v1/voices").status_code == 401
    denied = client.get("/v1/voices", headers={"Authorization": "Bearer wrong"})
    assert denied.status_code == 401
    ok = client.get("/v1/voices", headers={"X-API-Key": "secret-token"})
    assert ok.status_code == 200
    bearer = client.get("/v1/languages", headers={"Authorization": "Bearer secret-token"})
    assert bearer.status_code == 200
    assert "pl" in bearer.json()["languages"]
    assert "de" in bearer.json()["languages"]


def test_http_ogg_is_progressive_before_second_sentence():
    """The socket must deliver Ogg pages before the second PCM chunk is produced."""
    release = threading.Event()
    state = {"second": False}

    def override(*_args, **_kwargs):
        yield tone(MODEL_SAMPLE_RATE, 0.5, 440)
        release.wait(timeout=8)
        state["second"] = True
        yield tone(MODEL_SAMPLE_RATE, 0.5, 880)

    server._pcm_override = override
    port_sock = socket.socket()
    port_sock.bind(("127.0.0.1", 0))
    port = port_sock.getsockname()[1]
    port_sock.close()

    import uvicorn

    config = uvicorn.Config(
        server.app, host="127.0.0.1", port=port, log_level="warning"
    )
    running = uvicorn.Server(config)
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    deadline = time.time() + 8
    while time.time() < deadline and not running.started:
        time.sleep(0.05)
    assert running.started

    import httpx

    payload = {
        "input": "Erster Satz des Artikels. Zweiter Satz folgt erst danach.",
        "language": "de",
        "voice": "builtin",
        "speed": 1.0,
        "stream": True,
        "response_format": "ogg",
    }
    collected = bytearray()
    try:
        with httpx.stream(
            "POST",
            f"http://127.0.0.1:{port}/v1/audio/speech",
            json=payload,
            timeout=httpx.Timeout(10.0, read=10.0),
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("audio/ogg")
            assert response.headers.get("content-length") is None
            for block in response.iter_bytes():
                collected.extend(block)
                if collected.count(b"OggS") >= 4:
                    break
            assert collected.count(b"OggS") >= 4
            assert state["second"] is False
            assert collected.startswith(b"OggS")
    finally:
        release.set()
        running.should_exit = True
        thread.join(timeout=8)
        server._pcm_override = None
