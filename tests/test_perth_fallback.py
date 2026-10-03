"""Startup must not call a missing Perth watermarker. Synthesis still returns audio."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server


class _Wave:
    """Enough of a torch tensor for ``server.to_float``."""

    def __init__(self, samples: np.ndarray):
        self._samples = np.asarray(samples, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._samples


def _install_torch(monkeypatch) -> None:
    try:
        import torch
    except ImportError:
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch)
    else:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(server, "ALLOW_CPU", True)


def _install_chatterbox(monkeypatch, perth) -> None:
    samples = np.full(1000, 0.25, dtype=np.float32)

    class ChatterboxMultilingualTTS:
        sr = 24000

        def __init__(self):
            # Same call as chatterbox.mtl_tts.ChatterboxMultilingualTTS.__init__.
            watermarker_cls = perth.PerthImplicitWatermarker
            assert watermarker_cls is not None
            self.watermarker = watermarker_cls()

        @classmethod
        def from_pretrained(cls, device, t3_model=None):
            assert device in {"cpu", "cuda"}
            assert t3_model == server.T3_MODEL
            return cls()

        def generate(self, text, **_kwargs):
            marked = self.watermarker.apply_watermark(samples, sample_rate=self.sr)
            assert isinstance(marked, np.ndarray)
            assert marked.shape == samples.shape
            return _Wave(marked)

    chatterbox = types.ModuleType("chatterbox")
    chatterbox.__path__ = []
    mtl = types.ModuleType("chatterbox.mtl_tts")
    mtl.SUPPORTED_LANGUAGES = dict(server.SUPPORTED_LANGUAGES)
    mtl.ChatterboxMultilingualTTS = ChatterboxMultilingualTTS
    monkeypatch.setitem(sys.modules, "chatterbox", chatterbox)
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", mtl)
    monkeypatch.setitem(sys.modules, "perth", perth)


@pytest.fixture
def unloaded_model():
    previous = (server._model, server._device)
    server._model = None
    server._device = None
    yield
    server._model, server._device = previous


def test_none_watermarker_still_synthesizes(monkeypatch, unloaded_model):
    perth = types.ModuleType("perth")
    perth.PerthImplicitWatermarker = None
    _install_torch(monkeypatch)
    _install_chatterbox(monkeypatch, perth)
    monkeypatch.setenv("TTS_SKIP_MODEL", "")

    with TestClient(server.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert isinstance(server._model.watermarker, server._PassthroughWatermarker)
        assert perth.PerthImplicitWatermarker is server._PassthroughWatermarker

        response = client.post(
            "/v1/audio/speech",
            json={"input": "Guten Tag. Das ist ein Test.", "language": "de"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content[8:12] == b"WAVE"
    pcm = np.frombuffer(response.content[44:], dtype="<i2")
    assert pcm.size == 1000
    assert np.all(pcm == np.int16(0.25 * 32767))


def test_working_watermarker_is_left_in_place(monkeypatch):
    perth = types.ModuleType("perth")

    class Working:
        def apply_watermark(self, wav, sample_rate=None, **_kwargs):
            return wav

    perth.PerthImplicitWatermarker = Working
    monkeypatch.setitem(sys.modules, "perth", perth)

    server._ensure_perth_watermarker()

    assert perth.PerthImplicitWatermarker is Working
    audio = np.array([0.5, -0.5], dtype=np.float32)
    assert np.shares_memory(
        Working().apply_watermark(audio, sample_rate=24000), audio
    )
