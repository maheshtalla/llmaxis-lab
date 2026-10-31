from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from docchat.voice import (
    synthesize_speech,
    transcribe_audio,
    transcribe_locally,
)


class FakeWhisper:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.audio_path: Path | None = None
        self.options: dict[str, Any] = {}

    def transcribe(self, audio_path: str, **options: Any) -> tuple[list[Any], object]:
        self.audio_path = Path(audio_path)
        self.options = options
        assert self.audio_path.read_bytes() == b"recorded audio"
        return [SimpleNamespace(text=self.transcript)], object()


def test_transcribes_recording_using_local_whisper() -> None:
    model = FakeWhisper("  What is the launch date?  ")

    result = transcribe_audio(b"recorded audio", model)

    assert result == "What is the launch date?"
    assert model.options == {"beam_size": 1, "vad_filter": True}
    assert model.audio_path is not None
    assert not model.audio_path.exists()


def test_transcription_rejects_empty_audio_and_silence() -> None:
    model = FakeWhisper("  ")

    with pytest.raises(ValueError, match="recording is empty"):
        transcribe_audio(b"", model)
    with pytest.raises(ValueError, match="No speech"):
        transcribe_audio(b"recorded audio", model)


def test_local_transcription_prefers_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeWhisper("A question.")
    monkeypatch.setattr("docchat.voice.get_local_whisper_model", lambda _size: model)

    assert transcribe_locally(b"recorded audio", "tiny") == ("A question.", "Whisper tiny")


def test_windows_transcription_falls_back_when_whisper_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("docchat.voice.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "docchat.voice.get_local_whisper_model",
        lambda _size: (_ for _ in ()).throw(ImportError("blocked native DLL")),
    )
    monkeypatch.setattr(
        "docchat.voice._transcribe_with_windows_speech",
        lambda _audio: "Local Windows transcription.",
    )

    assert transcribe_locally(b"recorded audio", "tiny") == (
        "Local Windows transcription.",
        "Windows offline speech recognition",
    )


def test_non_windows_transcription_surfaces_whisper_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("docchat.voice.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "docchat.voice.get_local_whisper_model",
        lambda _size: (_ for _ in ()).throw(ImportError("model unavailable")),
    )

    with pytest.raises(RuntimeError, match="model unavailable"):
        transcribe_locally(b"recorded audio", "tiny")


def test_synthesizes_speech_to_local_wav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engines: list[Any] = []

    class FakeEngine:
        def __init__(self) -> None:
            self.path: str | None = None
            self.text: str | None = None

        def save_to_file(self, text: str, path: str) -> None:
            self.text = text
            self.path = path

        def runAndWait(self) -> None:
            assert self.path is not None
            Path(self.path).write_bytes(b"wav audio")

        def stop(self) -> None:
            pass

    def init() -> FakeEngine:
        engine = FakeEngine()
        engines.append(engine)
        return engine

    monkeypatch.setitem(sys.modules, "pyttsx3", ModuleType("pyttsx3"))
    monkeypatch.setattr(sys.modules["pyttsx3"], "init", init, raising=False)

    assert synthesize_speech("A local answer.") == b"wav audio"
    assert engines[0].text == "A local answer."
    assert engines[0].path is not None
    assert not Path(engines[0].path).exists()


def test_speech_rejects_blank_answer() -> None:
    with pytest.raises(ValueError, match="no answer text"):
        synthesize_speech("  ")
