"""Offline speech-to-text and text-to-speech helpers."""

from __future__ import annotations

import logging
import platform
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

@lru_cache(maxsize=2)
def get_local_whisper_model(model_size: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(audio: bytes, model: Any) -> str:
    """Transcribe a recorded WAV clip with an already-loaded local Whisper model."""
    if not audio:
        raise ValueError("The audio recording is empty.")
    with tempfile.TemporaryDirectory(prefix="local-docchat-audio-") as directory:
        audio_path = Path(directory) / "recording.wav"
        audio_path.write_bytes(audio)
        try:
            segments, _info = model.transcribe(
                str(audio_path),
                beam_size=1,
                vad_filter=True,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception:
            logger.exception("Local Whisper transcription failed")
            raise
    if not transcript:
        raise ValueError("No speech was detected. Try recording your question again.")
    return transcript


def _transcribe_with_windows_speech(audio: bytes) -> str:
    with tempfile.TemporaryDirectory(prefix="local-docchat-sapi-") as directory:
        audio_path = Path(directory) / "recording.wav"
        audio_path.write_bytes(audio)
        escaped_path = str(audio_path).replace("'", "''")
        script = f"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
try {{
    Add-Type -AssemblyName System.Speech
    $recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
    $recognizer.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
    $recognizer.SetInputToWaveFile('{escaped_path}')
    $result = $recognizer.Recognize()
    if ($null -eq $result -or [string]::IsNullOrWhiteSpace($result.Text)) {{
        throw 'No speech was recognized. Try recording your question again.'
    }}
    [Console]::Write($result.Text)
    $recognizer.Dispose()
}} catch {{
    [Console]::Error.Write($_.Exception.Message)
    exit 1
}}
"""
        try:
            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    script,
                ],
                capture_output=True,
                check=False,
                encoding="utf-8",
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Windows offline speech recognition failed: {exc}") from exc
    if result.returncode:
        message = result.stderr.strip() or "Windows did not recognize the recording."
        raise RuntimeError(message)
    transcript = result.stdout.strip()
    if not transcript:
        raise ValueError("No speech was detected. Try recording your question again.")
    return transcript


def transcribe_locally(audio: bytes, model_size: str) -> tuple[str, str]:
    """Prefer local Whisper; use Windows' offline recognizer if Whisper is unavailable."""
    try:
        model = get_local_whisper_model(model_size)
        return transcribe_audio(audio, model), f"Whisper {model_size}"
    except Exception as exc:
        if platform.system() != "Windows":
            raise RuntimeError(f"Local Whisper transcription failed: {exc}") from exc
        logger.warning(
            "Whisper is unavailable; falling back to Windows offline speech recognition: %s",
            exc,
        )
        return _transcribe_with_windows_speech(audio), "Windows offline speech recognition"


def synthesize_speech(text: str) -> bytes:
    """Return a WAV recording of a response using the operating system's local voice."""
    if not text.strip():
        raise ValueError("There is no answer text to speak.")
    try:
        import pyttsx3

        with tempfile.TemporaryDirectory(prefix="local-docchat-speech-") as directory:
            audio_path = Path(directory) / "answer.wav"
            engine = pyttsx3.init()
            engine.save_to_file(text, str(audio_path))
            engine.runAndWait()
            audio = audio_path.read_bytes()
            engine.stop()
            return audio
    except Exception:
        logger.exception("Local text-to-speech failed")
        raise
