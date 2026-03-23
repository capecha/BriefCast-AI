from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def load_voices_json(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_numpy(audio):
    if hasattr(audio, "detach"):
        return audio.detach().cpu().numpy()
    return np.asarray(audio)


def synthesize_with_kokoro(
    turns: list,
    voice_a: str,
    voice_b: str,
    output_wav: str | Path,
    lang_code: str = "a",
    speed: float = 1.0,
):
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=lang_code)
    sample_rate = 24000
    pieces = []

    def speak(text: str, voice_id: str):
        text = (text or "").strip()
        if not text:
            return

        for _, _, audio in pipeline(text, voice=voice_id, speed=speed):
            pieces.append(np.asarray(_to_numpy(audio), dtype=np.float32))

        pieces.append(np.zeros(int(sample_rate * 0.35), dtype=np.float32))

    for turn in turns:
        speaker = turn.get("speaker", "").strip().lower()
        text = turn.get("text", "")

        voice_id = voice_a if speaker in ["host a", "a", "speaker a"] else voice_b
        speak(text, voice_id)

    if not pieces:
        raise ValueError("No audio was generated from the provided turns.")

    final_audio = np.concatenate(pieces, axis=0)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, final_audio, sample_rate)
    return output_wav


def wav_to_mp3(input_wav: str | Path, output_mp3: str | Path):
    input_wav = Path(input_wav)
    output_mp3 = Path(output_mp3)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not available in PATH.")

    output_mp3.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        str(output_mp3),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_mp3


def synthesize_dialogue_from_labels(
    turns: list,
    host_a_label: str,
    host_b_label: str,
    voices_json_path: str | Path,
    output_wav: str | Path,
    lang_code: str = "a",
    speed: float = 1.0,
    output_mp3: str | Path | None = None,
):
    voices = load_voices_json(voices_json_path)

    if host_a_label not in voices:
        raise ValueError(f"Voice label not found for Host A: {host_a_label}")
    if host_b_label not in voices:
        raise ValueError(f"Voice label not found for Host B: {host_b_label}")

    voice_a = voices[host_a_label]
    voice_b = voices[host_b_label]

    wav_path = synthesize_with_kokoro(
        turns=turns,
        voice_a=voice_a,
        voice_b=voice_b,
        output_wav=output_wav,
        lang_code=lang_code,
        speed=speed,
    )

    result = {"wav_path": str(wav_path)}

    if output_mp3 is not None:
        mp3_path = wav_to_mp3(wav_path, output_mp3)
        result["mp3_path"] = str(mp3_path)

    return result