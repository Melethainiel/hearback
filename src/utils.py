"""Utility functions for audio download, output formatting, and cleanup."""

import logging
import os
import subprocess
import tempfile
from typing import Any

import requests

logger = logging.getLogger(__name__)


def download_audio(url: str, timeout: int = 300) -> str:
    """Download audio from URL to a temporary file.

    Args:
        url: URL of the audio file to download.
        timeout: Request timeout in seconds.

    Returns:
        Path to the downloaded temporary file.

    Raises:
        requests.RequestException: If download fails.
        ValueError: If URL is invalid or empty.
    """
    if not url:
        raise ValueError("Audio URL cannot be empty")

    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    # Determine file extension from content-type or URL
    content_type = response.headers.get("content-type", "")
    ext = _get_extension_from_content_type(content_type)
    if not ext:
        ext = os.path.splitext(url.split("?")[0])[1] or ".wav"

    # Create temp file with appropriate extension
    fd, temp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception:
        os.unlink(temp_path)
        raise

    return temp_path


def convert_to_wav(input_path: str) -> str:
    """Convert audio file to WAV format using ffmpeg.

    Args:
        input_path: Path to the input audio file.

    Returns:
        Path to the converted WAV file.

    Raises:
        RuntimeError: If ffmpeg conversion fails.
    """
    fd, output_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i",
            input_path,
            "-ar",
            "16000",  # 16kHz sample rate (whisperx expects this)
            "-ac",
            "1",  # Mono
            "-c:a",
            "pcm_s16le",  # 16-bit PCM
            output_path,
        ]
        logger.info(f"Converting audio to WAV: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

        logger.info(f"Audio converted to {output_path}")
        return output_path

    except Exception:
        # Clean up on failure
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise


def _get_extension_from_content_type(content_type: str) -> str:
    """Map content-type to file extension."""
    mapping = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/ogg": ".ogg",
        "audio/opus": ".opus",
        "audio/flac": ".flac",
        "audio/webm": ".webm",
    }
    return mapping.get(content_type.split(";")[0].strip().lower(), "")


def format_output(
    segments: list[dict[str, Any]],
    speakers: list[str],
    language: str,
    duration: float,
    processing_time: float,
    output_format: str = "json",
) -> dict[str, Any] | str:
    """Format transcription output to requested format.

    Args:
        segments: List of transcription segments with speaker info.
        speakers: List of unique speaker IDs.
        language: Detected or specified language.
        duration: Audio duration in seconds.
        processing_time: Processing time in seconds.
        output_format: Output format - "json", "srt", or "vtt".

    Returns:
        Formatted output as dict (json) or string (srt/vtt).

    Raises:
        ValueError: If output_format is not supported.
    """
    if output_format == "json":
        return _format_json(segments, speakers, language, duration, processing_time)
    elif output_format == "srt":
        return _format_srt(segments)
    elif output_format == "vtt":
        return _format_vtt(segments)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def _format_json(
    segments: list[dict[str, Any]],
    speakers: list[str],
    language: str,
    duration: float,
    processing_time: float,
) -> dict[str, Any]:
    """Format output as JSON structure per spec."""
    full_text = " ".join(seg.get("text", "").strip() for seg in segments)

    return {
        "transcription": {
            "text": full_text,
            "segments": segments,
        },
        "speakers": speakers,
        "language": language,
        "duration": round(duration, 2),
        "processing_time": round(processing_time, 2),
    }


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds to VTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_srt(segments: list[dict[str, Any]]) -> str:
    """Format segments as SRT subtitle format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp_srt(seg.get("start", 0))
        end = _format_timestamp_srt(seg.get("end", 0))
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()

        speaker_prefix = f"[{speaker}] " if speaker else ""
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(f"{speaker_prefix}{text}")
        lines.append("")

    return "\n".join(lines)


def _format_vtt(segments: list[dict[str, Any]]) -> str:
    """Format segments as WebVTT subtitle format."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp_vtt(seg.get("start", 0))
        end = _format_timestamp_vtt(seg.get("end", 0))
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()

        speaker_prefix = f"<v {speaker}>" if speaker else ""
        lines.append(f"{start} --> {end}")
        lines.append(f"{speaker_prefix}{text}")
        lines.append("")

    return "\n".join(lines)


def cleanup_temp_files(*paths: str | None) -> None:
    """Remove temporary files, ignoring errors.

    Args:
        paths: File paths to remove (None values are ignored).
    """
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
