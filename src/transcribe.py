"""WhisperX transcription logic with speaker diarization."""

import logging
import os
from typing import Any

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading on each request
_model_cache: dict[str, Any] = {}


def get_device() -> str:
    """Get the best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_compute_type() -> str:
    """Get compute type from environment or default."""
    return os.environ.get("COMPUTE_TYPE", "float16")


def get_whisper_model_name() -> str:
    """Get Whisper model name from environment or default."""
    return os.environ.get("WHISPER_MODEL", "large-v3")


def load_models(device: str | None = None, compute_type: str | None = None) -> None:
    """Pre-load models into cache for faster inference.

    Args:
        device: Device to load models on ("cuda" or "cpu").
        compute_type: Compute type for faster-whisper ("float16", "int8", etc.).
    """
    device = device or get_device()
    compute_type = compute_type or get_compute_type()
    model_name = get_whisper_model_name()

    if "whisper" not in _model_cache:
        logger.info(f"Loading Whisper model: {model_name} on {device}")
        _model_cache["whisper"] = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
        )
        logger.info("Whisper model loaded")

    if "diarize" not in _model_cache:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable is required for diarization"
            )

        logger.info("Loading diarization pipeline")
        _model_cache["diarize"] = DiarizationPipeline(
            use_auth_token=hf_token,
            device=device,
        )
        logger.info("Diarization pipeline loaded")


def transcribe_audio(
    audio_path: str,
    language: str = "auto",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict[str, Any]:
    """Transcribe audio with WhisperX and speaker diarization.

    Args:
        audio_path: Path to the audio file.
        language: Language code ("fr", "en") or "auto" for detection.
        min_speakers: Minimum number of speakers (optional).
        max_speakers: Maximum number of speakers (optional).

    Returns:
        Dictionary containing:
            - segments: List of transcription segments with speaker info
            - speakers: List of unique speaker IDs
            - language: Detected or specified language
            - duration: Audio duration in seconds
    """
    device = get_device()
    compute_type = get_compute_type()

    # Ensure models are loaded
    load_models(device, compute_type)

    whisper_model = _model_cache["whisper"]
    diarize_pipeline = _model_cache["diarize"]

    # Load audio
    logger.info(f"Loading audio from {audio_path}")
    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / 16000  # whisperx uses 16kHz

    # Transcribe
    logger.info("Running transcription")
    transcribe_options = {"batch_size": 16}
    if language and language != "auto":
        transcribe_options["language"] = language

    result = whisper_model.transcribe(audio, **transcribe_options)
    detected_language = result.get(
        "language", language if language != "auto" else "unknown"
    )

    # Align whisper output
    logger.info(f"Aligning transcription (language: {detected_language})")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Clean up align model to free memory
    del align_model
    torch.cuda.empty_cache() if device == "cuda" else None

    # Diarize
    logger.info("Running speaker diarization")
    diarize_kwargs = {}
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    diarize_segments = diarize_pipeline(audio, **diarize_kwargs)

    # Assign speakers to segments
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Extract unique speakers
    speakers = sorted(
        set(
            seg.get("speaker", "UNKNOWN")
            for seg in result.get("segments", [])
            if seg.get("speaker")
        )
    )

    # Format segments for output
    segments = _format_segments(result.get("segments", []))

    return {
        "segments": segments,
        "speakers": speakers,
        "language": detected_language,
        "duration": duration,
    }


def _format_segments(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format raw WhisperX segments to output format.

    Args:
        raw_segments: Raw segments from WhisperX.

    Returns:
        Formatted segments matching the API spec.
    """
    formatted = []
    for seg in raw_segments:
        formatted_seg = {
            "start": round(seg.get("start", 0), 3),
            "end": round(seg.get("end", 0), 3),
            "text": seg.get("text", "").strip(),
            "speaker": seg.get("speaker", "UNKNOWN"),
        }

        # Include word-level timestamps if available
        if "words" in seg:
            formatted_seg["words"] = [
                {
                    "word": w.get("word", ""),
                    "start": round(w.get("start", 0), 3),
                    "end": round(w.get("end", 0), 3),
                }
                for w in seg["words"]
                if "word" in w
            ]

        formatted.append(formatted_seg)

    return formatted
