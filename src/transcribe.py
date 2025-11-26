"""WhisperX transcription logic with speaker diarization."""

import logging
import os
import shutil
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


def log_system_stats() -> None:
    """Log system resource statistics."""
    # Disk space
    disk = shutil.disk_usage("/")
    disk_free_gb = disk.free / (1024**3)
    disk_total_gb = disk.total / (1024**3)
    disk_used_percent = (disk.used / disk.total) * 100
    logger.info(
        f"Disk: {disk_free_gb:.2f}GB free / {disk_total_gb:.2f}GB total ({disk_used_percent:.1f}% used)"
    )

    # GPU memory
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**3)
        gpu_total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(
            f"GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved / {gpu_total_mem:.2f}GB total"
        )

    # /tmp space (important for temp files)
    try:
        tmp_disk = shutil.disk_usage("/tmp")
        tmp_free_gb = tmp_disk.free / (1024**3)
        logger.info(f"/tmp space: {tmp_free_gb:.2f}GB free")
    except Exception as e:
        logger.warning(f"Could not check /tmp space: {e}")


def get_compute_type() -> str:
    """Get compute type from environment or default."""
    return os.environ.get("COMPUTE_TYPE", "float16")


def get_whisper_model_name() -> str:
    """Get Whisper model name from environment or default."""
    return os.environ.get("WHISPER_MODEL", "large-v3")


def load_models(
    device: str | None = None,
    compute_type: str | None = None,
    enable_diarization: bool = True,
) -> None:
    """Pre-load models into cache for faster inference.

    Args:
        device: Device to load models on ("cuda" or "cpu").
        compute_type: Compute type for faster-whisper ("float16", "int8", etc.).
        enable_diarization: Whether to load diarization model (default: True).
    """
    device = device or get_device()
    compute_type = compute_type or get_compute_type()
    model_name = get_whisper_model_name()

    # Log system stats before loading models
    log_system_stats()

    if "whisper" not in _model_cache:
        logger.info(f"Loading Whisper model: {model_name} on {device}")
        logger.info("Loading without VAD filter for better GPU performance...")
        _model_cache["whisper"] = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            vad_filter=False,  # Disable VAD to prevent CPU bottleneck
        )
        logger.info("Whisper model loaded")
        log_system_stats()  # Log after model load

    if enable_diarization and "diarize" not in _model_cache:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not set, diarization will be disabled")
            return

        logger.info("Loading diarization pipeline")
        try:
            _model_cache["diarize"] = DiarizationPipeline(
                use_auth_token=hf_token,
                device=device,
            )
            logger.info("Diarization pipeline loaded")
            log_system_stats()  # Log after diarization load
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            logger.warning("Continuing without diarization")


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
    diarize_pipeline = _model_cache.get("diarize")

    # Load audio
    logger.info(f"Loading audio from {audio_path}")
    log_system_stats()  # Check resources before processing
    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / 16000  # whisperx uses 16kHz

    # Transcribe
    logger.info("Running transcription")
    logger.info(f"Audio shape: {audio.shape}, duration: {duration:.2f}s")
    logger.info(f"Device: {device}, Compute type: {compute_type}")
    log_system_stats()  # Check resources before GPU inference

    transcribe_options = {
        "batch_size": 16,  # Optimal batch size for GPU utilization
    }
    if language and language != "auto":
        transcribe_options["language"] = language

    logger.info(f"Calling whisper_model.transcribe with options: {transcribe_options}")
    logger.info("Starting Whisper inference on GPU...")
    result = whisper_model.transcribe(audio, **transcribe_options)
    logger.info(
        f"Whisper transcription completed successfully - got {len(result.get('segments', []))} segments"
    )
    log_system_stats()  # Check resources after transcription

    detected_language = result.get(
        "language", language if language != "auto" else "unknown"
    )
    logger.info(
        f"Detected language: {detected_language}, segments: {len(result.get('segments', []))}"
    )

    # Align whisper output (optional for word-level timestamps)
    try:
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
        logger.info("Alignment completed")

        # Clean up align model to free memory
        del align_model
        torch.cuda.empty_cache() if device == "cuda" else None
        log_system_stats()  # Check resources after alignment
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        logger.warning("Continuing without word-level alignment")
        # Use original whisper segments without alignment
        result = {"segments": result["segments"]}

    # Diarize (if available)
    speakers = []
    if diarize_pipeline:
        logger.info("Running speaker diarization")

        diarize_kwargs = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

        try:
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
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            logger.warning("Continuing without speaker labels")
    else:
        logger.info("Diarization disabled, skipping speaker assignment")

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
