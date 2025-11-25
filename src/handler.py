"""RunPod serverless handler for audio transcription."""

import logging
import os
import sys
import time
from typing import Any

# Fix cuDNN 9.x incompatibility by forcing cuDNN 8.x from pip
# See: https://github.com/m-bain/whisperX/issues/902#issuecomment-2433187969
try:
    import nvidia.cudnn

    cudnn_path = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib")
    if os.path.exists(cudnn_path):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{cudnn_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        )
        logging.info(f"Set LD_LIBRARY_PATH to use cuDNN from: {cudnn_path}")
except ImportError:
    pass

import runpod

from transcribe import load_models, transcribe_audio
from utils import cleanup_temp_files, convert_to_wav, download_audio, format_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Valid values for input parameters
VALID_LANGUAGES = {"fr", "en", "auto"}
VALID_OUTPUT_FORMATS = {"json", "srt", "vtt"}


def validate_input(job_input: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize job input.

    Args:
        job_input: Raw input from RunPod job.

    Returns:
        Validated and normalized input dict.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if not job_input:
        raise ValueError("No input provided")

    audio_url = job_input.get("audio_url")
    if not audio_url:
        raise ValueError("audio_url is required")

    language = job_input.get("language", "auto")
    if language not in VALID_LANGUAGES:
        raise ValueError(
            f"Invalid language: {language}. Must be one of {VALID_LANGUAGES}"
        )

    output_format = job_input.get("output_format", "json")
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"Invalid output_format: {output_format}. Must be one of {VALID_OUTPUT_FORMATS}"
        )

    min_speakers = job_input.get("min_speakers")
    max_speakers = job_input.get("max_speakers")

    if min_speakers is not None and not isinstance(min_speakers, int):
        raise ValueError("min_speakers must be an integer")
    if max_speakers is not None and not isinstance(max_speakers, int):
        raise ValueError("max_speakers must be an integer")
    if min_speakers is not None and max_speakers is not None:
        if min_speakers > max_speakers:
            raise ValueError("min_speakers cannot be greater than max_speakers")

    return {
        "audio_url": audio_url,
        "language": language,
        "output_format": output_format,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }


def handler(job: dict[str, Any]) -> dict[str, Any] | str:
    """RunPod serverless handler function.

    Args:
        job: RunPod job containing input parameters.

    Returns:
        Transcription result in requested format.
    """
    start_time = time.time()
    audio_path = None
    wav_path = None

    try:
        # Validate input
        job_input = job.get("input", {})
        validated = validate_input(job_input)

        logger.info(f"Processing job with input: {validated}")

        # Download audio
        logger.info(f"Downloading audio from {validated['audio_url']}")
        audio_path = download_audio(validated["audio_url"])
        logger.info(f"Audio downloaded to {audio_path}")

        # Convert to WAV for better compatibility
        wav_path = convert_to_wav(audio_path)

        # Transcribe
        result = transcribe_audio(
            audio_path=wav_path,
            language=validated["language"],
            min_speakers=validated["min_speakers"],
            max_speakers=validated["max_speakers"],
        )

        processing_time = time.time() - start_time
        logger.info(f"Transcription complete in {processing_time:.2f}s")

        # Format output
        output = format_output(
            segments=result["segments"],
            speakers=result["speakers"],
            language=result["language"],
            duration=result["duration"],
            processing_time=processing_time,
            output_format=validated["output_format"],
        )

        return output

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.exception(f"Transcription failed: {e}")
        return {"error": f"Transcription failed: {str(e)}"}
    finally:
        # Cleanup temp files
        cleanup_temp_files(audio_path, wav_path)


# Pre-load models on cold start
logger.info("Pre-loading models...")
try:
    load_models()
    logger.info("Models pre-loaded successfully")
except Exception as e:
    logger.error(f"Failed to pre-load models: {e}")

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
