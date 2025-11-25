# AGENTS.md

## Project Overview
Python RunPod serverless handler for WhisperX audio transcription with speaker diarization.

## Build & Run
- **Install**: `pip install -r requirements.txt`
- **Run locally**: `python src/handler.py` (requires GPU + HF_TOKEN env var)
- **Test**: `pytest tests/ -v` | Single test: `pytest tests/test_file.py::test_name -v`
- **Docker build**: `docker build -t hearback .`

## Code Style
- **Python**: 3.10+, type hints required on all functions
- **Formatting**: Use `ruff format` (line length 88)
- **Linting**: Use `ruff check --fix`
- **Imports**: stdlib, third-party, local (separated by blank lines), sorted alphabetically
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Error handling**: Use specific exceptions, log errors before re-raising
- **Docstrings**: Required for public functions (Google style)

## Architecture
- `src/handler.py` - RunPod serverless entry point
- `src/transcribe.py` - WhisperX transcription logic
- `src/utils.py` - Audio download, output formatting helpers

## Environment Variables
- `HF_TOKEN` (required) - Hugging Face token for pyannote models
- `WHISPER_MODEL` - Model name (default: large-v3)
- `COMPUTE_TYPE` - float16 or int8 (default: float16)
