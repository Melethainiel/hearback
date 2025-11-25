# Hearback

Service de transcription audio self-hosted avec identification des locuteurs, déployé sur RunPod en mode serverless.

## Features

- 🎯 Transcription précise avec WhisperX (large-v3)
- 👥 Diarization (identification des locuteurs) avec pyannote-audio
- ⏱️ Timestamps word-level précis
- 🌍 Support FR/EN (extensible)
- 📝 Export JSON, SRT, VTT
- 🚀 Serverless RunPod (paiement à la seconde)

## Quick Start - RunPod Pod (Dev)

```bash
# Clone le repo
git clone git@github.com:Melethainiel/hearback.git
cd hearback

# Set cuDNN library path (REQUIS pour éviter segfaults)
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Install dependencies
pip install -r requirements.txt

# Set Hugging Face token (requis pour diarization)
export HF_TOKEN=your_hf_token_here

# Run handler
python src/handler.py --rp_serve_api --rp_api_host 0.0.0.0
```

## Usage

```bash
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/audio.mp3",
      "language": "fr",
      "min_speakers": 2,
      "max_speakers": 4,
      "output_format": "json"
    }
  }'
```

### Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio_url` | string | ✅ | - | URL du fichier audio |
| `language` | string | ❌ | "auto" | Code langue: "fr", "en", "auto" |
| `min_speakers` | int | ❌ | null | Nombre minimum de locuteurs |
| `max_speakers` | int | ❌ | null | Nombre maximum de locuteurs |
| `output_format` | string | ❌ | "json" | Format: "json", "srt", "vtt" |

### Output Format (JSON)

```json
{
  "transcription": {
    "text": "Transcription complète...",
    "segments": [
      {
        "start": 0.0,
        "end": 2.5,
        "text": "Bonjour, je suis Pierre.",
        "speaker": "SPEAKER_00",
        "words": [
          {"word": "Bonjour", "start": 0.0, "end": 0.4},
          {"word": "je", "start": 0.5, "end": 0.6}
        ]
      }
    ]
  },
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "language": "fr",
  "duration": 260.28,
  "processing_time": 45.3
}
```

## Déploiement RunPod Serverless

### 1. Build & Push Docker Image

```bash
# Build
docker build -t your-dockerhub-username/hearback:latest .

# Login Docker Hub
docker login

# Push
docker push your-dockerhub-username/hearback:latest
```

### 2. Créer Endpoint RunPod

1. Aller sur [RunPod Console](https://runpod.io/console/serverless)
2. **New Endpoint** > **Custom**
3. Configuration:
   - **Container Image**: `your-dockerhub-username/hearback:latest`
   - **GPU**: RTX 4000 Ada ou RTX 3090
   - **Environment Variables**:
     - `HF_TOKEN`: Votre token Hugging Face
     - `WHISPER_MODEL`: `large-v3` (optionnel)
     - `COMPUTE_TYPE`: `float16` (optionnel)

### 3. Utiliser l'Endpoint

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/runsync \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/audio.mp3",
      "language": "fr"
    }
  }'
```

## Prérequis Hugging Face

Accepter les conditions d'utilisation pour:
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

Puis créer un token (read) sur [Hugging Face Settings](https://huggingface.co/settings/tokens).

## Troubleshooting

### Segmentation fault (core dumped)

**Cause**: Incompatibilité cuDNN 9.x avec faster-whisper.

**Solution**: Set `LD_LIBRARY_PATH` avant de lancer:

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
python src/handler.py
```

Pour Docker, c'est déjà configuré dans le Dockerfile.

### Out of Memory

Réduire le modèle ou le compute type:

```bash
export WHISPER_MODEL=medium
export COMPUTE_TYPE=int8
```

## Coûts Estimés (RunPod Serverless)

| Durée audio | Temps GPU | Coût RTX 4000 |
|-------------|-----------|---------------|
| 30 min | ~3-5 min | ~0.10-0.15$ |
| 1h | ~5-10 min | ~0.15-0.30$ |
| 2h | ~10-20 min | ~0.30-0.60$ |

**Usage mensuel estimé**: 10 réunions × 1h = ~2-5$/mois

## Structure du Projet

```
hearback/
├── Dockerfile              # Image RunPod
├── requirements.txt        # Dependencies Python
├── src/
│   ├── handler.py          # RunPod serverless entry point
│   ├── transcribe.py       # WhisperX pipeline
│   └── utils.py            # Helpers (download, format)
├── test_input.json         # Payload de test
├── SPECS.md                # Spécifications détaillées
└── README.md
```

## License

BSD-2-Clause (voir WhisperX)

## Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperX)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [pyannote-audio](https://github.com/pyannote/pyannote-audio)
