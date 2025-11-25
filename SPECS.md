# Hearback - Spécifications RunPod

## Objectif

Service de transcription audio self-hosted avec identification des locuteurs, déployé sur RunPod en mode serverless pour un usage à la demande (réunions, sessions D&D, etc.).

## Stack technique

### Choix : WhisperX

WhisperX est retenu plutôt que Whisper + Pyannote séparés car il intègre nativement :

| Fonctionnalité | Description |
|----------------|-------------|
| ASR | faster-whisper sous le capot (CTranslate2, optimisé GPU) |
| Diarization | pyannote-audio intégré |
| Alignement | wav2vec2 pour timestamps word-level |
| VAD | Détection d'activité vocale (réduit les hallucinations) |
| Performance | ~70x real-time speedup |

**Limitations connues :**
- Fichiers très courts (<1s) : résultats dégradés
- Fichiers longs (>10min) : risque d'hallucinations (répétitions)
- Locuteurs similaires : diarization moins fiable
- Chevauchement de parole : non géré

### Langues cibles

- Français (prioritaire)
- Anglais

### Modèle Whisper

`large-v3` pour la meilleure précision FR/EN.

## Architecture RunPod

### Mode de déploiement

**Serverless** (facturation à la seconde)

- Cold start : ~15-30s (chargement modèles)
- Pas de coûts au repos
- Scale automatique

### GPU recommandé

| GPU | VRAM | Prix serverless | Adapté ? |
|-----|------|-----------------|----------|
| RTX 4000 Ada | 20GB | ~0.0005$/s | ✅ Optimal |
| RTX 3090 | 24GB | ~0.0004$/s | ✅ Bon |
| RTX A4000 | 16GB | ~0.0003$/s | ⚠️ Limite |
| A100 / H100 | 40-80GB | ~0.002$/s | ❌ Overkill |

**Choix recommandé : RTX 4000 Ada ou RTX 3090**

## API

### Endpoint

```
POST /transcribe
```

### Input

```json
{
  "audio_url": "https://...",  // URL du fichier audio
  "language": "fr",            // "fr" | "en" | "auto"
  "min_speakers": 2,           // optionnel
  "max_speakers": 6,           // optionnel
  "output_format": "json"      // "json" | "srt" | "vtt"
}
```

### Output

```json
{
  "transcription": {
    "text": "Bonjour, je suis...",
    "segments": [
      {
        "start": 0.0,
        "end": 2.5,
        "text": "Bonjour, je suis Pierre.",
        "speaker": "SPEAKER_00",
        "words": [
          {"word": "Bonjour", "start": 0.0, "end": 0.4},
          {"word": "je", "start": 0.5, "end": 0.6},
          ...
        ]
      }
    ]
  },
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "language": "fr",
  "duration": 3600.5,
  "processing_time": 245.3
}
```

## Intégration n8n

### Workflow type

```
[Webhook/Trigger] 
    → [Upload audio vers stockage temp]
    → [POST RunPod endpoint]
    → [Poll status jusqu'à complétion]
    → [Récupérer résultat]
    → [Post-traitement / sauvegarde]
```

### Stockage temporaire audio

Options :
- MinIO (self-hosted)
- Cloudflare R2
- URL signée temporaire

## Estimation des coûts

### Temps de traitement estimé

| Durée audio | Temps GPU (estimé) | Coût RTX 4000 |
|-------------|-------------------|---------------|
| 30 min | ~3-5 min | ~0.10-0.15$ |
| 1h | ~5-10 min | ~0.15-0.30$ |
| 2h | ~10-20 min | ~0.30-0.60$ |

### Usage mensuel estimé

```
10 réunions × 1h × 0.20$/transcription = ~2$/mois
+ cold starts occasionnels
≈ 3-5$/mois
```

## Structure du projet

```
hearback-runpod/
├── Dockerfile
├── src/
│   ├── handler.py          # RunPod handler principal
│   ├── transcribe.py       # Logique WhisperX
│   └── utils.py            # Helpers (download, format output)
├── requirements.txt
├── test_input.json         # Pour tests locaux
└── README.md
```

## Dépendances

```txt
runpod
torch
faster-whisper
whisperx
pyannote.audio
ffmpeg-python
```

## Variables d'environnement

| Variable | Description | Obligatoire |
|----------|-------------|-------------|
| `HF_TOKEN` | Token Hugging Face (pour pyannote) | ✅ |
| `WHISPER_MODEL` | Modèle à utiliser (default: large-v3) | ❌ |
| `COMPUTE_TYPE` | float16 / int8 (default: float16) | ❌ |

## Prérequis Hugging Face

Accepter les conditions d'utilisation pour :
- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`

## Roadmap

### Phase 1 - MVP
- [ ] Image Docker fonctionnelle
- [ ] Handler RunPod basique
- [ ] Test local avec GPU
- [ ] Déploiement serverless

### Phase 2 - Intégration
- [ ] Workflow n8n complet
- [ ] Stockage audio (MinIO/R2)
- [ ] Webhook de callback

### Phase 3 - Améliorations
- [ ] Cache des modèles (réduire cold start)
- [ ] Support formats multiples (mp3, wav, m4a, opus)
- [ ] Post-traitement LLM (résumé, action items)
- [ ] Speaker naming (identification persistante)

## Références

- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [RunPod Serverless Docs](https://docs.runpod.io/serverless)
- [Pyannote Speaker Diarization](https://huggingface.co/pyannote/speaker-diarization-3.1)
