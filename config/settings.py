"""
Configurações globais para o pipeline de comparação de diarização
"""

import os
import glob

# Diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Configurações de modelos
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.0"
PYANNOTE_AUTH_TOKEN = "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" 

ASR_MODEL = "openai/whisper-small"

# Configurações de processamento
SAMPLE_RATE = 16000

# Lista de arquivos de áudio para processamento
AUDIO_FILES = glob.glob("audios/*.wav")
