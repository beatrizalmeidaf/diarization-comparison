"""
Inicialização do pacote de processadores
"""

from processors.audio_processor import AudioProcessor
from processors.transcription import TranscriptionProcessor
from processors.evaluation import EvaluationProcessor

__all__ = ['AudioProcessor', 'TranscriptionProcessor', 'EvaluationProcessor']