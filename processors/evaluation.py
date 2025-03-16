"""
Funções de avaliação para comparar resultados de diarização
"""

import logging
from jiwer import wer, cer

logger = logging.getLogger(__name__)

class EvaluationProcessor:
    """Classe para avaliação de resultados de diarização e transcrição"""
    
    @staticmethod
    def evaluate_diarization(reference, hypothesis):
        """
        Calcula métricas de avaliação entre a referência e a hipótese
        
        Args:
            reference: Anotação de referência (PyAnnote)
            hypothesis: Anotação de hipótese (PyAnnote)
            
        Returns:
            dict: Dicionário com a métrica DER
        """
        from pyannote.metrics.diarization import DiarizationErrorRate
        
        der = DiarizationErrorRate()
        try:
            der_score = der(reference, hypothesis)
            return {"DER": der_score}
        except Exception as e:
            logger.error(f"Erro ao calcular DER: {e}")
            return {"DER": "N/A"}
            
    @staticmethod
    def calculate_wer_cer(reference_transcription, hypothesis_transcription):
        """
        Calcula as métricas WER e CER entre a transcrição de referência e a hipótese
        
        Args:
            reference_transcription: Transcrição de referência
            hypothesis_transcription: Transcrição de hipótese
            
        Returns:
            dict: Dicionário com métricas WER e CER
        """
        if not reference_transcription or not hypothesis_transcription:
            return {"WER": None, "CER": None}
        
        try:
            wer_score = wer(reference_transcription, hypothesis_transcription)
            cer_score = cer(reference_transcription, hypothesis_transcription)
            return {"WER": wer_score, "CER": cer_score}
        except Exception as e:
            logger.error(f"Erro ao calcular WER/CER: {e}")
            return {"WER": None, "CER": None}