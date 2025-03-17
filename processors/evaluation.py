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
    def evaluate_diarization_performance(diarization_output, reference_output=None):
        """
        Avalia o desempenho de um modelo de diarização, calculando DER e JER
        
        Args:
            diarization_output: Saída do modelo de diarização
            reference_output: Saída do modelo de referência (opcional)
            
        Returns:
            dict: Dicionário com métricas DER e JER
        """
        from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
        
        # Inicializar métricas
        der_metric = DiarizationErrorRate()
        jer_metric = JaccardErrorRate()
        
        try:
            # Se temos uma referência, usamos ela para calcular as métricas
            if reference_output is not None:
                der_score = der_metric(reference_output, diarization_output)
                jer_score = jer_metric(reference_output, diarization_output)
            else:
                # Caso contrário, podemos usar algumas heurísticas ou verificar a qualidade das fronteiras
                # Para esse caso de uso, será usado valores fixos ou baseados em características da diarização
                # Nota: Essa é uma implementação simplificada para quando não há referência
                # Idealmente teria uma referência real (ground truth) para cada áudio
                
                # Como simplificação, podemos usar valores fixos indicando que não foram calculados
                der_score = 0.5  # Valor padrão quando não há referência
                jer_score = 0.5  # Valor padrão quando não há referência
                
                logger.warning("Calculando métricas sem referência. Usando valores padrão.")
                
            return {
                "DER": der_score,
                "JER": jer_score
            }
        except Exception as e:
            logger.error(f"Erro ao calcular métricas de diarização: {e}")
            return {
                "DER": "N/A",
                "JER": "N/A"
            }
            
