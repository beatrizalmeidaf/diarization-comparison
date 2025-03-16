"""
Funções para conversão entre diferentes formatos de dados de diarização
"""

from pyannote.core import Segment, Annotation
import logging

logger = logging.getLogger(__name__)

def convert_diar_to_annotation(diar_output):
    """
    Converte a saída do NeMo SORTFormer para o formato PyAnnote
    
    Args:
        diar_output: Saída do diarizador do NeMo
        
    Returns:
        Annotation: Objeto de anotação do PyAnnote
    """
    annotation = Annotation()
    
    try:
        if isinstance(diar_output, list):
            for segment in diar_output:
                if isinstance(segment, dict):
                    if 'start' in segment and 'end' in segment:
                        start = float(segment['start'])
                        end = float(segment['end'])
                        speaker = segment.get('speaker', f"speaker_{segment.get('speaker_id', 0)}")
                        segment_obj = Segment(start, end)
                        annotation[segment_obj] = speaker
                elif isinstance(segment, list) and len(segment) >= 3:
                    start, end, speaker_id = segment[0], segment[1], segment[2]
                    segment_obj = Segment(float(start), float(end))
                    annotation[segment_obj] = f"speaker_{speaker_id}"
                elif isinstance(segment, str):
                    # Corrigido: Tratamento adequado para strings com formato "start end speaker"
                    parts = segment.strip().split()
                    if len(parts) >= 3:
                        try:
                            start = float(parts[0])
                            end = float(parts[1])
                            # O terceiro elemento em diante forma o nome do speaker
                            speaker = " ".join(parts[2:])
                            segment_obj = Segment(start, end)
                            annotation[segment_obj] = speaker
                        except ValueError as ve:
                            logger.warning(f"Não foi possível converter valores para float: {segment}. Erro: {ve}")
        
        # Se ainda não temos anotações e diar_output é um dicionário
        if len(annotation) == 0:
            if isinstance(diar_output, dict):
                for speaker_id, segments in diar_output.items():
                    for segment in segments:
                        if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                            start, end = segment[0], segment[1]
                            segment_obj = Segment(float(start), float(end))
                            annotation[segment_obj] = f"speaker_{speaker_id}"
                        elif isinstance(segment, str):
                            # Adicionando tratamento para segmentos de string dentro de dicionários
                            parts = segment.strip().split()
                            if len(parts) >= 2:
                                try:
                                    start = float(parts[0])
                                    end = float(parts[1])
                                    segment_obj = Segment(start, end)
                                    annotation[segment_obj] = f"speaker_{speaker_id}"
                                except ValueError:
                                    logger.warning(f"Formato de segmento inválido no dicionário: {segment}")
    
    except Exception as e:
        logger.error(f"Erro na conversão do formato NeMo: {e}")
    
    return annotation

def convert_to_pyannote_format(sortformer_output, sample_rate=16000):
    """
    Converte a saída do SORTFormer (NeMo) para o formato do PyAnnote
    
    Args:
        sortformer_output: Saída do diarizador SORTFormer
        sample_rate: Taxa de amostragem do áudio
        
    Returns:
        Annotation: Objeto de anotação do PyAnnote
    """
    return convert_diar_to_annotation(sortformer_output)
