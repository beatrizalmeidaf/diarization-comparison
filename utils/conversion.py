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
        # Caso 1: Lista de listas de strings 
        if (isinstance(diar_output, list) and len(diar_output) > 0 and 
            isinstance(diar_output[0], list) and len(diar_output[0]) > 0 and 
            isinstance(diar_output[0][0], str)):
            
            # Processar cada string na lista interna
            for segment_list in diar_output:
                for segment_str in segment_list:
                    try:
                        # Dividir a string em componentes
                        parts = segment_str.strip().split()
                        if len(parts) >= 3:
                            start = float(parts[0])
                            end = float(parts[1])
                            speaker = " ".join(parts[2:])  # Juntar tudo depois do segundo elemento
                            segment_obj = Segment(start, end)
                            annotation[segment_obj] = speaker
                    except ValueError as ve:
                        logger.warning(f"Erro ao processar segmento: {segment_str}. Erro: {ve}")
        
        # Caso 2: Lista de strings ou dicionários
        elif isinstance(diar_output, list):
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
                    parts = segment.strip().split()
                    if len(parts) >= 3:
                        try:
                            start = float(parts[0])
                            end = float(parts[1])
                            speaker = " ".join(parts[2:])
                            segment_obj = Segment(start, end)
                            annotation[segment_obj] = speaker
                        except ValueError as ve:
                            logger.warning(f"Erro ao processar string: {segment}. Erro: {ve}")
        
        # Caso 3: Dicionário
        elif isinstance(diar_output, dict):
            for speaker_id, segments in diar_output.items():
                for segment in segments:
                    if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                        start, end = segment[0], segment[1]
                        segment_obj = Segment(float(start), float(end))
                        annotation[segment_obj] = f"speaker_{speaker_id}"
                    elif isinstance(segment, str):
                        parts = segment.strip().split()
                        if len(parts) >= 2:
                            try:
                                start = float(parts[0])
                                end = float(parts[1])
                                segment_obj = Segment(start, end)
                                annotation[segment_obj] = f"speaker_{speaker_id}"
                            except ValueError:
                                logger.warning(f"Formato de segmento inválido: {segment}")
        
        # Caso 4: String única
        elif isinstance(diar_output, str):
            lines = diar_output.strip().split('\n')
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        speaker = " ".join(parts[2:])
                        segment_obj = Segment(start, end)
                        annotation[segment_obj] = speaker
                    except ValueError:
                        logger.warning(f"Não foi possível converter para float: {line}")
    
    except Exception as e:
        logger.error(f"Erro na conversão do formato NeMo: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # Verificar se foram criadas anotações
    if len(annotation) == 0:
        logger.warning(f"Nenhuma anotação foi criada a partir da entrada: {repr(diar_output)[:100]}...")
    
    return annotation

def debug_diar_format(diar_output):
    """
    Função para ajudar a debugar o formato da saída de diarização
    
    Args:
        diar_output: Saída do diarizador do NeMo
    
    Returns:
        str: Informações de debug
    """
    import json
    result = f"Tipo: {type(diar_output)}\n"
    
    if isinstance(diar_output, str):
        result += f"String: {diar_output}\n"
    elif isinstance(diar_output, list):
        result += f"Lista com {len(diar_output)} elementos\n"
        for i, item in enumerate(diar_output[:3]):  # Primeiros 3 elementos
            result += f"Item {i}: {type(item)} - {repr(item)}\n"
    elif isinstance(diar_output, dict):
        result += f"Dicionário com chaves: {list(diar_output.keys())}\n"
    
    try:
        # Tentar serializar para JSON para ver a estrutura completa
        json_str = json.dumps(diar_output, indent=2)
        result += f"JSON:\n{json_str[:500]}...\n"  
    except:
        result += "Não foi possível converter para JSON\n"
    
    return result

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
