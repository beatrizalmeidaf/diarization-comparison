"""
Processador de áudio para operações básicas
"""

import soundfile as sf
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Classe para processar arquivos de áudio"""
    
    @staticmethod
    def load_audio(audio_path):
        """
        Carrega um arquivo de áudio
        
        Args:
            audio_path: Caminho do arquivo de áudio
            
        Returns:
            tuple: (audio_data, sample_rate) ou (None, None) em caso de erro
        """
        try:
            audio, sr = sf.read(audio_path)
            return audio, sr
        except Exception as e:
            logger.error(f"Erro ao carregar áudio {audio_path}: {e}")
            return None, None
            
    @staticmethod
    def extract_segment(audio_data, sample_rate, start_time, end_time):
        """
        Extrai um segmento de áudio
        
        Args:
            audio_data: Dados do áudio
            sample_rate: Taxa de amostragem
            start_time: Tempo de início em segundos
            end_time: Tempo de fim em segundos
            
        Returns:
            ndarray: Dados do segmento de áudio
        """
        try:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            if end_sample > len(audio_data):
                end_sample = len(audio_data)
                
            if start_sample >= end_sample:
                return None
                
            return audio_data[start_sample:end_sample]
        except Exception as e:
            logger.error(f"Erro ao extrair segmento de áudio: {e}")
            return None
            
    @staticmethod
    def save_segment(audio_data, sample_rate, output_path=None):
        """
        Salva um segmento de áudio em um arquivo temporário ou em um caminho especificado
        
        Args:
            audio_data: Dados do áudio
            sample_rate: Taxa de amostragem
            output_path: Caminho de saída (opcional)
            
        Returns:
            str: Caminho do arquivo salvo
        """
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_path = temp_file.name
                
        try:
            sf.write(output_path, audio_data, sample_rate)
            return output_path
        except Exception as e:
            logger.error(f"Erro ao salvar segmento de áudio: {e}")
            return None
            
    @staticmethod
    def cleanup_temp_file(file_path):
        """
        Remove um arquivo temporário
        
        Args:
            file_path: Caminho do arquivo a ser removido
        """
        try:
            os.remove(file_path)
        except Exception as e:
            logger.error(f"Erro ao remover arquivo temporário {file_path}: {e}")